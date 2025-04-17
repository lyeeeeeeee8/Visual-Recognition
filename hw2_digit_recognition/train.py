import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# Custom Compose (applies transforms to both image and target)
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# Custom Resize transform
# 1. Resizes the shorter side of the image to `min_size` while maintaining aspect ratio.
# 2. Adjusts bounding box coordinates in the target accordingly.
class Resize(object):
    def __init__(self, min_size=256):
        self.min_size = min_size

    def __call__(self, image, target):
        w, h = image.size
        scale = self.min_size / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)

        # Adjust bounding boxes
        for ann in target:
            x, y, bw, bh = ann['bbox']
            ann['bbox'] = [x * scale, y * scale, bw * scale, bh * scale]
        return image, target


# Custom ColorJitter transform (applies only to the image)
class ColorJitterTransform(object):
    def __init__(self, **kwargs):
        self.transform = T.ColorJitter(**kwargs)

    def __call__(self, image, target):
        image = self.transform(image)
        return image, target


# Custom ToTensor transform (converts image to tensor, leaves target unchanged)
class ToTensorTransform(object):
    def __call__(self, image, target):
        image = T.ToTensor()(image)
        return image, target


# Define transformations for training and validation
def get_transform(train):
    if train:
        return Compose([
            Resize(min_size=256),
            ColorJitterTransform(contrast=0.3),
            ToTensorTransform()
        ])
    else:
        return Compose([
            Resize(min_size=256),
            ToTensorTransform()
        ])


# Custom Dataset: Inherits from CocoDetection and applies transforms
class CocoDetectionWithAugment(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


# Get Dataset
def get_dataset(img_dir, ann_file, train):
    transforms = get_transform(train)
    dataset = CocoDetectionWithAugment(root=img_dir, annFile=ann_file, transforms=transforms)
    return dataset


# Convert target bounding boxes to [x_min, y_min, x_max, y_max] format
def convert_targets(annotations):
    boxes = []
    labels = []
    for ann in annotations:
        x_min, y_min, w, h = ann['bbox']
        boxes.append([x_min, y_min, x_min + w, y_min + h])
        labels.append(ann['category_id'])
    return {
        'boxes': torch.as_tensor(boxes, dtype=torch.float32),
        'labels': torch.as_tensor(labels, dtype=torch.int64)
    }


# Collate function: Prepares a batch of data
def collate_fn(batch):
    images, annotations = list(zip(*batch))
    targets = [convert_targets(ann) for ann in annotations]
    return list(images), targets


# Create DataLoader
def get_dataloaders(
    train_img_dir, train_ann_file, valid_img_dir, valid_ann_file,
    batch_size=8, num_workers=4
):
    dataset_train = get_dataset(train_img_dir, train_ann_file, train=True)
    dataset_valid = get_dataset(valid_img_dir, valid_ann_file, train=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    return data_loader_train, data_loader_valid

#-------------------------------------------------------------------------------------------------
# Validation function: Computes average loss
def evaluate_validloss(model, data_loader, device):
    model.train()  # Must be in train mode to return loss
    total_loss = 0.0
    count = 0
    for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            count += 1
    return total_loss / count if count > 0 else 0.0


def evaluate_mAP(model, dataset, device, ann_file):
    model.eval()
    predictions_list = []
    for i in tqdm(range(len(dataset)), desc="Evaluating mAP"):
        image, target = dataset[i]
        # 取得 image_id 從 ground truth target 中 (若沒有，則使用 i)
        if len(target) > 0 and "image_id" in target[0]:
            image_id = target[0]["image_id"]
        else:
            image_id = i
        image = image.to(device)
        with torch.no_grad():
            pred = model([image])[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            prediction = {
                "image_id": image_id,
                "bbox": [float(x_min), float(y_min), float(w), float(h)],
                "score": float(score),
                "category_id": int(label)
            }
            predictions_list.append(prediction)
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(predictions_list)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP = coco_eval.stats[0]
    return mAP, predictions_list


#-------------------------------------------------------------------------------------------------
# Training loop: Uses OneCycleLR and saves checkpoints 
def train_model(
    model, optimizer, scheduler, data_loader_train, data_loader_valid,
    device, num_epochs=16
):
    best_weights = []  # Stores (val_loss, epoch, file_path)
    train_losses = []
    val_losses = []

    total_steps = len(data_loader_train) * num_epochs  # For OneCycleLR

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0

        # Training loop
        for images, targets in tqdm(
            data_loader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_sum += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()  # Update scheduler per batch

        # Evaluate training loss
        avg_train_loss = loss_sum / len(data_loader_train)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

        # Evaluate validation loss
        val_loss = evaluate_validloss(model, data_loader_valid, device)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        model_file = f"best_fasterrcnn_epoch{epoch+1}_loss_{val_loss:.4f}.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Model checkpoint saved: {model_file}")
        
        # Update best_weights (keep top 3)
        best_weights.append((val_loss, epoch + 1, model_file))
        best_weights.sort(key=lambda x: x[0])
        if len(best_weights) > 3:
            removed = best_weights.pop()
            print(f"Removed checkpoint: {removed[2]}")

    print("Top 3 best model checkpoints:")
    for loss_val, epoch_num, file in best_weights:
        print(f"Epoch {epoch_num}: Loss {loss_val:.4f}, File: {file}")

    return train_losses, val_losses


# Plot loss curve
def plot_loss_curve(train_losses, val_losses, num_epochs, save_path='loss_curve.png'):
    plt.figure()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)


#-------------------------------------------------------------------------------------------------
# Main function: Sets up device, data, model, optimizer, scheduler, and starts training
def main():
    gpu_id = 1
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    data_dir = "dataset"
    train_img_dir = os.path.join(data_dir, "train")
    train_ann_file = os.path.join(data_dir, "train.json")
    valid_img_dir = os.path.join(data_dir, "valid")
    valid_ann_file = os.path.join(data_dir, "valid.json")

    data_loader_train, data_loader_valid = get_dataloaders(
        train_img_dir, train_ann_file, valid_img_dir, valid_ann_file,
        batch_size=8, num_workers=4
    )

    print("Building model with ResNet101 backbone...")
    num_classes = 11  # Background + 10 classes (0-9)
    backbone = resnet_fpn_backbone(
        backbone_name='resnet101', weights=ResNet101_Weights.DEFAULT
    )
    model = FasterRCNN(backbone, num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    num_epochs = 22
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.005, steps_per_epoch=len(data_loader_train),
        epochs=num_epochs
    )

    print("Start training...")
    train_losses, val_losses = train_model(
        model, optimizer, scheduler, data_loader_train, data_loader_valid,
        device, num_epochs=num_epochs
    )

    plot_loss_curve(train_losses, val_losses, num_epochs, save_path='loss_curve.png')

    print("Training complete.")


if __name__ == '__main__':
    main()