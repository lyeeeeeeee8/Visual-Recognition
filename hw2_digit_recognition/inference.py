import os
import json
import csv
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Custom collate function for test data to handle variable image sizes
def test_collate_fn(batch):
    images, image_ids = zip(*batch)
    return list(images), list(image_ids)


# Custom dataset for loading test images
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Get files with extensions jpg, jpeg, or png
        self.image_files = sorted(
            [f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        # Remove file extension and try to convert to int; fallback to string
        base_name = os.path.splitext(img_name)[0]
        try:
            image_id = int(base_name)
        except ValueError:
            image_id = base_name
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_id


# Define test transformations (only ToTensor)
def get_test_transform():
    return T.Compose([T.ToTensor()])


# Load the trained Faster R-CNN model with ResNet101 backbone
def load_model(device):
    print("Loading model with ResNet101 backbone...")

    num_classes = 11  # Background + 10 digits (0-9)

    # Build ResNet101 + FPN backbone with pretrained weights
    backbone = resnet_fpn_backbone(
        backbone_name='resnet101',
        weights=ResNet101_Weights.DEFAULT  # Pretrained on ImageNet
    )

    # Create Faster R-CNN model with custom backbone
    model = FasterRCNN(backbone, num_classes=num_classes)

    # Load the weights saved during training
    weight_path = "weight/best_fasterrcnn.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.to(device)
    model.eval()
    return model


#-------------------------------------------------------------------------------------------------
# Perform inference and post-processing
def inference(model, dataloader, device, score_threshold=0.5):
    """
    - Task 1: Generate predictions for each bounding box.
    - Task 2: Generate final digit results for each image.
    """
    task1_results = []  # Store Task 1 predictions (list of dicts)
    task2_results = {}  # Store Task 2 results with image_id as key

    with torch.no_grad():
        for images, image_ids in tqdm(dataloader, desc="Inferring"):
            images = [img.to(device) for img in images]
            predictions = model(images)
            for pred, image_id in zip(predictions, image_ids):
                boxes = pred['boxes'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()

                # Filter predictions with low confidence (score < threshold)
                valid_idx = scores >= score_threshold
                boxes = boxes[valid_idx]
                labels = labels[valid_idx]
                scores = scores[valid_idx]

                # Convert bounding boxes to [x_min, y_min, w, h]
                coco_boxes = []
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min
                    coco_boxes.append([float(x_min), float(y_min), float(w), float(h)])

                # Task 1: Generate predictions for each bounding box
                for bbox, label, score in zip(coco_boxes, labels, scores):
                    pred_entry = {
                        "image_id": image_id,
                        "bbox": bbox,
                        "score": float(score),
                        "category_id": int(label)
                    }
                    task1_results.append(pred_entry)

                # Task 2: If no valid predictions, mark as "-1"
                if len(coco_boxes) == 0:
                    task2_results[image_id] = "-1"
                else:
                    preds = list(zip(coco_boxes, labels))
                    preds.sort(key=lambda x: x[0][0])  # Sort by x_min
                    # Convert category_id to digit: digit = category_id - 1
                    digit_str = "".join([str(int(label) - 1) for _, label in preds])
                    task2_results[image_id] = int(digit_str)

    # Sort task1_results by image_id
    task1_results = sorted(task1_results, key=lambda x: x["image_id"])
    # Sort task2_results by image_id
    sorted_task2 = {k: task2_results[k] for k in sorted(task2_results)}
    return task1_results, sorted_task2


# Save inference results to JSON and CSV files
def save_results(task1_results, task2_results, json_path="pred.json", csv_path="pred.csv"):
    # Write pred.json (Task 1)
    with open(json_path, "w") as f:
        json.dump(task1_results, f, indent=4)

    # Write pred.csv (Task 2), columns: image_id, pred_label
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "pred_label"])
        for image_id in sorted(task2_results):
            writer.writerow([image_id, task2_results[image_id]])

    print(f"Results saved: {json_path}, {csv_path}")


#-------------------------------------------------------------------------------------------------
# Main function to perform inference
def main():
    gpu_id = 1
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Path to test dataset, assumed to be in dataset/test
    test_dir = os.path.join("dataset", "test")

    transform = get_test_transform()
    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=test_collate_fn
    )

    print("Loading model...")
    model = load_model(device)

    print("Starting inference...")
    task1_results, task2_results = inference(
        model, test_loader, device, score_threshold=0.7
    )

    save_results(
        task1_results,
        task2_results,
        json_path="./prediction/pred.json",
        csv_path="./prediction/pred.csv"
    )
    print("Inference complete.")


if __name__ == '__main__':
    main()