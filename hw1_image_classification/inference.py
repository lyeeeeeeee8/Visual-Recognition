import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from model_resnet import CustomResNet

##----------------- Set filename -----------------
model_num = 101
weight_filename = "./weight/res4_360_" + str(model_num) + ".pth"
output_filename = "./prediction/res4_360_tta" + str(model_num) + ".csv"

##----------------- Load Model and Data -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomResNet(num_classes=100, pretrained=False).to(device)
model.load_state_dict(torch.load(weight_filename, weights_only=True))
model.eval()

DATA_DIR = "/home/hscc/EN/hw1/data/"
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=None)
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

##----------------- TTA Transform -----------------
IMG_SIZE = 360 #####
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
flip_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=1.0), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
tta_transforms = [base_transform, flip_transform]

##----------------- TTA Inference -----------------
test_dir = os.path.join(DATA_DIR, "test")
test_files = [
    os.path.join(test_dir, f) 
    for f in os.listdir(test_dir) 
    if f.lower().endswith(".jpg")
]
results = []
with torch.no_grad():
    for img_path in test_files:
        img = Image.open(img_path).convert("RGB")
        preds = []
        for t in tta_transforms:
            transformed_img = t(img).unsqueeze(0).to(device) 
            output = model(transformed_img)
            prob = torch.softmax(output, dim=1)
            preds.append(prob)
        avg_pred = torch.mean(torch.stack(preds), dim=0)
        _, predicted = torch.max(avg_pred, 1)
    
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0]
        pred_label = idx_to_class[predicted.item()]
        results.append((img_name, pred_label))


##----------------- Save Prediction -----------------
df = pd.DataFrame(results, columns=["image_name", "pred_label"])
df.to_csv(output_filename, index=False)
print("TTA Inference completed!\nFile saved as:", output_filename)
