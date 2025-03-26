import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

## ----------------- Custom ResNet -----------------
class CustomResNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(CustomResNet, self).__init__()
        if pretrained:
            # self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
            # self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            # self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            # self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            
        else:
            # self.resnet = models.resnet18(weights=None)
            # self.resnet = models.resnet34(weights=None)
            # self.resnet = models.resnet50(weights=None)
            self.resnet = models.resnet101(weights=None)
            # self.resnet = models.resnet152(weights=None)
            
        in_features = self.resnet.fc.in_features

        ### Change the final layer 
        hidden_dim = 1024  
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        

    def forward(self, x):
        return self.resnet(x)

## ----------------- Testing the model -----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=100).to(device)
    summary(model, input_size=(1, 3, 224, 224))
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    print(f"shape of output: {output.shape}")  
