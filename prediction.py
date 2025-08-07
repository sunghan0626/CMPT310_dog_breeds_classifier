import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import os
import sys

MODEL_PATH = "best_model.pth"
DATA_DIR = "data/Processed/train"
IMG_PATH = "sample_dog.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load class labels
class_names = sorted(os.listdir(DATA_DIR))

# Image preprocessing
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def get_resnet_model(num_classes):
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
    return model

# Load model
model = get_resnet_model(num_classes=len(class_names)).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted = outputs.argmax(1).item()
    return class_names[predicted]

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMG_PATH
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
    else:
        result = predict(img_path)
        print(f"Predicted dog breed: {result}")
