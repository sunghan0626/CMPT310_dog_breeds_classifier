import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import os
import sys

==== CONFIG ====
MODEL_PATH = "best_model.pth"       # Path to the trained model
DATA_DIR = "data/Processed/train"   # Folder to extract class labels
IMG_PATH = "sample_dog.jpg"         # <-- Replace with your image path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

==== Load class labels ====
class_names = sorted(os.listdir(DATA_DIR))

==== Image preprocessing (same as training normalization) ====
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

==== Load model (ResNet18 with modified FC layer) ====
def get_resnet_model(num_classes):
    model = resnet18(weights=None)    # no pretrained weights, only structure
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

==== Initialize and load weights ====
model = get_resnet_model(num_classes=len(class_names)).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

==== Prediction function ====
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        outputs = model(imgtensor)
        , predicted = outputs.max(1)

    breed = class_names[predicted.item()]
    return breed

==== Run prediction ====
if name == "main":
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMG_PATH
    result = predict(img_path)
    print(f"Predicted dog breed: {result}")