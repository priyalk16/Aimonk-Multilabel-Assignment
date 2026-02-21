import torch
from torchvision import transforms
from PIL import Image
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load("models/aimonk_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = "images/image_0.jpg"  # Change if needed

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()

attributes = ["Attr1", "Attr2", "Attr3", "Attr4"]

present_attributes = [attributes[i] for i in range(4) if preds[0][i] == 1]

print("Attributes present:", present_attributes)