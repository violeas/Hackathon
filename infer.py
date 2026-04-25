import torch
import torchvision.transforms as T
from PIL import Image
from model import TerraMindChangeDetector

# -------- LOAD MODEL --------
model = TerraMindChangeDetector()
model.load_state_dict(torch.load("model_best.pth", map_location="cpu"))
model.eval()
print("✅ TerraMind Change Detector loaded\n")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def predict(before_path, after_path):
    img1 = Image.open(before_path).convert("RGB")
    img2 = Image.open(after_path).convert("RGB")

    t1 = transform(img1).unsqueeze(0)
    t2 = transform(img2).unsqueeze(0)

    with torch.no_grad():
        output = model(t1, t2)
        prob   = torch.sigmoid(output).item()

    print(f"Before : {before_path}")
    print(f"After  : {after_path}")
    print(f"Score  : {prob:.4f}")

    if prob > 0.5:
        print(f"Result : 🏙️  URBANIZED ({prob*100:.1f}% confidence)")
    else:
        print(f"Result : 🌿  NOT URBANIZED ({(1-prob)*100:.1f}% confidence)")

    return prob

predict("sample_input/before.png", "sample_input/after.png")