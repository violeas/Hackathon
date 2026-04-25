import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from model import TerraMindChangeDetector
from collections import Counter

# -------- PATHS --------
path_A     = "data/train/A"
path_B     = "data/train/B"
path_label = "data/train/label"

# -------- LOAD FILES --------
images = sorted([f for f in os.listdir(path_A) if f.endswith('.png')])
print(f"Total images: {len(images)}")

# -------- IMAGE TRANSFORM --------
# TerraMind expects normalized tensors
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -------- CHECK LABELS --------
print("Scanning labels...")
label_counts = Counter()
for img_name in images:
    label_img = Image.open(os.path.join(path_label, img_name)).convert("L").resize((64, 64))
    label_arr = np.array(label_img, dtype=np.float32)
    val = 1 if np.max(label_arr) > 10 else 0
    label_counts[val] += 1

print(f"Urbanized   (1): {label_counts[1]}")
print(f"Not Urban   (0): {label_counts[0]}")
pos_weight_val = label_counts[0] / label_counts[1]
print(f"pos_weight      : {pos_weight_val:.2f}\n")

# -------- MODEL --------
print("Loading IBM TerraMind... (first time downloads weights, may take a few minutes)")
model     = TerraMindChangeDetector()
print("✅ TerraMind loaded!\n")

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

# -------- TRAINING --------
best_acc = 0.0

for epoch in range(20):
    model.train()
    correct    = 0
    total      = 0
    total_loss = 0.0

    for img_name in images:
        # ---- LOAD + TRANSFORM ----
        img1 = Image.open(os.path.join(path_A, img_name)).convert("RGB")
        img2 = Image.open(os.path.join(path_B, img_name)).convert("RGB")

        t1 = transform(img1).unsqueeze(0)  # shape: [1, 3, 224, 224]
        t2 = transform(img2).unsqueeze(0)

        # ---- LABEL ----
        label_img   = Image.open(os.path.join(path_label, img_name)).convert("L").resize((64, 64))
        label_arr   = np.array(label_img, dtype=np.float32)
        label_value = 1.0 if np.max(label_arr) > 10 else 0.0
        label       = torch.tensor([label_value], dtype=torch.float32)

        # ---- FORWARD + LOSS ----
        output = model(t1, t2)
        loss   = criterion(output.squeeze(0), label)

        # ---- BACKPROP ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0
        if pred == int(label_value):
            correct += 1
        total += 1

    acc    = correct / total
    marker = " ⭐ best" if acc > best_acc else ""
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Accuracy: {acc*100:.1f}%{marker}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model_best.pth")

torch.save(model.state_dict(), "model.pth")
print(f"\n✅ Training complete!")
print(f"✅ Best Accuracy : {best_acc*100:.1f}%")
print(f"✅ Saved: model.pth and model_best.pth")