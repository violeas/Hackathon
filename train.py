import os
import numpy as np
from PIL import Image
import torch
from model import ChangeDetector

# -------- PATHS --------
path_A = "data/train/A copy"
path_B = "data/train/B"
path_label = "data/train/label"

# -------- LOAD FILE NAMES --------
images_A = sorted(os.listdir(path_A))
images_B = sorted(os.listdir(path_B))

# -------- MODEL --------
model = ChangeDetector()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------- TRAINING --------
for epoch in range(5):
    correct = 0
    total = 0
    total_loss = 0

    for i in range(min(len(images_A), len(images_B))):
        img1 = Image.open(os.path.join(path_A, images_A[i])).resize((64,64))
        img2 = Image.open(os.path.join(path_B, images_B[i])).resize((64,64))

        img1 = np.array(img1)
        img2 = np.array(img2)

        diff = img2 - img1
        diff = torch.tensor(diff.flatten(), dtype=torch.float32)

        label_img = Image.open(os.path.join(path_label, images_A[i])).resize((64,64))
        label_img = np.array(label_img)

      # convert to binary (urban / not)
        label_value = 1 if np.mean(label_img) > 127 else 0
        label = torch.tensor([label_value], dtype=torch.float32)

        output = model(diff)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = 1 if output.item() > 0.5 else 0
        if pred == int(label.item()):
            correct += 1
        total += 1

        


    acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}")

# -------- SAVE MODEL --------
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")