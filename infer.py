import torch
import numpy as np
from PIL import Image
from model import ChangeDetector

def run_inference(before_path, after_path):
    model = ChangeDetector()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    img1 = Image.open(before_path).resize((64,64))
    img2 = Image.open(after_path).resize((64,64))

    img1 = np.array(img1)
    img2 = np.array(img2)

    diff = img2 - img1
    diff = torch.tensor(diff.flatten(), dtype=torch.float32)

    with torch.no_grad():
        output = model(diff)

    prediction = "Urbanized" if output.item() > 0.5 else "Not Urbanized"
    confidence = output.item()

    return prediction, confidence


# test run
if __name__ == "__main__":
    pred, conf = run_inference("sample_input/before.png", "sample_input/after.png")
    print(f"Prediction: {pred}")
    print(f"Confidence: {conf:.2f}")