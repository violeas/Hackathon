import torch
import torch.nn as nn
from terratorch import BACKBONE_REGISTRY

class TerraMindChangeDetector(nn.Module):
    def __init__(self):
        super(TerraMindChangeDetector, self).__init__()

        self.terramind = BACKBONE_REGISTRY.build(
            'terramind_v1_base',
            pretrained=True,
            modalities=['RGB']
        )

        for param in self.terramind.parameters():
            param.requires_grad = False

        # 768 per image x 2 images = 1536
        self.classifier = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def extract_features(self, img_tensor):
        with torch.no_grad():
            out = self.terramind({'RGB': img_tensor})

        if isinstance(out, (list, tuple)):
            feat = out[0]
        else:
            feat = out

        # TerraMind returns [1, 196, 768] → average tokens → [1, 768]
        return feat.mean(dim=1)

    def forward(self, img1, img2):
        feat1 = self.extract_features(img1)   # [1, 768]
        feat2 = self.extract_features(img2)   # [1, 768]
        combined = torch.cat([feat1, feat2], dim=1)  # [1, 1536]
        return self.classifier(combined)       # [1, 1]