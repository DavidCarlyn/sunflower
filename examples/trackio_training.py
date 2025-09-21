from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
from scipy.io import loadmat
from tqdm import tqdm
from PIL import Image

from sunflower.models import AIModel


class FlowerClassifier(AIModel):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        self.model(x)


class OxfordFlowersDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.labels = loadmat(self.root_dir / "imagelabels.mat")["labels"][
            0
        ]  # I believed they are ordered by id

        self.num_classes = max(self.labels) + 1

        setids = loadmat(self.root_dir / "setid.mat")
        train_ids = setids["trnid"][0]
        valid_ids = setids["valid"][0]
        test_ids = setids["tstid"][0]

        paths = []
        for root, dirs, files in Path(self.root_dir, "jpg").walk():
            for f in files:
                num = int(Path(f).stem.split("_")[1])  # files are named 'image_{id}.jpg
                paths.append([num, root / f])
        self.paths = [y for x, y in sorted(paths, key=lambda x: x[0])]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        lbl = self.labels[idx]

        if self.transforms:
            img = self.transforms(img)

        return img, lbl


def model_transform():
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


flowers_dir = "/local/scratch/datasets/oxford-flowers/"
dataset = OxfordFlowersDataset(root_dir=flowers_dir, transforms=model_transform())
model = FlowerClassifier(num_classes=dataset.num_classes).cuda()

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=16, num_workers=4, shuffle=True
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.0003)
for epoch in tqdm(range(50), desc="Training", colour="#333333"):
    batch_tbar = tqdm(train_dataloader, desc="Going through batch", colour="#333333")
    correct = 0
    total = 0
    for imgs, lbls in batch_tbar:
        output = model(imgs.cuda())
        loss = loss_fn(output, lbls.cuda())

        _, preds = torch.max(output, 1)

        correct += (preds == lbls.cuda()).sum().item()
        total += len(lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_tbar.set_postfix(
            {"Loss": loss.item(), "Running Accuracy": correct / total}
        )
