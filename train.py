import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor, ToTensor
from torchvision.datasets import MNIST
from model import FFN


# Load MNIST dataset
trainset = MNIST(
            root="data",
            train=True,
            download=True,
            transform=PILToTensor(),
        )

valset = MNIST(
            root="data",
            train=False,
            download=True,
            transform=PILToTensor(),
        )

# Model config (Feedforward Network)
INPUT_SIZE = 784
OUTPUT_SIZE = 10
HIDDEN_LAYERS = [100, 100, 100]
BATCH_SIZE = 60
EPOCHS = 50
LR = 1e-3

# Data Loader for convenient batching + image flattening
def collate_fn(samples):
    images, labels = zip(*samples)
    batch_size = len(images)
    images = list(map(lambda x: x.flatten().to(dtype=torch.float32), images)) # flatten
    # images = list(map(lambda x: x / 255., images)) # normalize
    return torch.stack(images), torch.tensor(labels)

train_loader = DataLoader(
                    trainset, 
                    collate_fn=collate_fn,
                    batch_size=BATCH_SIZE,
                    shuffle=True
                )
val_loader = DataLoader(
                    valset,
                    collate_fn=collate_fn,
                    batch_size=BATCH_SIZE,
                    shuffle=True
                )

# Initialize model (w/o batch normalization layer)
model = FFN(HIDDEN_LAYERS, INPUT_SIZE, OUTPUT_SIZE)

# Optimizer, here we use Stochastic Gradient Descent as in the original paper
optimizer = optim.SGD(model.parameters(), lr=LR)

# Training criterion (or loss function)
CE_loss = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train(True)

    print(f"[EPOCH {epoch + 1}]")
    tmp_loss = 0.
    for idx, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(imgs) 
        loss = CE_loss(logits, labels)
        loss.backward()
        optimizer.step()
        tmp_loss += loss.item()
        if (idx + 1) % 200 == 0:
            print(f"Train Loss = {tmp_loss / 200:.3f} / {epoch * 1000 + (idx + 1)} (steps)")
            tmp_loss = 0.

    model.train(False)
    val_loss = 0.
    for idx, (imgs, labels) in enumerate(val_loader):
        logits = model(imgs)
        loss = CE_loss(logits, labels)
        val_loss += loss.item()
    print(f"Validation Loss = {val_loss / (idx + 1):.3f}\n")

