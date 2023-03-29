import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import RegressionDataset


def train(model, X_train, y_train):
    train_dataset = RegressionDataset(X_train, y_train)
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define training parameters
    torch.manual_seed(0)
    loss_fn = nn.L1Loss()

    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 25

    # training loop
    for epoch in range(epochs):
        model.train()
        curr_loss = 0.0

        for batch, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.double(), y_batch.double()
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

        print(
            f"[Epoch {(epoch + 1):>3} of {epochs}]: avg_loss = {(curr_loss / (len(train_dataset) / BATCH_SIZE)):>4.4f}")
        curr_loss = 0.0
