import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import RegressionDataset


def train(model, X_train, y_train, save_path=None):
    train_dataset = RegressionDataset(X_train, y_train)
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define training parameters
    torch.manual_seed(0)
    loss_fn = nn.L1Loss()

    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 10

    best_loss = np.Infinity

    # training loop
    for epoch in range(epochs):
        model.train()
        curr_loss = 0.0

        for batch, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} of {epochs}')):
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

        if curr_loss < best_loss and save_path is not None:
            torch.save(model, save_path)

        print(
            f"\t- avg_loss = {(curr_loss / (len(train_dataset) / BATCH_SIZE)):>4.4f}")
        curr_loss = 0.0
