import os

import numpy as np
import settings
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.datasets import RegressionDataset


def train(model, X_train, y_train, batch_size, save_path=None):
    train_dataset = RegressionDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define training parameters
    torch.manual_seed(0)
    loss_fn = nn.L1Loss()

    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 10

    best_loss = np.Infinity

    print('Begin training')
    # training loop
    for epoch in range(epochs):
        model.train()
        curr_loss = 0.0
        for batch, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} of {epochs}')):
            X_batch, y_batch = X_batch.to(settings.device), y_batch.to(settings.device)
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

        print(f"- avg_loss = {(curr_loss / len(train_loader)):>.3f}")
        epoch_avg_loss = curr_loss / len(train_loader)

        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            if save_path is not None:
                save_model(model, save_path)

    print(f'End of training\n- best_loss = {best_loss}')


def test(model, X_test, y_test, batch_size):
    dataset = RegressionDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.L1Loss()

    total_loss = 0.0
    print('Begin testing')
    model.eval()
    with torch.no_grad():
        for batch, (X_batch, y_batch) in enumerate(tqdm(dataloader, desc=f'Testing')):
            X_batch, y_batch = X_batch.to(settings.device), y_batch.to(settings.device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()

    print(f"End of testing\n- loss = {(total_loss / len(dataloader)):>.3f}")


def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
