import os

import numpy as np
import settings
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import MeanAbsolutePercentageError

from utils.datasets import RegressionDataset


def train(model, X_train, y_train, batch_size, save_path=None, grad_clipping=None):
    train_dataset = RegressionDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define training parameters
    torch.manual_seed(0)
    mae_loss = nn.L1Loss()
    mape_loss = MeanAbsolutePercentageError()
    mse_loss = nn.MSELoss()

    def rmse_loss(x, y):
        return torch.sqrt(mse_loss(x, y))

    optimizer = Adam(model.parameters(), lr=0.00001)
    epochs = 20

    best_loss = np.Infinity
    losses = []

    print('Begin training')
    # training loop
    for epoch in range(epochs):
        model.train()
        curr_loss = 0.0
        curr_mape = 0.0
        curr_rmse = 0.0
        for batch, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} of {epochs}')):
            X_batch, y_batch = X_batch.to(settings.device), y_batch.to(settings.device)
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = mae_loss(y_pred, y_batch)
            mape = mape_loss(y_pred, y_batch)
            rmse = rmse_loss(y_pred, y_batch)

            loss.backward()

            # perform gradient clipping
            if grad_clipping is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

            optimizer.step()

            curr_loss += loss.item()
            curr_mape += mape.item()
            curr_rmse += rmse.item()

        epoch_avg_loss = curr_loss / len(train_loader)
        epoch_avg_mape = curr_mape / len(train_loader)
        epoch_avg_rmse = curr_rmse / len(train_loader)
        print(f"- MAE = {epoch_avg_loss:>.3f}, MAPE = {epoch_avg_mape:>.3f}, RMSE = {epoch_avg_rmse:>.3f}")
        losses.append(epoch_avg_loss)

        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            if save_path is not None:
                save_model(model, save_path)

    print(f'End of training\n- best MAE = {best_loss}')
    return losses


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
