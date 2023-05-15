import os

import numpy as np
import settings
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import SymmetricMeanAbsolutePercentageError

from utils.datasets import RegressionDataset
from utils.plotting import plot_loss_history


def train(model, X_train, y_train, X_val, y_val, batch_size, learning_rate, epochs, y_scaler, save_path=None, grad_clipping=None):
    train_dataset = RegressionDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.Generator(device=settings.device))
    val_dataset = RegressionDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, generator=torch.Generator(device=settings.device))

    # Define training parameters
    mae_loss = nn.L1Loss().to(settings.device)
    smape_loss = SymmetricMeanAbsolutePercentageError().to(settings.device)
    mse_loss = nn.MSELoss().to(settings.device)

    y_mean = torch.from_numpy(y_scaler.mean_).to(settings.device)
    y_scale = torch.from_numpy(y_scaler.scale_).to(settings.device)

    def rmse_loss(x, y):
        return torch.sqrt(mse_loss(x, y)).to(settings.device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_loss = np.Infinity
    best_val_smape = np.Infinity
    best_val_rmse = np.Infinity
    losses = []

    print('Begin training')
    # training loop
    for epoch in range(epochs):
        model.train()
        curr_train_loss = 0.0
        curr_train_smape = 0.0
        curr_train_rmse = 0.0
        for batch, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f'(Train) Epoch {epoch + 1} of {epochs}')):
            X_batch, y_batch = X_batch.to(settings.device), y_batch.to(settings.device)
            optimizer.zero_grad()

            y_pred = model(X_batch) * y_scale + y_mean

            loss = mae_loss(y_pred, y_batch)
            smape = smape_loss(y_pred, y_batch)
            rmse = rmse_loss(y_pred, y_batch)

            loss.backward()

            # perform gradient clipping
            if grad_clipping is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

            optimizer.step()

            curr_train_loss += loss.item()
            curr_train_smape += smape.item()
            curr_train_rmse += rmse.item()

        epoch_avg_train_loss = curr_train_loss / len(train_loader)
        epoch_avg_train_smape = curr_train_smape / len(train_loader)
        epoch_avg_train_rmse = curr_train_rmse / len(train_loader)

        print(f"- Losses: MAE = {epoch_avg_train_loss:>.3f}, SMAPE = {100 * epoch_avg_train_smape:>.2f}%, RMSE = {epoch_avg_train_rmse:>.3f}")

        # Validation step
        model.eval()
        curr_val_loss = 0.0
        curr_val_smape = 0.0
        curr_val_rmse = 0.0

        with torch.no_grad():
            for batch, (X_batch, y_batch) in enumerate(tqdm(val_loader, desc=f'(Val) Epoch {epoch + 1} of {epochs}')):
                X_batch, y_batch = X_batch.to(settings.device), y_batch.to(settings.device)

                y_pred = model(X_batch) * y_scale + y_mean

                curr_val_loss += mae_loss(y_pred, y_batch).item()
                curr_val_smape += smape_loss(y_pred, y_batch).item()
                curr_val_rmse += rmse_loss(y_pred, y_batch).item()

        epoch_avg_val_loss = curr_val_loss / len(val_loader)
        epoch_avg_val_smape = curr_val_smape / len(val_loader)
        epoch_avg_val_rmse = curr_val_rmse / len(val_loader)

        print(f"- Validation losses: MAE = {epoch_avg_val_loss:>.3f}, SMAPE = {100 * epoch_avg_val_smape:>.2f}%, RMSE = {epoch_avg_val_rmse:>.3f}")

        losses.append(epoch_avg_val_loss)

        if epoch_avg_val_loss < best_val_loss:
            best_val_loss = epoch_avg_val_loss
            best_val_smape = epoch_avg_val_smape
            best_val_rmse = epoch_avg_val_rmse
            if save_path is not None:
                save_model(model, save_path)


    print(f'End of training\nVal losses: MAE = {best_val_loss:>.3f}, SMAPE = {100 * best_val_smape:>.3f}%, RMSE = {best_val_rmse:>.3f}')
    plot_loss_history(model, losses)
    return [best_val_loss, best_val_smape, best_val_rmse]


def test(model, X_test, y_test, batch_size, y_scaler):
    dataset = RegressionDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, generator=torch.Generator(device=settings.device))
    loss_fn = nn.L1Loss().to(settings.device)
    smape = SymmetricMeanAbsolutePercentageError().to(settings.device)
    mse = nn.MSELoss().to(settings.device)

    y_mean = torch.from_numpy(y_scaler.mean_).to(settings.device)
    y_scale = torch.from_numpy(y_scaler.scale_).to(settings.device)

    def rmse(x, y):
        return torch.sqrt(mse(x, y))

    total_loss = 0.0
    total_smape = 0.0
    total_rmse = 0.0
    print('Begin testing')
    model.eval()
    with torch.no_grad():
        for batch, (X_batch, y_batch) in enumerate(tqdm(dataloader, desc=f'Testing')):
            X_batch, y_batch = X_batch.to(settings.device), y_batch.to(settings.device)

            y_pred = model(X_batch) * y_scale + y_mean

            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            total_smape += smape(y_pred, y_batch).item()
            total_rmse += rmse(y_pred, y_batch).item()

    total_loss /= len(dataloader)
    total_smape /= len(dataloader)
    total_rmse /= len(dataloader)
    print(f"End of testing\n- MAE = {total_loss:>.3f}, SMAPE = {100 * total_smape:>.3f}%, RMSE = {total_rmse:>.3f}")
    return [total_loss, total_smape, total_rmse]


def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
