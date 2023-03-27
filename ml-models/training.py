import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import RegressionDataset


def train(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=False)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)

    train_dataset = RegressionDataset(X_train, y_train)
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define training parameters
    torch.manual_seed(0)
    criterion = nn.MSELoss()

    def loss_fn(X, y):
        return torch.sqrt(criterion(X, y))

    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 50

    # training loop
    for epoch in range(epochs):
        model.train()
        curr_loss = 0.0

        for batch, (X_batch, y_batch) in enumerate(train_loader, 0):
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
