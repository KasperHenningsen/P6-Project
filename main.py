from datetime import datetime

import torch
import time
from data import DataHandler
from model import MultiLayerPerceptron
import pandas as pd
import numpy as np


def main():
    start_time = time.perf_counter()
    dh = DataHandler()
    df = dh.get_data()

    test = torch.tensor(datetime.now().timestamp)
    print(test)

    X_dts = pd.to_datetime(df.index)
    X_dts = X_dts.map(lambda t: t.timestamp)
    X = torch.tensor(X_numpy.astype(np.float32))
    print(X)
    y = torch.tensor(df.values, np.dtype('float32'))
    print(y)

    model = MultiLayerPerceptron('MLP')
    print(model)
    learning_rate = 1e-3
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 10
    loss_values = []
    for epoch in range(epochs):
        print('test?')
    
    end_time = time.perf_counter()
    print(f'Completed in: {end_time - start_time} seconds')


if __name__ == '__main__':
    main()
