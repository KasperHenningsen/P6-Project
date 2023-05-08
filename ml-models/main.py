import time

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

import settings
from baselines.cnn import CNN
from baselines.gru import GRU
from baselines.mlp import MLP
from baselines.rnn import RNN
from baselines.lstm import LSTM
from baselines.tcn import TCN
from baselines.transformer import Transformer
from mtgnn.mtgnn import MTGNN
from training import train, test
from utils.plotting import plot
from utils.data_utils import get_processed_data, prepare_X_and_y
from utils.file_utils import set_next_save_path, generate_train_test_log, set_load_path

if __name__ == '__main__':
    seq_length = 12         # Number of time-steps to use for each prediction
    target_length = 12      # Number of time-steps to predict
    target_col = 'temp'     # The column to predict
    batch_size = 8
    epochs = 15
    learning_rate = 1e-4
    train_size = 0.6
    val_size = 0.2
    grad_clipping = None
    cnn = CNN(input_channels=32, hidden_size=12, kernel_size=12, dropout_prob=0)
    mlp = MLP(input_size=32, hidden_size=256, output_size=1, num_layers=1, seq_length=seq_length)
    gru = GRU(input_size=32, hidden_size=32, output_size=1, dropout_prob=0, num_layers=1)
    rnn = RNN(input_size=32, hidden_size=256, output_size=1, dropout_prob=0.2, num_layers=3)
    lstm = LSTM(input_size=32, hidden_size=32, output_size=1, dropout_prob=0, num_layers=1)
    tcn = TCN(input_size=32, output_size=1, hidden_size=12)
    mtgnn = MTGNN(num_features=32, seq_length=seq_length, num_layers=3, subgraph_size=8, subgraph_node_dim=16, use_output_convolution=False, dropout=0.3)
    transformer = Transformer(input_size=32, d_model=128, nhead=4, num_layers=6, output_size=12, dropout=0.1)

    train_model = tcn
    print(f'Model: {train_model.get_name()}')

    set_next_save_path(train_model)
    print(f'Will save model in \'{train_model.path}\'')

    os.makedirs(settings.models_path, exist_ok=True)
    os.makedirs(settings.plots_path, exist_ok=True)

    # Prepare data
    df = get_processed_data('./data/open-weather-aalborg-2000-2022.csv')

    # Generate RBF plot
    #plot_rbf_small(df)

    train_val_df, test_df = train_test_split(df, train_size=train_size + val_size, shuffle=False)

    # Split train into train + val
    train_df, val_df = train_test_split(train_val_df, test_size=(val_size / (train_size + val_size)), shuffle=False)

    # Train
    print("\n========== Training ==========")
    train_losses = [None, None, None]
    X_train, y_train = prepare_X_and_y(train_df, n_steps_in=seq_length, n_steps_out=target_length, target_column=target_col)
    X_val, y_val = prepare_X_and_y(val_df, n_steps_in=seq_length, n_steps_out=target_length, target_column=target_col)
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = X_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, y_train.shape[-1]))

    # Save scaler for later use
    joblib.dump(X_scaler, os.path.join(settings.scalers_path, f'horizon_{seq_length}', 'X_scaler.gz'))
    joblib.dump(y_scaler, os.path.join(settings.scalers_path, f'horizon_{seq_length}', 'y_scaler.gz'))

    train_start_time = time.time()
    total_train_time = None
    has_trained = False
    try:
        train_losses = train(train_model, X_train, y_train, X_val, y_val, batch_size, learning_rate, epochs, grad_clipping=grad_clipping, save_path=train_model.path, y_scaler=y_scaler)
        total_train_time = time.time() - train_start_time
        has_trained = True
    except KeyboardInterrupt:
        print("Exiting early from training")

    set_load_path(train_model)
    train_model.load_saved_model()
    print(f'Loaded model from \'{train_model.path}\'')

    # Test
    print("\n========== Testing ==========")
    test_losses = [None, None, None]
    X_test, y_test = prepare_X_and_y(test_df, n_steps_in=seq_length, n_steps_out=target_length, target_column=target_col, step_size=seq_length)
    X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    has_tested = False
    try:
        test_losses = test(train_model, X_test, y_test, batch_size, y_scaler=y_scaler)
        has_tested = True
    except KeyboardInterrupt:
        print("Exiting early from testing")

    if has_trained and has_tested:
        generate_train_test_log(train_model,
                                train_losses=train_losses,
                                test_losses=test_losses,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                batch_size=batch_size,
                                target_col=target_col,
                                target_len=target_length,
                                train_size=train_size,
                                grad_clipping=grad_clipping,
                                seq_len=seq_length,
                                train_time=total_train_time)

    # Plotting
    print("\n========== Plotting ==========")
    plot(train_model, X_test, y_test, start=0, end=40, y_scaler=y_scaler)
    #plot((cnn, gru, rnn, lstm, tcn, transformer), X_test, y_test, start=0, end=240, step=seq_length)
