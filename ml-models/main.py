import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

import settings
from baselines.cnn import Conv1D
from baselines.gru import GRUNet
from baselines.rnn import RNNNet
from baselines.lstm import LSTM
from baselines.tcn import TemporalConvolutionNetwork
from mtgnn.mtgnn import MTGNN
from training import train, test
from plotting import plot
from data_utils import get_processed_data, prepare_X_and_y

if __name__ == '__main__':
    seq_length = 12         # Number of time-steps to use for each prediction
    target_length = 12      # Number of time-steps to predict
    target_col = 'temp'     # The column to predict
    batch_size = 32
    cnn = Conv1D(input_channels=32, kernel_size=12, output_size=target_length, dropout_prob=0)
    gru = GRUNet(input_size=32, hidden_size=32, output_size=1, dropout_prob=0, num_layers=1)
    rnn = RNNNet(input_size=32, hidden_size=256, output_size=1, dropout_prob=0.2, num_layers=3, nonlinearity='relu')
    lstm = LSTM(input_size=32, hidden_size=32, output_size=1, dropout_prob=0, num_layers=1)
    tcn = TemporalConvolutionNetwork(input_size=32, output_size=1, hidden_size=12)
    mtgnn = MTGNN(num_features=32, seq_length=12, is_training=True, use_output_convolution=False)

    os.makedirs(settings.models_path, exist_ok=True)
    os.makedirs(settings.plots_path, exist_ok=True)

    # Prepare data
    df = get_processed_data('./data/open-weather-aalborg-2000-2022.csv')
    train_df, test_df = train_test_split(df, train_size=0.6, shuffle=False)

    # Train
    print("\n========== Training ==========")
    X_train, y_train = prepare_X_and_y(train_df, n_steps_in=seq_length, n_steps_out=target_length, target_column=target_col)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

    # Save scaler for later use
    joblib.dump(scaler, os.path.join(settings.models_path, 'scaler.gz'))

    try:
        train(mtgnn, X_train, y_train, batch_size, os.path.join(settings.models_path, mtgnn.get_name()))
    except KeyboardInterrupt:
        print("Exiting early from training")
        tcn.load_saved_model()

    # Test
    print("\n========== Testing ==========")
    X_test, y_test = prepare_X_and_y(test_df, n_steps_in=seq_length, n_steps_out=target_length, target_column=target_col)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    test(tcn, X_test, y_test, batch_size)

    # Plotting
    print("\n========== Plotting ==========")
    plot(tcn, X_test, y_test, start=0, end=240, step=seq_length)
    #plot((model1, model2, model3, model4), X_test, y_test, start=0, end=1000, step=seq_length)
