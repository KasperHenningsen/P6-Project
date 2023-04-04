from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from gru.gru import GRUNet
from training import train
from mlp.mlp import MLP
from data_utils import get_processed_data, prepare_X_and_y, flatten_X_for_MLP

if __name__ == '__main__':
    seq_length = 12         # Number of time-steps to use for each prediction
    target_length = 12      # Number of time-steps to predict
    target_col = 'temp'     # The column to predict

    df = get_processed_data('./data/open-weather-aalborg-2000-2022.csv')

    train_df, test_df = train_test_split(df, train_size=0.7, shuffle=False)

    X_train, y_train = prepare_X_and_y(train_df, n_steps_in=seq_length, n_steps_out=target_length, target_column=target_col)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

    model = GRUNet(input_size=32, hidden_size=128, output_size=1, dropout_prob=0.0, num_layers=2)

    train(model, X_train, y_train)

