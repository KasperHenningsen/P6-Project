from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from gru.gru import GRUNet
from training import train, test
from mlp.mlp import MLP
from data_utils import get_processed_data, prepare_X_and_y, flatten_X_for_MLP

if __name__ == '__main__':
    seq_length = 12         # Number of time-steps to use for each prediction
    target_length = 12      # Number of time-steps to predict
    target_col = 'temp'     # The column to predict
    batch_size = 32

    model = GRUNet(input_size=32, hidden_size=32, output_size=1, dropout_prob=0, num_layers=1)

    # Prepare data
    df = get_processed_data('./data/open-weather-aalborg-2000-2022.csv')
    train_df, test_df = train_test_split(df, train_size=0.6, shuffle=False)

    # Train
    X_train, y_train = prepare_X_and_y(train_df, n_steps_in=seq_length, n_steps_out=target_length, target_column=target_col)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    try:
        train(model, X_train, y_train, batch_size, './saved-models/gru01.pt')
    except KeyboardInterrupt:
        print("Exiting early from training")
        state_dict = torch.load('./saved-models/gru01.pt')
        model.load_state_dict(state_dict)
        model.eval()

    print("\n====================\n")

    # Test
    X_test, y_test = prepare_X_and_y(test_df, n_steps_in=seq_length, n_steps_out=target_length, target_column=target_col)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    test(model, X_test, y_test, batch_size)

