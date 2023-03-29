from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from training import train
from mlp.mlp import MLP
from data_utils import get_processed_data, prepare_X_and_y, flatten_X_for_MLP

if __name__ == '__main__':
    SEQUENCE_LENGTH = 12    # Number of time-steps to use for each prediction
    TARGET_LENGTH = 12      # Number of time-steps to predict
    TARGET_COLUMN = 'temp'  # The column to predict

    data = get_processed_data('./data/open-weather-aalborg-2000-2022.csv')
    X, y = prepare_X_and_y(data, n_steps_in=SEQUENCE_LENGTH, n_steps_out=TARGET_LENGTH, target_column='temp')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

    X_train, n_input = flatten_X_for_MLP(X_train)
    X_test, _ = flatten_X_for_MLP(X_test)

    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    model = MLP(n_input, y.shape[1])

    train(model, X_train, y_train)



