from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import training
from mlp.mlp import MLP
from preprocessing import get_processed_data, prepare_X_and_y

if __name__ == '__main__':
    data = get_processed_data('./data/open-weather-aalborg-2000-2022.csv')
    X, y = prepare_X_and_y(data, 12, 12, 12)
    model = MLP(X.shape[1], y.shape[1])
    training.train(model, X, y)



