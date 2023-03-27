import numpy as np
import pandas as pd
from sklego.preprocessing import RepeatingBasisFunction


def get_processed_data(path):
    """

    :param path:
    :return:
    """
    df = pd.read_csv(path).drop_duplicates(['dt'])
    temp = df['temp'].to_numpy()

    # Radial basis functions
    # See https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/
    day_in_year = pd.to_datetime(df['dt_iso'], format='%Y-%m-%d %H:%M:%S +0000 UTC').dt.dayofyear.to_numpy()
    rbf = RepeatingBasisFunction(n_periods=12, input_range=(1, 365))
    rbf_df = pd.DataFrame(data=day_in_year)
    rbf_series = rbf.fit_transform(rbf_df)

    series = []
    for i in range(len(df)):
        series.append(np.concatenate(([temp[i]], rbf_series[i])))
    series = np.array(series)

    return series


def prepare_X_and_y(sequences, n_steps_in, n_steps_out, step_size=1):
    X, y = [], []
    for i in range(0, len(sequences), step_size):
        X_end_idx = i + n_steps_in
        y_end_idx = X_end_idx + n_steps_out

        if y_end_idx > len(sequences):
            break

        X_seq = sequences[i:X_end_idx]
        y_seq = sequences[X_end_idx:y_end_idx]
        X.append(X_seq)
        y.append(y_seq)
    X, y = np.array(X), np.array(y)
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))
    n_output = y.shape[1] * y.shape[2]
    y = y.reshape((y.shape[0], n_output))
    return X, y

