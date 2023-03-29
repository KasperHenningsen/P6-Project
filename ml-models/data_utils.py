import numpy as np
import pandas as pd
from sklego.preprocessing import RepeatingBasisFunction


def get_processed_data(path):
    # Remove duplicate readings from the same hour
    # Occurs when different weather enums are combined, but all other values are the same
    df = pd.read_csv(path).drop_duplicates(['dt'])

    # Fill NaN values with 0
    df = df.fillna(0)

    # One-hot encoding of weather_main enum
    df = pd.concat([df, pd.get_dummies(df['weather_main'])], axis=1)
    df = df.drop(columns=['weather_main'])

    # Sine / cosine encoding of the hour of day
    sec_in_day = 60 * 60 * 24
    df['day_sin'] = np.sin(df['dt'].to_numpy() * (2 * np.pi / sec_in_day))
    df['day_cos'] = np.cos(df['dt'].to_numpy() * (2 * np.pi / sec_in_day))

    # Radial basis functions
    # (https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/)
    day_in_year = pd.to_datetime(df['dt_iso'], format='%Y-%m-%d %H:%M:%S +0000 UTC').dt.dayofyear.to_numpy()
    rbf = RepeatingBasisFunction(n_periods=12, input_range=(1, 365))
    day_in_year_df = pd.DataFrame(data=day_in_year)
    rbf_df = pd.DataFrame(data=rbf.fit_transform(day_in_year_df),
                          index=df.index,
                          columns=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    df = pd.concat([df, rbf_df], axis=1)

    # Drop columns that are useless for training
    df = df.drop(
        columns=['dt', 'dt_iso', 'visibility', 'timezone', 'city_name', 'lat', 'lon', 'sea_level', 'grnd_level',
                 'weather_id', 'weather_description', 'weather_icon'])

    return df


def prepare_X_and_y(input, n_steps_in=12, n_steps_out=12, target_column='temp'):
    X, y = [], []
    for i in range(0, len(input)-(n_steps_in*2), n_steps_in):
        X_end_idx = i + n_steps_in
        y_end_idx = X_end_idx + n_steps_out
        if y_end_idx > len(input):
            break
        X_seq = input[i:X_end_idx]
        y_seq = input[target_column][X_end_idx:y_end_idx]
        X.append(X_seq)
        y.append(y_seq)
    X, y = np.array(X), np.array(y)
    return X, y


def flatten_X_for_MLP(X):
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))
    return X, n_input


