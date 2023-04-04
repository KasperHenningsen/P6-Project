import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import RepeatingBasisFunction
from tqdm import tqdm


def get_processed_data(path):
    # Remove duplicate readings from the same hour
    # Occurs when different weather enums are combined, but all other values are the same
    cols = ['dt', 'dt_iso', 'temp', 'dew_point', 'pressure', 'humidity', 'rain_1h', 'snow_1h', 'weather_main']
    df = pd.read_csv(path, usecols=cols).drop_duplicates(['dt'])
    dates = pd.to_datetime(df['dt_iso'], format='%Y-%m-%d %H:%M:%S +0000 UTC')

    # Fill NaN values with 0
    df = df.fillna(0)

    # One-hot encoding of weather_main enum
    df = pd.concat([df, pd.get_dummies(df['weather_main'])], axis=1)
    df = df.drop(columns=['weather_main', 'Clear', 'Clouds', 'Drizzle', 'Dust', 'Fog', 'Haze', 'Mist', 'Thunderstorm',
                          'Tornado', 'dt', 'dt_iso'])

    # Sine / cosine encoding of the hour of day
    # sec_in_day = 60 * 60 * 24
    # df['day_sin'] = np.sin(df['dt'].to_numpy() * (2 * np.pi / sec_in_day))
    # df['day_cos'] = np.cos(df['dt'].to_numpy() * (2 * np.pi / sec_in_day))
    # Radial basis functions might be better

    # Radial basis functions
    # (https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/)
    day_in_year = dates.dt.dayofyear.to_numpy()
    rbf = RepeatingBasisFunction(n_periods=12, input_range=(1, 365))
    day_in_year_df = pd.DataFrame(data=day_in_year)
    rbf_df = pd.DataFrame(data=rbf.fit_transform(day_in_year_df),
                          index=df.index,
                          columns=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    df = pd.concat([df, rbf_df], axis=1)

    hour_in_day = dates.dt.hour.to_numpy()
    rbf2 = RepeatingBasisFunction(n_periods=12, input_range=(0, 23))
    hour_in_day_df = pd.DataFrame(data=hour_in_day)
    rbf2_df = pd.DataFrame(data=rbf2.fit_transform(hour_in_day_df),
                           index=df.index,
                           columns=[f'hour_{i:02d}' for i in range(0, 24, 2)])
    df = pd.concat([df, rbf2_df], axis=1)

    return df


def prepare_X_and_y(input, n_steps_in=12, n_steps_out=12, target_column='temp', step_size=1):
    input_np = input.to_numpy()
    target_idx = input.columns.get_loc(target_column)
    X, y = [], []
    for i in tqdm(range(0, len(input)-n_steps_in-n_steps_out, step_size), desc='Preparing X and y'):
        X_end_idx = i + n_steps_in
        y_end_idx = X_end_idx + n_steps_out
        if y_end_idx > len(input_np):
            break
        X_seq = input_np[i:X_end_idx]
        y_seq = input_np[X_end_idx:y_end_idx, target_idx]
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

def flatten_X_for_MLP(X):
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))
    return X, n_input