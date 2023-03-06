import os
import pandas as pd


class DataHandler:

    def __init__(self):
        # consider the following: file_path = "/data/open-weather-aalborg-2000-2022.csv"
        self.file_path = os.path.join(os.getcwd(), 'data', 'open-weather-aalborg-2000-2022.csv')

    def get_data(self):
        df = pd.read_csv(self.file_path)
        df = df[100:]
        df.index = pd.to_datetime(df['dt_iso'], format='%Y-%m-%d %H:%M:%S +0000 UTC')
        return df
