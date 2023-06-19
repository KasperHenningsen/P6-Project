from utils.data_utils import get_processed_data, get_processed_data_energy

if __name__ == '__main__':
    df_weather = get_processed_data('../data/open-weather-aalborg-2000-2022.csv')
    df_weather.corr().to_csv('./correlation_coefficient_matrix_weather.csv')
    df_energy = get_processed_data_energy('../data/energidata.csv').drop(columns=["hour_00", "hour_02", "hour_04", "hour_06", "hour_08", "hour_10", "hour_12", "hour_14", "hour_16", "hour_18", "hour_20", "hour_22", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    df_energy.corr().to_csv('./correlation_coefficient_matrix_energy.csv')

