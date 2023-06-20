from utils.data_utils import get_processed_data, get_processed_data_energy

if __name__ == '__main__':
    df_weather = get_processed_data('../data/open-weather-aalborg-2000-2022.csv')
    df_weather.corr().to_csv('./correlation_coefficient_matrix_weather.csv')
    df_energy = get_processed_data_energy('../data/energidata.csv')
    df_energy.corr().to_csv('./correlation_coefficient_matrix_energy.csv')

