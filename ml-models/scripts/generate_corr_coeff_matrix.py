import sys
from data_utils import get_processed_data


if __name__ == '__main__':
    df = get_processed_data('../data/open-weather-aalborg-2000-2022.csv')
    df_corr = df.corr()
    df_corr.to_csv('./correlation_coefficient_matrix.csv')
