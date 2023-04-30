import pandas as pd
import numpy as np
import torch
from flask import Flask, request, send_file, make_response

import settings
from utils.data_utils import get_processed_data
from baselines.cnn import CNN
from baselines.gru import GRU
from baselines.mlp import MLP
from baselines.rnn import RNN
from baselines.lstm import LSTM
from baselines.tcn import TCN
from baselines.transformer import Transformer
from mtgnn.mtgnn import MTGNN

import datetime
import json
import os

app = Flask(__name__)

featurematrix = f"{settings.scripts_path}\\correlation_coefficient_matrix.csv"
data = get_processed_data(f"{settings.data_path}\\open-weather-aalborg-2000-2022.csv")


@app.route('/featurematrix')
def get_feature_matrix():
    """
    :return: CSV of feature correlations
    """
    return send_file(featurematrix)


# TODO: Remove?
@app.route('/dataset')
def get_dataset():
    """
    :return: A list of datetime/temp tuples over the entire dataset
    """
    dates = np.ndarray.tolist(data.index.to_pydatetime())
    temp = np.ndarray.tolist(data["temp"].values)

    response = list(zip(dates, temp))

    return make_response(response)


@app.route('/dataset/dates')
def get_dates():
    """
    :return: The min and max datetimes present in the dataset
    """
    return make_response([data.index.min(), data.index.max()])


@app.route('/predictions/models')
def get_models():
    """
    :return: A list of model folders in the settings.models_path directory
    """
    models = [f for f in os.listdir(settings.models_path) if f not in {'X_scaler.gz', 'y_scaler.gz'}]

    return make_response(models)


@app.route('/predictions/models/cnn')
def get_cnn():
    args = request.args
    horizon = args.get('horizon', type=int)
    start_date = args.get('start_date', type=to_date)
    end_date = args.get('end_date', type=to_date)

    if not (isinstance(start_date, datetime.date)):
        return "Wrong format: start_date", 400
    elif not (isinstance(end_date, datetime.date)):
        return "Wrong format: end_date", 400

    cnn = get_model_object('cnn', horizon)
    response = get_inference_data(cnn, horizon, start_date, end_date)

    return make_response(response)


@app.route('/predictions/models/mtgnn')
def get_mtgnn():
    args = request.args
    horizon = args.get('horizon')
    # start_date = args.get('start_date', type=to_date)
    # end_date = args.get('end_date', type=to_date)

    model_json = json.load(open(f'{settings.models_path}\\MTGNN\\horizon_{horizon}\\log.json'))
    model_params = model_json['model_parameters']

    num_features = model_params['model_parameters']['num_features']
    seq_length = model_params['model_parameters']['seq_length']
    num_layers = model_params['model_parameters']['num_layers']
    subgraph_size = model_params['model_parameters']['subgraph_size']
    subgraph_node_dim = model_params['model_parameters']['subgraph_node_dim']
    use_output_convolution = model_params['model_parameters']['use_output_convolution']
    dropout = model_params['model_parameters']['dropout']

    MTGNN(num_features, seq_length, num_layers, subgraph_size, subgraph_node_dim, use_output_convolution,
          dropout)


def to_date(string):
    date = datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S %z %Z").replace(tzinfo=None)
    return date


def get_model_object(model, horizon):
    model_path = f'{settings.models_path}\\{model.upper()}\\horizon_{horizon}'
    model_json = json.load(open(f'{model_path}\\log.json'))
    model_params = model_json['model_parameters']

    if model == 'cnn':
        input_channels = model_params['input_channels']
        hidden_size = model_params['hidden_size']
        kernel_size = model_params['kernel_size']
        dropout_prob = model_params['dropout']

        cnn = CNN(input_channels, hidden_size, kernel_size, dropout_prob)

        state_dict = torch.load(os.path.join(f'saved-models/CNN/horizon_{horizon}/model.pt'))
        cnn.load_state_dict(state_dict)
        cnn.eval()

        return cnn
    elif model == 'mlp':
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        output_size = model_params['output_size']
        num_layers = model_params['num_layers']
        seq_length = model_params['seq_length']

        mlp = MLP(input_size, hidden_size, output_size, num_layers, seq_length)
        return mlp.load_saved_model()
    elif model == 'gru':
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        output_size = model_params['output_size']
        dropout_prob = model_params['dropout']
        num_layers = model_params['num_layers']

        gru = GRU(input_size, hidden_size, output_size, dropout_prob, num_layers)
        return gru.load_saved_model()
    elif model == 'rnn':
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        output_size = model_params['output_size']
        dropout_prob = model_params['dropout']
        num_layers = model_params['num_layers']

        rnn = RNN(input_size, hidden_size, output_size, dropout_prob, num_layers)
        return rnn.load_saved_model()
    elif model == 'lstm':
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        output_size = model_params['output_size']
        dropout_prob = model_params['dropout']
        num_layers = model_params['num_layers']

        lstm = LSTM(input_size, hidden_size, output_size, dropout_prob, num_layers)
        return lstm.load_saved_model()
    elif model == 'tcn':
        input_size = model_params['input_size']
        output_size = model_params['output_size']
        hidden_size = model_params['hidden_size']

        tcn = TCN(input_size, output_size, hidden_size)
        return tcn.load_saved_model()
    elif model == 'transformer':
        input_size = model_params['input_size']
        d_model = model_params['d_model']
        nhead = model_params['nhead']
        num_layers = model_params['num_layers']
        output_size = model_params['output_size']
        dropout = model_params['dropout']

        transformer = Transformer(input_size, d_model, nhead, num_layers, output_size, dropout)
        return transformer.load_saved_model()


def get_inference_data(model, horizon, start_date, end_date):
    timedelta = datetime.timedelta(hours=horizon)

    min_index = pd.to_datetime(data.index.min())
    min_inference = min_index + timedelta

    max_index = pd.to_datetime(data.index.max())
    max_inference = max_index - timedelta

    result = []
    if start_date < min_inference:
        result = initialize_inference(model, horizon)
        if end_date > min_inference:
            result += rec_inference(model, horizon, min_index, end_date)
    elif start_date > max_index:
        result = rec_inference(model, horizon, max_inference, end_date)
    elif start_date > min_inference:
        result += rec_inference(model, horizon, min_inference, end_date)

    crop_result(start_date, end_date, result)

    return result


def initialize_inference(model, horizon):
    """
    Backward infers the temperature values of the horizon before the dataset and the first horizon of the dataset,
    since these sets cannot be infered forwards due to missing dataset values
    :param model: The NN model used
    :param horizon: The input/output horizon
    :return: The horizon preceding the dataset and the first horizon's worth of inference
    """
    inference_set = []
    one_hour = datetime.timedelta(hours=1)

    for x in range(2, 0, -1):
        input_data = data[horizon * (x - 1):horizon * x]
        input_data = input_data[::-1]

        data_tensor = torch.from_numpy(input_data.to_numpy())[:horizon]
        data_tensor = data_tensor.reshape(1, horizon, 32)

        result = model(data_tensor.to(settings.device)).detach().flatten().tolist()

        inference_start_date = data.index[horizon * (x - 1)] - one_hour

        for y in range(0, horizon):
            inference = [result[(horizon - 1) - y], [inference_start_date - (one_hour * y)]]
            inference_set.append(inference)

    inference_set.reverse()

    return inference_set


def rec_inference(model, horizon, current_date, target_date, inference_set=None):
    return


def crop_result(start_date, end_date, inference_set):
    return


if __name__ == '__main__':
    app.run(debug=True)
