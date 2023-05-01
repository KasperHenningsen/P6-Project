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
    params = verify_params(request.args)

    cnn = get_model_object('cnn', params[0])
    response = get_inference_data(cnn, params[0], params[1], params[2])

    return make_response(response)


@app.route('/predictions/models/mlp')
def get_mlp():
    params = verify_params(request.args)

    mlp = get_model_object('mlp', params[0])
    response = get_inference_data(mlp, params[0], params[1], params[2])

    return make_response(response)


@app.route('/predictions/models/gru')
def get_gru():
    params = verify_params(request.args)

    gru = get_model_object('gru', params[0])
    response = get_inference_data(gru, params[0], params[1], params[2])

    return make_response(response)


@app.route('/predictions/models/rnn')
def get_rnn():
    params = verify_params(request.args)

    rnn = get_model_object('rnn', params[0])
    response = get_inference_data(rnn, params[0], params[1], params[2])

    return make_response(response)


@app.route('/predictions/models/lstm')
def get_lstm():
    params = verify_params(request.args)

    lstm = get_model_object('lstm', params[0])
    response = get_inference_data(lstm, params[0], params[1], params[2])

    return make_response(response)


@app.route('/predictions/models/tcn')
def get_tcn():
    params = verify_params(request.args)

    tcn = get_model_object('tcn', params[0])
    response = get_inference_data(tcn, params[0], params[1], params[2])

    return make_response(response)


@app.route('/predictions/models/transformer')
def get_transformer():
    params = verify_params(request.args)

    transformer = get_model_object('transformer', params[0])
    response = get_inference_data(transformer, params[0], params[1], params[2])

    return make_response(response)


@app.route('/predictions/models/mtgnn')
def get_mtgnn():
    params = verify_params(request.args)

    mtgnn = get_model_object('mtgnn', params[0])
    response = get_inference_data(mtgnn, params[0], params[1], params[2])

    return make_response(response)


def verify_params(args):
    horizon = args.get('horizon', type=int)
    start_date = args.get('start_date', type=to_date)
    end_date = args.get('end_date', type=to_date)

    if not (isinstance(start_date, datetime.date)):
        return "Wrong format: start_date", 400
    elif not (isinstance(end_date, datetime.date)):
        return "Wrong format: end_date", 400

    return [horizon, start_date, end_date]


# TODO: Might remove timezone info from dates
def to_date(string):
    date = datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S %z %Z").replace(tzinfo=None)
    return date


def get_model_object(model, horizon):
    model_obj = None
    if model == 'mtgnn':
        model_json = json.load(open(f'{settings.models_path}\\MTGNN\\horizon_{horizon}\\log.json'))
        model_params = model_json['model_parameters']

        num_features = model_params['model_parameters']['num_features']
        seq_length = model_params['model_parameters']['seq_length']
        num_layers = model_params['model_parameters']['num_layers']
        subgraph_size = model_params['model_parameters']['subgraph_size']
        subgraph_node_dim = model_params['model_parameters']['subgraph_node_dim']
        use_output_convolution = model_params['model_parameters']['use_output_convolution']
        dropout = model_params['model_parameters']['dropout']

        model_obj = MTGNN(num_features, seq_length, num_layers, subgraph_size, subgraph_node_dim,
                          use_output_convolution, dropout)
    else:
        model_json = json.load(open(f'{settings.models_path}\\{model.upper()}\\horizon_{horizon}\\log.json'))
        model_params = model_json['model_parameters']

        if model == 'cnn':
            input_channels = model_params['input_channels']
            hidden_size = model_params['hidden_size']
            kernel_size = model_params['kernel_size']
            dropout_prob = model_params['dropout']

            model_obj = CNN(input_channels, hidden_size, kernel_size, dropout_prob)
        elif model == 'mlp':
            input_size = model_params['input_size']
            hidden_size = model_params['hidden_size']
            output_size = model_params['output_size']
            num_layers = model_params['num_layers']
            seq_length = model_params['seq_length']

            model_obj = MLP(input_size, hidden_size, output_size, num_layers, seq_length)
        elif model == 'gru':
            input_size = model_params['input_size']
            hidden_size = model_params['hidden_size']
            output_size = model_params['output_size']
            dropout_prob = model_params['dropout']
            num_layers = model_params['num_layers']

            model_obj = GRU(input_size, hidden_size, output_size, dropout_prob, num_layers)
        elif model == 'rnn':
            input_size = model_params['input_size']
            hidden_size = model_params['hidden_size']
            output_size = model_params['output_size']
            dropout_prob = model_params['dropout']
            num_layers = model_params['num_layers']

            model_obj = RNN(input_size, hidden_size, output_size, dropout_prob, num_layers)
        elif model == 'lstm':
            input_size = model_params['input_size']
            hidden_size = model_params['hidden_size']
            output_size = model_params['output_size']
            dropout_prob = model_params['dropout']
            num_layers = model_params['num_layers']

            model_obj = LSTM(input_size, hidden_size, output_size, dropout_prob, num_layers)
        elif model == 'tcn':
            input_size = model_params['input_size']
            output_size = model_params['output_size']
            hidden_size = model_params['hidden_size']

            model_obj = TCN(input_size, output_size, hidden_size)
        elif model == 'transformer':
            input_size = model_params['input_size']
            d_model = model_params['d_model']
            nhead = model_params['nhead']
            num_layers = model_params['num_layers']
            output_size = model_params['output_size']
            dropout = model_params['dropout']

            model_obj = Transformer(input_size, d_model, nhead, num_layers, output_size, dropout)

    state_dict = torch.load(os.path.join(f'saved-models/{model.upper()}/horizon_{horizon}/model.pt'))
    model_obj.load_state_dict(state_dict)
    model_obj.eval()

    return model_obj


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
            result += inference(model, horizon, min_inference, end_date, result)
    elif start_date > max_index:
        result = inference(model, horizon, max_inference, end_date)
    elif start_date >= min_inference:
        result = inference(model, horizon, start_date, end_date)

    result = crop_result(start_date, end_date, result)

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
            inference_res = [result[(horizon - 1) - y], inference_start_date - (one_hour * y)]
            inference_set.append(inference_res)

    # TODO: Might not be correct, would be easier to verify with less sporadic models
    inference_set.reverse()

    return inference_set


def inference(model, horizon, start_date, end_date, inference_set=None):
    if inference_set is None:
        inference_set = []

    one_hour = datetime.timedelta(hours=1)
    time_horizon = datetime.timedelta(hours=horizon)

    current_date = start_date - time_horizon

    inference_step = 1
    while current_date + time_horizon < end_date and current_date <= pd.to_datetime(
            data.index.max()) or inference_step == 1:
        start_index = current_date
        end_index = start_index + time_horizon - one_hour
        input_data = data[start_index:end_index]

        data_tensor = torch.from_numpy(input_data.to_numpy())[:horizon]
        data_tensor = data_tensor.reshape(1, horizon, 32)

        result = model(data_tensor.to(settings.device)).detach().flatten().tolist()

        for y in range(0, horizon):
            inference_res = [result[y], start_index + time_horizon + (one_hour * y)]
            inference_set.append(inference_res)

        current_date += time_horizon
        inference_step += 1

    return inference_set


def crop_result(start_date, end_date, inference_set):
    start_index = 0
    end_index = 0

    for i, inference_res in enumerate(inference_set):
        if start_date in inference_res:
            start_index = i
        elif end_date in inference_res:
            end_index = i

    result = inference_set[start_index:end_index + 1]

    return result


if __name__ == '__main__':
    app.run(debug=True)
