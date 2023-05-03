import joblib
import pandas as pd
import numpy as np
import torch
import werkzeug.exceptions
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

from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)

featurematrix = os.path.join(settings.scripts_path, 'correlation_coefficient_matrix.csv')
data = get_processed_data(os.path.join(settings.data_path, 'open-weather-aalborg-2000-2022.csv'))


@app.route('/featurematrix')
def get_feature_matrix():
    """
    :return: CSV of feature correlations
    """
    return send_file(featurematrix)


@app.route('/actuals/full')
def get_data_full():
    """
    :return: A list of datetime/temp tuples over the entire dataset
    """
    dates = np.ndarray.tolist(data.index.to_pydatetime())
    temp = np.ndarray.tolist(data["temp"].values)

    response = list(zip(temp, dates))

    return make_response(response)


@app.route('/actuals/subset')
def get_data_subset():
    """
    :return: A subset list of the datetime/temp tuples over a date range
    """
    date_params = verify_date_params(request.args)

    data_subset = data[date_params[0]:date_params[1]]
    dates = np.ndarray.tolist(data_subset.index.to_pydatetime())
    temp = np.ndarray.tolist(data_subset["temp"].values)

    response = list(zip(temp, dates))

    return make_response(response)


@app.route('/actuals/dates')
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
    models = [f for f in os.listdir(settings.models_path)]

    return make_response(models)


@app.route('/predictions/models/cnn')
def get_cnn():
    start_date, end_date = verify_date_params(request.args)
    horizon = verify_horizon_param(request.args)

    cnn = get_model_object('cnn', horizon)
    response = get_inference_data(cnn, horizon, start_date, end_date)

    return make_response(response)


@app.route('/predictions/models/mlp')
def get_mlp():
    start_date, end_date = verify_date_params(request.args)
    horizon = verify_horizon_param(request.args)

    mlp = get_model_object('mlp', horizon)
    response = get_inference_data(mlp, horizon, start_date, end_date)

    return make_response(response)


@app.route('/predictions/models/gru')
def get_gru():
    start_date, end_date = verify_date_params(request.args)
    horizon = verify_horizon_param(request.args)

    gru = get_model_object('gru', horizon)
    response = get_inference_data(gru, horizon, start_date, end_date)

    return make_response(response)


@app.route('/predictions/models/rnn')
def get_rnn():
    start_date, end_date = verify_date_params(request.args)
    horizon = verify_horizon_param(request.args)

    rnn = get_model_object('rnn', horizon)
    response = get_inference_data(rnn, horizon, start_date, end_date)

    return make_response(response)


@app.route('/predictions/models/lstm')
def get_lstm():
    start_date, end_date = verify_date_params(request.args)
    horizon = verify_horizon_param(request.args)

    lstm = get_model_object('lstm', horizon)
    response = get_inference_data(lstm, horizon, start_date, end_date)

    return make_response(response)


@app.route('/predictions/models/tcn')
def get_tcn():
    start_date, end_date = verify_date_params(request.args)
    horizon = verify_horizon_param(request.args)

    tcn = get_model_object('tcn', horizon)
    response = get_inference_data(tcn, horizon, start_date, end_date)

    return make_response(response)


@app.route('/predictions/models/transformer')
def get_transformer():
    start_date, end_date = verify_date_params(request.args)
    horizon = verify_horizon_param(request.args)

    transformer = get_model_object('transformer', horizon)
    response = get_inference_data(transformer, horizon, start_date, end_date)

    return make_response(response)


@app.route('/predictions/models/mtgnn')
def get_mtgnn():
    start_date, end_date = verify_date_params(request.args)
    horizon = verify_horizon_param(request.args)

    mtgnn = get_model_object('mtgnn', horizon)
    response = get_inference_data(mtgnn, horizon, start_date, end_date)

    return make_response(response)


def verify_date_params(args):
    """
    Verifies that the date arguments given in the url is of the correct format
    :param args: A dict of arguments given in the url
    :return: An array of the verified arguments, where [0] is the input/output horizon and [1] and [2] are the start and end dates respectively.
    """
    start_date = args.get('start_date', type=datetime.fromisoformat)
    end_date = args.get('end_date', type=datetime.fromisoformat)

    if not (isinstance(start_date, datetime)):
        raise werkzeug.exceptions.BadRequest('Wrong format: start_date\n'
                                             'A valid format is e.g. 2000-01-01 00:00:00\n'
                                             'Might be missing leading zeros for the date')
    elif not (isinstance(end_date, datetime)):
        raise werkzeug.exceptions.BadRequest('Wrong format: end_date\n'
                                             'A valid format is e.g. 2000-01-01 00:00:00\n'
                                             'Might be missing leading zeros for the date')

    return start_date, end_date


def verify_horizon_param(args):
    """
    Verifies that the horizon argument is valid
    :param args: A dict of arguments given in the url
    :return: The verified horizon given that it is correct
    """
    horizon = args.get('horizon', type=int)

    valid_horizons = [12, 24, 48]
    if horizon not in valid_horizons:
        raise werkzeug.exceptions.BadRequest(f'Invalid horizon: {horizon}\n'
                                             f'Valid horizons include {valid_horizons}')

    return horizon


@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    return e, 400


def get_model_object(model, horizon):
    """
    Gets the model object based on model name as string and a horizon
    :param model: The models name in the form of a string
    :param horizon: The input/output horizon that the model is trained with
    :return: A model object
    """
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

    state_dict = torch.load(os.path.join(f'saved-models/{model.upper()}/horizon_{horizon}/model.pt'), map_location=settings.device)
    model_obj.load_state_dict(state_dict)
    model_obj.eval()

    return model_obj


def get_inference_data(model, horizon, start_date, end_date):
    """
    Initializes the inference of the data range, based on the start and end dates and crops the final result
    :param model: The model inferred from
    :param horizon: The input/output horizon
    :param start_date: The start date of the inferrence
    :param end_date: The end date of the inferrence
    :return: The final result of the inferrence
    """
    min_index = data.index.min()
    min_inference = min_index + timedelta(hours=horizon)

    max_index = data.index.max()
    max_inference = max_index - timedelta(hours=horizon)

    result = []

    if start_date < min_inference:
        result = infer_start(model, horizon)
        if end_date > min_inference:
            result += infer_range(model, horizon, min_inference, end_date, result)
    elif start_date > max_index:
        result = infer_range(model, horizon, max_inference, end_date)
    elif start_date >= min_inference:
        result = infer_range(model, horizon, start_date, end_date)

    result = crop_result(start_date, end_date, result)

    return result


def infer_start(model, horizon):
    """
    Backward infers the temperature values of the horizon before the dataset and the first horizon of the dataset, since these sets cannot be inferred forwards due to missing dataset values
    :param model: The NN model used
    :param horizon: The input/output horizon
    :return: The horizon preceding the dataset and the first horizon's worth of inference
    """
    result = []
    X_scaler, y_scaler = get_scalers(horizon)

    for x in range(2, 0, -1):
        input_data = data[horizon * (x - 1):horizon * x]
        input_data = input_data[::-1]

        inference_set = infer(model, horizon, input_data, X_scaler, y_scaler)

        inference_start_date = data.index[horizon * (x - 1)] - timedelta(hours=1)

        for y in range(0, horizon):
            inference = [inference_set[(horizon - 1) - y], inference_start_date - (timedelta(hours=1) * y)]
            result.append(inference)

    result.reverse()

    return result


def infer_range(model, horizon, start_date, end_date, result=None):
    """
    Infers horizon by horizon until the end date is reached
    :param model: The NN model used
    :param horizon: The input/output horizon
    :param start_date: The start date of the inference
    :param end_date: The end date of the inference
    :param result: The final set of inferences created in the function
    :return: A nested list of inferences where, [x][0] is the inferred temperature and [x][1] is the date of the temperature
    """
    if result is None:
        result = []

    X_scaler, y_scaler = get_scalers(horizon)

    current_date = start_date
    current_index = start_date - timedelta(hours=horizon)

    while current_index < end_date and current_index <= pd.to_datetime(data.index.max()):
        end_index = current_index + timedelta(hours=horizon)

        if end_index <= data.index.max():
            input_data = data[current_index:end_index]

            inference_set = infer(model, horizon, input_data, X_scaler, y_scaler)
        else:
            input_data = data[data.index.max() - timedelta(hours=horizon):data.index.max()]

            inference_set = infer(model, horizon, input_data, X_scaler, y_scaler)
            inference_set = inference_set[-(horizon - end_index.hour):]

        for y in range(0, len(inference_set)):
            inference = [inference_set[y], current_index + timedelta(hours=horizon) + (timedelta(hours=1) * y)]
            result.append(inference)

        current_date += timedelta(hours=horizon)
        current_index += timedelta(hours=horizon)

    return result


def get_scalers(horizon):
    X_scaler = joblib.load(os.path.join(settings.scalers_path, f'horizon_{horizon}', 'X_scaler.gz'))
    y_scaler = joblib.load(os.path.join(settings.scalers_path, f'horizon_{horizon}', 'y_scaler.gz'))

    return X_scaler, y_scaler


def infer(model, horizon, input_data, X_scaler, y_scaler):
    """
    Infers temp values based on given model, horizon and input data
    :return: A scaled list of inferences
    """
    input_data = X_scaler.transform(input_data.values)

    data_tensor = torch.from_numpy(input_data)[:horizon]
    data_tensor = data_tensor.reshape(1, horizon, 32).to(settings.device)

    result = model(data_tensor).cpu().detach()
    result = y_scaler.inverse_transform(result).flatten()

    return result


def crop_result(start_date, end_date, inference_set):
    """
    Crops the overflow and underflow of the inference set using the dates present in the nested list
    :param start_date: The start date used to remove earlier inferences
    :param end_date: The end date used to remove later inferences
    :param inference_set: The set of inferences
    :return: The cropped inference set
    """
    start_index = 0
    end_index = 0

    for i, inference in enumerate(inference_set):
        if start_date in inference:
            start_index = i
        elif end_date in inference:
            end_index = i

    result = inference_set[start_index:end_index + 1]

    return result


if __name__ == '__main__':
    app.run(debug=True)
