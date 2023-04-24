import torch
from flask import Flask, request, send_file, make_response
import os
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

app = Flask(__name__)

ML_PATH = "./saved-models"


@app.route('/featurematrix')
def get_feature_matrix():
    """
    Retrieve feature correlation coefficient matrix
    :return: CSV of feature correlations
    """
    path = "scripts/correlation_coefficient_matrix.csv"
    if os.path.isfile(path):
        return send_file(path)
    else:
        return "File not found", 404


# TODO: Remove?
@app.route('/dataset')
def get_data():
    path = "data/open-weather-aalborg-2000-2022.csv"
    if os.path.isfile(path):
        return send_file(path)
    else:
        return "File not found", 404


@app.route('/predictions/models')
def get_models():
    models = [f.split('.')[0] for f in os.listdir('./baselines') if f != '__init__.py']
    models.pop()
    models.append('mtgnn')

    models.sort()

    return make_response(models)


def to_date(string):
    return datetime.datetime.strptime(string, "%Y-%m-%d").date()


def get_model_object(model, horizon):
    model_json = json.load(open(ML_PATH + f'/{model.upper()}/horizon_{horizon}/log.json'))
    model_params = model_json['model_parameters']

    match model:
        case 'cnn':
            input_channels = model_params['input_channels']
            hidden_size = model_params['hidden_size']
            kernel_size = model_params['kernel_size']
            dropout_prob = model_params['dropout']

            return CNN(input_channels, hidden_size, kernel_size, dropout_prob)
        case 'mlp':
            input_size = model_params['input_size']
            hidden_size = model_params['hidden_size']
            output_size = model_params['output_size']
            num_layers = model_params['num_layers']
            seq_length = model_params['seq_length']

            return MLP(input_size, hidden_size, output_size, num_layers, seq_length)
        case 'gru':
            input_size = model_params['input_size']
            hidden_size = model_params['hidden_size']
            output_size = model_params['output_size']
            dropout_prob = model_params['dropout']
            num_layers = model_params['num_layers']

            return GRU(input_size, hidden_size, output_size, dropout_prob, num_layers)
        case 'rnn':
            input_size = model_params['input_size']
            hidden_size = model_params['hidden_size']
            output_size = model_params['output_size']
            dropout_prob = model_params['dropout']
            num_layers = model_params['num_layers']

            return RNN(input_size, hidden_size, output_size, dropout_prob, num_layers)
        case 'lstm':
            input_size = model_params['input_size']
            hidden_size = model_params['hidden_size']
            output_size = model_params['output_size']
            dropout_prob = model_params['dropout']
            num_layers = model_params['num_layers']

            return LSTM(input_size, hidden_size, output_size, dropout_prob, num_layers)
        case 'tcn':
            input_size = model_params['input_size']
            output_size = model_params['output_size']
            hidden_size = model_params['hidden_size']

            return TCN(input_size, output_size, hidden_size)
        case 'transformer':
            input_size = model_params['input_size']
            d_model = model_params['d_model']
            nhead = model_params['nhead']
            num_layers = model_params['num_layers']
            output_size = model_params['output_size']
            dropout = model_params['dropout']

            return Transformer(input_size, d_model, nhead, num_layers, output_size, dropout)


if __name__ == '__main__':
    app.run(debug=True)
