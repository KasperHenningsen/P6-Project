import datetime
from os import path

from torch.nn import RNN

import settings
from glob import glob
import json
from baselines.cnn import ConvolutionalNet
from baselines.gru import GatedRecurrentUnitNet
from baselines.lstm import LongShortTermMemoryNet
from baselines.mlp import MultiLayerPerceptronNet
from baselines.rnn import RecurrentNeuralNet
from baselines.tcn import TemporalConvolutionNet
from baselines.transformer import TransformerModel
from mtgnn.mtgnn import MultiTaskGraphNeuralNet


def set_next_save_path(model):
    base_path = path.join(settings.models_path, model.get_name())
    folders = glob(path.join(base_path, '*/'), recursive=False)
    if len(folders) == 0:
        model.path = path.join(base_path, "run_0")
    else:
        latest_folder = folders[-1]
        run_no = int(latest_folder.split('\\')[-2].split('_')[-1]) + 1
        model.path = path.join(base_path, f'run_{run_no}')


def generate_train_test_log(model, train_losses, test_losses, seq_len, target_len, target_col, batch_size, epochs, learning_rate, train_size, grad_clipping, train_time):
    if train_time is not None:
        train_time = int(train_time)
    json_obj = {
        'model': model.get_name(),
        'trained_at': datetime.datetime.now().isoformat(),
        'train_time_seconds': train_time,
        'loss': {
            'train': {
                'mae': train_losses[0],
                'mape': train_losses[1],
                'rmse': train_losses[2]
            },
            'test': {
                'mae': test_losses[0],
                'mape': test_losses[1],
                'rmse': test_losses[2]
            }
        },
        'hyperparameters': {
            'seq_len': seq_len,
            'target_len': target_len,
            'target_col': target_col,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'train_size': train_size,
            'grad_clipping': grad_clipping
        },
        'model_parameters': get_model_params(model)
    }
    with open(path.join(model.path, "log.json"), mode='x') as f:
        json.dump(json_obj, fp=f, indent=2)


def get_model_params(model) -> object:
    if isinstance(model, ConvolutionalNet):
        return {
            'input_channels': model.input_channels,
            'hidden_size': model.hidden_size,
            'kernel_size': model.kernel_size,
            'dropout': model.dropout_prob
        }
    elif isinstance(model, LongShortTermMemoryNet):
        return {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'output_size': model.output_size,
            'num_layers': model.num_layers,
            'dropout': model.dropout
        }
    elif isinstance(model, RecurrentNeuralNet):
        return {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'output_size': model.output_size,
            'num_layers': model.num_layers,
            'dropout': model.dropout,
            'nonlinearity': model.nonlinearity
        }
    elif isinstance(model, MultiLayerPerceptronNet):
        return {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'output_size': model.output_size
        }
    elif isinstance(model, TemporalConvolutionNet):
        return {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'output_size': model.output_size,
            'depth': model.depth,
            'kernel_size': model.kernel_size,
            'dilation_base': model.dilation_base
        }
    elif isinstance(model, GatedRecurrentUnitNet):
        return {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'output_size': model.output_size,
            'num_layers': model.num_layers,
            'dropout': model.dropout
        }
    elif isinstance(model, TransformerModel):
        return {
            'input_size': model.input_size,
            'output_size': model.output_size,
            'num_layers': model.num_layers,
            'd_model': model.d_model,
            'nhead': model.nhead,
            'dim_feedforward': model.dim_feedforward,
            'dropout': model.dropout
        }
    elif isinstance(model, MultiTaskGraphNeuralNet):
        return {
            'seq_length': model.seq_length,
            'num_features': model.num_features,
            'num_layers': model.num_layers,
            'dropout': model.dropout,
            'use_output_convolution': model.use_output_convolution,
            'build_adj_matrix': model.build_adj_matrix,
            'conv_channels': model.conv_channels,
            'residual_channels': model.residual_channels,
            'skip_channels': model.skip_channels,
            'end_channels': model.end_channels,
            'tan_alpha': model.tan_alpha,
            'prop_alpha': model.prop_alpha
        }
    else:
        return None
