import os

import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter1d

import settings


def generate_plot(yhat, y, model):
    plt.plot(yhat.reshape(-1), label='predicted')
    plt.plot(y.reshape(-1), label='actual')
    plt.xlabel('Time')
    plt.ylabel('Temp (°C)')
    plt.title(f"{model.get_name()}")
    plt.legend()
    plt.grid()


def plot(model_or_models, X_vals, y, start=0, end=2000, step=1):
    if isinstance(model_or_models, tuple):
        multiplot(model_or_models, X_vals, y, start, end, step)
        return

    model = model_or_models
    model.load_saved_model()

    y = y[start:end:step]
    print(f'Plotting {model.get_name()} predictions')
    plt.figure(figsize=(15, 5))

    yhat = model(torch.tensor(X_vals[start:end:step]).to(settings.device))
    yhat = yhat.cpu().detach()

    if yhat.shape != y.shape:
        raise ValueError(
            f"The predicted and actual tensors must have the same shape. Shapes are {yhat.shape} and {y.shape}")

    generate_plot(yhat, y, model)

    # TODO: Infer names (see comment in multiplot) - also, determine if plots should be saved in plots folder instead
    save_path = os.path.join(model.path, 'plot.png')
    plt.savefig(save_path)
    plt.show()
    print(f'Done! Plot saved at {save_path}')


def multiplot(models, X_vals, y, start=0, end=2000, step=1):
    num_models = len(models)
    if num_models < 1:
        raise ValueError("At least one model must be provided")

    for model in models:
        model.load_saved_model()

    y = y[start:end:step]

    # TODO: Make get_name() function for plots, which infers a name from model params and start, end and step for plot
    model_names = map(lambda m: m.get_name(), models)
    figure_name = ", ".join(s for s in model_names)
    print(f'Plotting {figure_name} predictions')

    plt.figure(figsize=(15, 5 * num_models))

    for i, model in enumerate(models):
        yhat = model(torch.tensor(X_vals[start:end:step]).to(settings.device))
        yhat = yhat.cpu().detach()

        if yhat.shape != y.shape:
            raise ValueError(
                f"The predicted and actual tensors must have the same shape. Shapes are {yhat.shape} and {y.shape}")

        plt.subplot(num_models, 1, i + 1)
        generate_plot(yhat, y, model)

    plt.tight_layout()

    save_path = os.path.join(settings.plots_path, f'{figure_name}.png')
    plt.savefig(save_path)
    plt.show()
    print(f'Done! Plot saved at {save_path}')


def plot_rbf_small(df):
    # Interpolate missing values in columns 20 to 32
    cols_to_interpolate = df.columns[20:26]
    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear', limit_direction='forward', axis=0)

    # Apply a Gaussian filter to columns 20 to 32
    df[cols_to_interpolate] = gaussian_filter1d(df[cols_to_interpolate], sigma=8, axis=0)

    plt.rcParams['figure.figsize'] = [7, 3.5]
    plt.figure().tight_layout()
    df.iloc[:4344, 20:26].plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6)
    plt.xlabel("")

    save_path = os.path.join(settings.plots_path, 'RBF small.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_rbf_large(df):
    # Interpolate missing values in columns 20 to 32
    cols_to_interpolate = df.columns[20:32]
    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear', limit_direction='forward', axis=0)

    # Apply a Gaussian filter to columns 20 to 32
    df[cols_to_interpolate] = gaussian_filter1d(df[cols_to_interpolate], sigma=8, axis=0)

    plt.rcParams['figure.figsize'] = [13.75, 4]
    plt.figure().tight_layout()
    df.iloc[:8760, 20:32].plot()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=12)
    plt.xlabel("")

    save_path = os.path.join(settings.plots_path, 'RBF large.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_loss_history(model, losses):
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Mean Absolute Error)")
    plt.plot(losses)
    plt.savefig(os.path.join(model.path, 'plot-train-loss.png'))
