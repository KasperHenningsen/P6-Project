import os

import matplotlib.pyplot as plt
import torch

import settings


def plot(model_or_models, X_vals, y, start=0, end=2000, step=1):
    if isinstance(model_or_models, tuple):
        multiplot(model_or_models, X_vals, y, start, end, step)
        return

    model = model_or_models
    model.load_saved_model()

    y = y[start:end:step]
    print(f'Plotting {model.get_name()} predictions')

    yhat = model(torch.tensor(X_vals[start:end:step]).to(settings.device))
    yhat = yhat.cpu().detach()

    if yhat.shape != y.shape:
        raise ValueError(
            f"The predicted and actual tensors must have the same shape. Shapes are {yhat.shape} and {y.shape}")

    plt.figure(figsize=(15, 5))
    plt.plot(yhat.reshape(-1), label='predicted')
    plt.plot(y.reshape(-1), label='actual')
    plt.xlabel('Time')
    plt.ylabel('Temp (°C)')
    plt.legend()
    plt.grid()

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
        plt.plot(yhat.reshape(-1), label='predicted')
        plt.plot(y.reshape(-1), label='actual')
        plt.xlabel('Time')
        plt.ylabel('Temp (°C)')
        plt.title(f"{model.get_name()}")
        plt.legend()
        plt.grid()

    plt.tight_layout()

    save_path = os.path.join(settings.plots_path, f'{figure_name}.png')
    plt.savefig(save_path)
    plt.show()
    print(f'Done! Plot saved at {save_path}')