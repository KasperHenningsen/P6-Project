import torch.cuda

use_cuda_if_available = True
if torch.cuda.is_available() and use_cuda_if_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Training on device: {device}')

models_path = '.\\saved-models'
plots_path = '.\\plots'
