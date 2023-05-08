import torch.cuda

use_cuda_if_available = True
if torch.cuda.is_available() and use_cuda_if_available:
    device = torch.device('cuda')
    torch.set_default_device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

models_path = '.\\saved-models'
plots_path = '.\\plots'
data_path = '.\\data'
scripts_path = '.\\scripts'
