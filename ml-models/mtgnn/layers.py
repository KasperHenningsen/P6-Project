import torch
import numbers
from torch import nn
import torch.nn.functional as F
import settings


class GraphLearningLayer(nn.Module):
    def __init__(self, n_nodes, k, dim, alpha=3):
        super(GraphLearningLayer, self).__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        self.k = k  # number of neighbors to consider in the learned adjacency matrix
        self.alpha = alpha
        self.emb1 = nn.Embedding(n_nodes, dim, dtype=torch.float64)
        self.emb2 = nn.Embedding(n_nodes, dim, dtype=torch.float64)
        self.linear1 = nn.Linear(dim, dim, dtype=torch.float64)
        self.linear2 = nn.Linear(dim, dim, dtype=torch.float64)

    def forward(self, x):
        # retrieve node embeddings
        m1 = self.emb1(x)
        m2 = self.emb2(x)

        # perform forward pass
        M1 = torch.tanh(self.alpha * self.linear1(m1))
        M2 = torch.tanh(self.alpha * self.linear2(m2))

        A = torch.mm(M1, M2.transpose(1, 0)) - torch.mm(M2, M1.transpose(1, 0))
        A = F.relu(torch.tanh(self.alpha * A))

        # this is some MTGNN magic
        # for each node, select the top-k closest nodes. Set weights for non-neighbors to 0.
        mask = torch.zeros(x.size(0), x.size(0), dtype=torch.float64).to(settings.device)
        s1, t1 = (A + torch.rand_like(A) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask

        return A


class DilatedInceptionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_factor=2):
        super(DilatedInceptionLayer, self).__init__()
        # We add a convolution layer (filter) for each kernel size in kernel_set
        self.kernel_set = [2, 3, 6, 7]
        self.n_kernels = len(self.kernel_set)
        self.conv_layers = nn.ModuleList()
        out_channels = int(out_channels / self.n_kernels)
        for kernel_size in self.kernel_set:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), dilation=(1, dilation_factor), dtype=torch.float64))

    def forward(self, x):
        out = []
        # pass through each filter separately and append result to x
        for i in range(self.n_kernels):
            conv = self.conv_layers[i]
            out.append(conv(x))

        # the results in out will have different lengths, so we truncate them based on the size of the output
        # from the largest filter
        largest_filter_size = out[-1].size(3)
        for i in range(self.n_kernels):
            out[i] = out[i][..., -largest_filter_size:]

        # finally we concatenate the truncated results of each filter
        return torch.concat(out, dim=1)


class MixHopPropagationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, depth, alpha=3):
        super(MixHopPropagationLayer, self).__init__()
        self.depth = depth
        self.alpha = alpha

        self.graph_conv = GraphConvolutionLayer()
        # 1x1 convolution layer
        self.conv2d = nn.Conv2d(
            in_channels=(depth+1)*in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            dtype=torch.float64
        )

    def forward(self, x, A):
        # add 1 to the diagonal in A
        A = A + torch.eye(A.size(0)).to(settings.device)
        d = A.sum(dim=1)
        a = A / d.view(-1, 1)

        h = x
        out = [h]
        for i in range(self.depth):
            h = self.alpha*x + (1-self.alpha)*self.graph_conv(h, a)
            out.append(h)

        ho = torch.cat(out, dim=1)
        return self.conv2d(ho)


class GraphConvolutionLayer(nn.Module):
    def __init__(self):
        super(GraphConvolutionLayer, self).__init__()

    def forward(self, x, A):
        out = torch.einsum('ncwl,vw->ncvl', (x, A))
        return out.contiguous()


class LayerNormalization(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNormalization, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x, nodes):
        if self.elementwise_affine:
            return F.layer_norm(x, tuple(x.shape[1:]), self.weight[:, nodes, :], self.bias[:, nodes, :], self.eps)
        else:
            return F.layer_norm(x, tuple(x.shape[1:]), self.weight, self.bias, self.eps)
