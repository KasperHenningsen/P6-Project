import os

import torch
from torch import nn
import torch.nn.functional as F

import settings
from .layers import GraphLearningLayer, MixHopPropagationLayer, DilatedInceptionLayer, LayerNormalization


class MTGNN(nn.Module):
    def __init__(self,
                 num_features,
                 seq_length,
                 num_layers=3,
                 conv_channels=32,
                 residual_channels=16,
                 skip_channels=64,
                 end_channels=128,
                 build_adj_matrix=True,
                 tan_alpha=3,
                 prop_alpha=0.05,
                 dropout=0.2,
                 use_output_convolution=True
                 ):
        super(MTGNN, self).__init__()
        input_size = 1  # in the MTGNN our time-series data is always 1-dimensional (each feature at time-step i is just a single value)
        self.path = os.path.join(settings.models_path, self.get_name())
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_output_convolution = use_output_convolution
        self.build_adj_matrix = build_adj_matrix
        self.nodes = torch.arange(num_features).to(settings.device)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs_1 = nn.ModuleList()
        self.graph_convs_2 = nn.ModuleList()
        self.normalization = nn.ModuleList()

        # 1x1 convolution
        self.start_conv = nn.Conv2d(in_channels=input_size, out_channels=residual_channels, kernel_size=(1, 1), dtype=torch.float64)

        subgraph_size, node_dim = 20, 40
        graph_conv_network_depth = 2
        self.graph_learning_layer = GraphLearningLayer(num_features, k=subgraph_size, dim=node_dim, alpha=tan_alpha)

        kernel_size = 7
        dilation_exponential = 2
        # receptive field = how much of the input sequence is actually used for forecasting
        self.receptive_field = int(1 + (kernel_size-1) * (dilation_exponential ** num_layers-1) / (dilation_exponential-1))

        dilation = 1
        for i in range(1, num_layers+1):
            receptive_field_i = int(1 + (kernel_size-1) * (dilation_exponential ** i-1) / (dilation_exponential-1))

            self.filter_convs.append(DilatedInceptionLayer(residual_channels, conv_channels, dilation_factor=dilation))
            self.gate_convs.append(DilatedInceptionLayer(residual_channels, conv_channels, dilation_factor=dilation))
            self.residual_convs.append(nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1), dtype=torch.float64))

            z = self.seq_length-receptive_field_i+1 \
                if seq_length > self.receptive_field \
                else self.receptive_field-receptive_field_i+1
            self.skip_convs.append(nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, z), dtype=torch.float64))
            self.normalization.append(LayerNormalization((residual_channels, num_features, z), elementwise_affine=False))

            if build_adj_matrix:
                self.graph_convs_1.append(MixHopPropagationLayer(conv_channels, residual_channels, depth=graph_conv_network_depth, alpha=prop_alpha))
                self.graph_convs_2.append(MixHopPropagationLayer(conv_channels, residual_channels, depth=graph_conv_network_depth, alpha=prop_alpha))

            dilation *= dilation_exponential

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), dtype=torch.float64)
        self.end_conv_2 = nn.Conv2d(end_channels, seq_length, kernel_size=(1, 1), dtype=torch.float64)
        self.output_conv = nn.Conv1d(num_features, 1, kernel_size=1, dtype=torch.float64)

        if seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(input_size, skip_channels, kernel_size=(1, seq_length), dtype=torch.float64)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, seq_length-self.receptive_field+1), dtype=torch.float64)
        else:
            self.skip0 = nn.Conv2d(input_size, skip_channels, kernel_size=(1, self.receptive_field), dtype=torch.float64)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1), dtype=torch.float64)

        self.to(settings.device)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = x.transpose(2, 3)
        seq_len = x.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            x = F.pad(x, (self.receptive_field-self.seq_length, 0, 0, 0))

        adp = self.graph_learning_layer(self.nodes) if self.build_adj_matrix else None

        out = self.start_conv(x)
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        for i in range(self.num_layers):
            residual = out
            filter = torch.tanh(self.filter_convs[i](out))
            gate = torch.sigmoid(self.gate_convs[i](out))
            out = F.dropout(filter * gate, self.dropout, training=self.training)

            s = self.skip_convs[i](out)
            skip = skip + s

            if adp is not None:
                gc1 = self.graph_convs_1[i](out, adp)
                gc2 = self.graph_convs_2[i](out, adp.transpose(1, 0))
                out = gc1 + gc2
            else:
                out = self.residual_convs[i](out)

            out = out + residual[:, :, :, -out.size(3):]
            out = self.normalization[i](out, self.nodes)

        skip = skip + self.skipE(out)
        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)

        # at this point the output shape is (batch_size, 1, time_steps, features)
        out = torch.squeeze(out)  # reshape to (batch_size, time_steps, features)

        """ here we can try either picking just the first element in the last dimension, or do a 1x1 convolution to reduce it to 1
        - some ChatGPT comments on this:
        
        Taking just the first feature in each time-step in the output tensor could work in some cases, but whether it yields 
        similar results to using a convolutional layer depends on the specific problem and dataset you're working with.
        
        If the target feature you're interested in is always the first feature in each time-step of the output tensor, 
        then taking just the first feature would give you the information you need. However, if the position of the target 
        feature changes within each time-step, or if there is noise or other features that are correlated with the target feature, 
        then taking just the first feature may not be sufficient.
        
        Using a convolutional layer, on the other hand, can help capture correlations between different features in 
        each time-step and across time-steps. It can also help reduce noise and extract more meaningful features. 
        However, adding a convolutional layer also increases the complexity and training time of your model, so it's
        important to weigh the benefits and drawbacks for your specific use case.
        """

        if self.use_output_convolution:
            # transpose to (batch_size, features, time_steps) to do 1x1 convolution along feature dimension
            out = out.transpose(1, 2)
            out = self.output_conv(out)
            out = out.transpose(1, 2)  # transpose back to (batch_size, time_steps, 1)
        else:
            # for each time-step we take only the first feature (the target feature)
            out = out[:, :, :1]

        return torch.squeeze(out)  # squeeze to (batch_size, time_steps)

    def load_saved_model(self):
        state_dict = torch.load(os.path.join(self.path, 'model.pt'))
        self.load_state_dict(state_dict)
        self.eval()

    def get_name(self):
        return self.__class__.__name__
