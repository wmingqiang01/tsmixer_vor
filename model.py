import torch
import torch.nn as nn

class TSBatchNorm2d(nn.Module):
    def __init__(self):
        super(TSBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        x = x.unsqueeze(1)  # Reshape to (batch_size, 1, timepoints, features)
        output = self.bn(x)
        output = output.squeeze(1)  # Reshape back to (batch_size, timepoints, features)
        return output

class TSTimeMixingResBlock(nn.Module):
    def __init__(self, width_time: int, dropout: float):
        super(TSTimeMixingResBlock, self).__init__()
        self.norm = TSBatchNorm2d()
        self.lin = nn.Linear(in_features=width_time, out_features=width_time)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)
        y = torch.transpose(y, 1, 2)  # Shape: (batch_size, features, time)
        y = self.lin(y)
        y = self.act(y)
        y = torch.transpose(y, 1, 2)  # Shape: (batch_size, time, features)
        y = self.dropout(y)
        return x + y

class TSFeatMixingResBlock(nn.Module):
    def __init__(self, width_feats: int, width_feats_hidden: int, dropout: float):
        super(TSFeatMixingResBlock, self).__init__()
        self.norm = TSBatchNorm2d()
        self.lin_1 = nn.Linear(in_features=width_feats, out_features=width_feats_hidden)
        self.lin_2 = nn.Linear(in_features=width_feats_hidden, out_features=width_feats)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)
        y = self.lin_1(y)
        y = self.act(y)
        y = self.dropout_1(y)
        y = self.lin_2(y)
        y = self.dropout_2(y)
        return x + y

class TSMixingLayer(nn.Module):
    def __init__(self, input_length: int, no_feats: int, feat_mixing_hidden_channels: int, dropout: float):
        super(TSMixingLayer, self).__init__()
        self.time_mixing = TSTimeMixingResBlock(width_time=input_length, dropout=dropout)
        self.feat_mixing = TSFeatMixingResBlock(width_feats=no_feats, width_feats_hidden=feat_mixing_hidden_channels, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.time_mixing(x)
        y = self.feat_mixing(y)
        return y

class TSTemporalProjection(nn.Module):
    def __init__(self, input_length: int, forecast_length: int):
        super(TSTemporalProjection, self).__init__()
        self.lin = nn.Linear(in_features=input_length, out_features=forecast_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.transpose(x, 1, 2)  # Shape: (batch_size, features, time=input_length)
        y = self.lin(y)  # Shape: (batch_size, features, time=forecast_length)
        y = torch.transpose(y, 1, 2)  # Shape: (batch_size, time=forecast_length, features)
        return y

class TSMixerModelExclRIN(nn.Module):
    def __init__(self, input_length: int, forecast_length: int, no_feats: int, feat_mixing_hidden_channels: int, no_mixer_layers: int, dropout: float):
        super(TSMixerModelExclRIN, self).__init__()
        self.temp_proj = TSTemporalProjection(input_length=input_length, forecast_length=forecast_length)
        mixer_layers = []
        for _ in range(no_mixer_layers):
            mixer_layers.append(TSMixingLayer(input_length=input_length, no_feats=no_feats, feat_mixing_hidden_channels=feat_mixing_hidden_channels, dropout=dropout))
        self.mixer_layers = nn.ModuleList(mixer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.temp_proj(x)
        return x

class TSMixerModel(nn.Module):
    def __init__(self, input_length: int, forecast_length: int, no_feats: int, feat_mixing_hidden_channels: int, no_mixer_layers: int, dropout: float, eps: float = 1e-8):
        super(TSMixerModel, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(no_feats))
        self.shift = nn.Parameter(torch.zeros(no_feats))
        self.ts = TSMixerModelExclRIN(
            input_length=input_length, 
            forecast_length=forecast_length, 
            no_feats=no_feats, 
            feat_mixing_hidden_channels=feat_mixing_hidden_channels,
            no_mixer_layers=no_mixer_layers,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.scale + self.shift
        x = self.ts(x)
        x = (x - self.shift) / self.scale
        x = x * torch.sqrt(var + self.eps) + mean
        return x

class TSMixerClassifier(nn.Module):
    def __init__(self, input_length: int, no_feats: int, feat_mixing_hidden_channels: int, no_mixer_layers: int, dropout: float):
        super(TSMixerClassifier, self).__init__()
        self.ts_mixer = TSMixerModel(
            input_length=input_length,
            forecast_length=1,  # Predict 1 time step
            no_feats=no_feats,
            feat_mixing_hidden_channels=feat_mixing_hidden_channels,
            no_mixer_layers=no_mixer_layers,
            dropout=dropout
        )
        self.classifier = nn.Linear(no_feats, 1)  # Map (batch_size, 1, no_feats) to (batch_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ts_mixer(x)  # Shape: (batch_size, 1, no_feats)
        x = x.squeeze(1)  # Shape: (batch_size, no_feats)
        x = self.classifier(x)  # Shape: (batch_size, 1)
        return x.squeeze(-1)  # Shape: (batch_size,)