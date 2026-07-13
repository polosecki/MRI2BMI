import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainToBMINet(nn.Module):
    """CNN for predicting BMI from 3D brain MRI using soft-label KL-divergence training.

    Outputs a log-probability distribution over binned BMI values.
    """

    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=1, dropout=True):
        """
        Args:
            channel_number: list of channel widths for each conv block
            output_dim: number of output bins (e.g. 50 for BMI range 10-60 with step 1)
            dropout: whether to apply dropout (p=0.5) before the final layer
        """
        super(BrainToBMINet, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    'conv_%d' % i,
                    self.conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1))
            else:
                self.feature_extractor.add_module(
                    'conv_%d' % i,
                    self.conv_layer(in_channel, out_channel, maxpool=False, kernel_size=1, padding=0))
        self.classifier = nn.Sequential()
        self.classifier.add_module('average_pool', nn.AvgPool3d([5, 6, 5]))
        if dropout:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        self.classifier.add_module(
            'conv_%d' % n_layer,
            nn.Conv3d(channel_number[-1], output_dim, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        return [x]


class BrainToDiseaseNet(nn.Module):
    """CNN for end-to-end multi-label disease prediction from 3D brain MRI.

    Uses an adaptive average pool so the network accepts arbitrary input sizes,
    followed by a linear prediction head with BCEWithLogitsLoss at training time.
    """

    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=1):
        """
        Args:
            channel_number: list of channel widths for each conv block
            output_dim: number of disease labels to predict
        """
        super(BrainToDiseaseNet, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    'conv_%d' % i,
                    self.conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1))
            else:
                self.feature_extractor.add_module(
                    'conv_%d' % i,
                    self.conv_layer(in_channel, out_channel, maxpool=False, kernel_size=1, padding=0))
        self.feature_extractor.add_module('adaptive_average_pool', nn.AdaptiveAvgPool3d(1))
        self.prediction_head = nn.Linear(channel_number[-1], output_dim)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.prediction_head(x)
        return x
