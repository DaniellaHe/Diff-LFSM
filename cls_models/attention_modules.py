from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_attention(nn.Module):
    def __init__(self, nf_i, nf_o, kernel, stride, padding, dilation, bias, dropout_rate):
        super(conv_attention, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=nf_i,
            out_channels=nf_o,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )

        self.bn = nn.BatchNorm2d(
            track_running_stats=True, affine=True, num_features=nf_o, eps=1e-5
        )

        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        return out


class Spatial_attention(nn.Module):
    def __init__(self, nf_i, nf_o, kernel, stride, padding, dilation, bias, dropout_rate):
        super(Spatial_attention, self).__init__()
        self.att_conv = conv_attention(nf_i, nf_o, kernel, stride, padding, dilation, bias, dropout_rate)

    def forward(self, sal):
        sal_avg = torch.max(sal, 1)[0].unsqueeze(1)
        sal_avg = self.att_conv(sal_avg)
        soft_attention = torch.sigmoid(sal_avg)
        return soft_attention


class SliceAttention(nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_units, dropout_rate):
        super(SliceAttention, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.dropout_rate = dropout_rate
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)
        out = dummy_x.view(dummy_x.shape[0], -1)

        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_hidden_layers):
            self.layer_dict["fcc_{}".format(i)] = nn.Linear(
                in_features=out.shape[1], out_features=self.num_hidden_units, bias=True
            )
            out = F.leaky_relu(self.layer_dict["fcc_{}".format(i)].forward(out))
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        self.layer_dict["attention_layer"] = nn.Linear(
            in_features=out.shape[1], out_features=input_shape[1], bias=True
        )

        attention_mask = torch.sigmoid(
            self.layer_dict["attention_layer"].forward(out)
        ).unsqueeze(
            dim=2
        )  # b, s, 1

        out = torch.sum(
            dummy_x * attention_mask, dim=1
        )

        self.is_built = True

        print("Built", self.__class__.__name__, "output shape", out.shape)

    def forward(self, x):

        if not self.is_built:
            self.build(x.shape)

        out = x.view(x.shape[0], -1)

        for i in range(self.num_hidden_layers):
            out = F.leaky_relu(self.layer_dict["fcc_{}".format(i)].forward(out))
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        attention_mask = torch.sigmoid(
            self.layer_dict["attention_layer"].forward(out)
        ).unsqueeze(
            dim=2
        )

        out = torch.sum(
            x * attention_mask, dim=1
        )

        return out, attention_mask
