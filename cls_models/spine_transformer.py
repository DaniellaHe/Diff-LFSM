import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # resnet = resnet18(pretrained=True)
        resnet = resnet18()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 2)
        resnet.load_state_dict(torch.load(f'/data/hsd/DiffCASM/model/params-ARGS=2/checkpoint/classifier_epoch=60.pt'))
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification layer.

    def forward(self, x):
        x = self.feature_extractor(x)  # [BS, len, 512, 1, 1]
        x = x.view(x.size(0), x.size(1), -1)  # Flatten to [BS, len, 512]
        return x

class CNNFeatureExtractor(nn.Module):
    """ A simple CNN to extract features from images. """
    def __init__(self, input_channels, output_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(output_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        return x.view(x.size(0), x.size(1), -1)  # Flatten the features
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class EvidentialLayer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(EvidentialLayer, self).__init__()
        # Each class gets a set of evidence parameters alpha
        self.evidence = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        evidence = F.softplus(self.evidence(x))  # Ensure the evidence is positive
        alpha = evidence + 1
        uncertainty = self.num_classes / torch.sum(alpha, dim=-1, keepdim=True)  # Calculate uncertainty
        return uncertainty


class SaliencyModulatedAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SaliencyModulatedAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.saliency_to_query = nn.Linear(d_model, d_model)
        self.saliency_to_key = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.head_dim = d_model // nhead

    def forward(self, x, saliency_feats):
        bs, seq_len = x.shape[:2]

        # Process inputs
        query = self.query(x).view(bs, seq_len, self.nhead, self.head_dim)
        key = self.key(x).view(bs, seq_len, self.nhead, self.head_dim)
        value = self.value(x).view(bs, seq_len, self.nhead, self.head_dim)

        # Modulate query and key with saliency features
        saliency_query = self.saliency_to_query(saliency_feats).view(bs, seq_len, self.nhead, self.head_dim)
        saliency_key = self.saliency_to_key(saliency_feats).view(bs, seq_len, self.nhead, self.head_dim)
        query += saliency_query
        key += saliency_key

        # Attention mechanism
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = self.softmax(scores)
        context = torch.matmul(attention, value).transpose(1, 2).contiguous().view(bs, seq_len, -1)
        output = self.output(context)
        return output


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, saliency_src):
        attention = self.conv1(saliency_src)
        attention = self.sigmoid(attention)

        return src * attention

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.classifier = nn.Conv1d(feature_dim, num_classes, kernel_size=1)  # 1x1 Convolution

    def forward(self, src):
        src = self.pos_encoder(src)  # [bs, seq_len, 512]
        output = self.transformer_encoder(src)  # [bs, seq_len, 512]
        output = output.transpose(1, 2)  # [bs, 512, seq_len]
        output = self.classifier(output)  # [bs, 2, seq_len]
        output = output.transpose(1, 2)  # [bs, seq_len, 2]
        return output


class SaliencyModulatedTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(SaliencyModulatedTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_dim)
        # self.attention_mod = SaliencyModulatedAttention(feature_dim, nhead=8)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.classifier = nn.Conv1d(feature_dim, num_classes, kernel_size=1)
        self.feature_extractor = FeatureExtractor().eval()
        self.sal_feature_extractor = CNNFeatureExtractor(input_channels=1, output_dim=feature_dim)
        self.attention_mod = SpatialAttentionModule(in_channels=1)

    def forward(self, src, saliency_src):
        bs, seq_len, c, h, w = src.size()
        src = src.view(-1, c, h, w)
        saliency_src = saliency_src.view(-1, c, h, w)

        src = self.attention_mod(src, saliency_src)

        src = self.feature_extractor(src).view(bs, seq_len, -1)  # [bs, seq_len, 512]
        src = self.pos_encoder(src)  # [bs, seq_len, 512]
        output = self.transformer_encoder(src)  # [bs, seq_len, 512]
        output = output.transpose(1, 2)  # [bs, 512, seq_len]
        output = self.classifier(output)  # [bs, num_classes, seq_len]
        output = output.transpose(1, 2)  # [bs, seq_len, num_classes]
        return output

class EvidentialSaliencyTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(EvidentialSaliencyTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_dim)
        # self.attention_mod = SaliencyModulatedAttention(feature_dim, nhead=8)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.classifier = nn.Conv1d(feature_dim, num_classes, kernel_size=1)
        self.feature_extractor = FeatureExtractor().eval()
        self.sal_feature_extractor = CNNFeatureExtractor(input_channels=1, output_dim=feature_dim)
        self.attention_mod = SpatialAttentionModule(in_channels=1)
        self.evidential_layer = EvidentialLayer(feature_dim, num_classes=2)

    def forward(self, src, saliency_src):
        bs, seq_len, c, h, w = src.size()
        src = src.view(-1, c, h, w)
        saliency_src = saliency_src.view(-1, c, h, w)

        src = self.attention_mod(src, saliency_src)

        src = self.feature_extractor(src).view(bs, seq_len, -1)  # [bs, seq_len, 512]
        src = self.pos_encoder(src)  # [bs, seq_len, 512]

        # Calculate uncertainty for each vertebra
        uncertainty = self.evidential_layer(src)  # [bs, seq_len, 1]
        weighted_src = src * (1 - uncertainty)  # [bs, seq_len, 512]

        output = self.transformer_encoder(weighted_src)  # [bs, seq_len, 512]
        output = output.transpose(1, 2)  # [bs, 512, seq_len]
        output = self.classifier(output)  # [bs, num_classes, seq_len]
        output = output.transpose(1, 2)  # [bs, seq_len, num_classes]
        return output