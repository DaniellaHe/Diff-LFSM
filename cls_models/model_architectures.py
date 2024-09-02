from typing import Type, Any, Callable, Union, List, Optional

from torch import Tensor

from cls_models.attention_modules import *


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 2,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            p=0
    ) -> None:

        self.is_layer_built = False

        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.p = p
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.conv1_s = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_s = norm_layer(self.inplanes)
        self.relu_s = nn.ReLU(inplace=True)
        self.maxpool_s = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_s = self._make_layer(block, 64, layers[0])
        self.layer2_s = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3_s = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4_s = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.att_1 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.att_2 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)
        self.att_3 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.slice_attention = SliceAttention(
            num_hidden_layers=1, num_hidden_units=64, dropout_rate=0.1
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fusion = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, bias=False)

        self.dropout = nn.Dropout(p=self.p)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def build(self, input_shape):
        """
        Builds network whilst automatically inferring shapes of layers.
        """

        original_shape = input_shape
        out = torch.zeros(input_shape)

        c, h, w = out.shape[-3:]
        out = out.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        #
        out1 = self.conv1(out)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        out1 = self.layer1(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)
        out6 = self.fc(out5)

        self.is_layer_built = True

    def _forward_impl(self, x: Tensor) -> Tensor:

        input = x

        if not self.is_layer_built:
            self.build(input_shape=input.shape)
            self.to(x.device)

        original_shape = input.shape

        c, h, w = input.shape[-3:]
        input = input.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)

        out1 = self.conv1(input)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        out1 = self.layer1(out1)
        if self.p > 0:
            out1 = self.dropout(out1)
        out2 = self.layer2(out1)
        if self.p > 0:
            out2 = self.dropout(out2)
        out3 = self.layer3(out2)
        if self.p > 0:
            out3 = self.dropout(out3)
        out4 = self.layer4(out3)
        out4_f = self.layer4(out3)
        out4 = self.avgpool(out4_f)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)

        out6 = self.fc(out5)

        return out6

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResnetACAT(nn.Module):
    def __init__(
            self,
            # num_classes_dict,
            # task_filters,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 2,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        self.is_layer_built = False

        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # inplanes 表示输入通道数或是正在处理的特征图的深度。
        self.dilation = 1  # 卷积中的扩张率。在这里，设置为 1 表示卷积核的膨胀率是默认的，即不扩张。
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.conv1_s = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_s = norm_layer(self.inplanes)
        self.relu_s = nn.ReLU(inplace=True)
        self.maxpool_s = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_s = self._make_layer(block, 64, layers[0])
        self.layer2_s = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3_s = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4_s = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.att_1 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.att_2 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)
        self.att_3 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.slice_attention = SliceAttention(
            num_hidden_layers=1, num_hidden_units=64, dropout_rate=0.1
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fusion = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def build(self, input_shape):
        """
        Builds network whilst automatically inferring shapes of layers.
        """

        original_shape = input_shape
        out = torch.zeros(input_shape)

        c, h, w = out.shape[-3:]
        out = out.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        #
        out1 = self.conv1(out)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        out1 = self.layer1(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)
        out6 = self.fc(out5)

        self.is_layer_built = True

    def _forward_impl(self, x: Tensor, y: Tensor) -> Tensor:

        input = x
        input_sal = y

        if not self.is_layer_built:
            self.build(input_shape=input.shape)
            self.to(x.device)

        original_shape = input.shape

        c, h, w = input.shape[-3:]
        input = input.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)
        input_sal = input_sal.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            input_sal = input_sal.repeat(1, 3, 1, 1)

        # print(input.dtype)  # torch.float32
        out1 = self.conv1(input)
        s1 = self.conv1_s(input_sal)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        s1 = self.bn1_s(s1)
        s1 = self.relu_s(s1)
        s1 = self.maxpool_s(s1)

        out1 = self.layer1(out1)
        s1 = self.layer1_s(s1)
        s1_mask_o = self.att_1(s1)
        out1 = out1 + torch.multiply(out1, s1_mask_o)

        out2 = self.layer2(out1)
        s2 = self.layer2_s(s1)
        s2_mask_o = self.att_2(s2)
        out2 = out2 + torch.multiply(out2, s2_mask_o)

        out3 = self.layer3(out2)
        s3 = self.layer3_s(s2)
        s3_mask_o = self.att_3(s3)
        out3 = out3 + torch.multiply(out3, s3_mask_o)

        s1_mask = self.pool(s1_mask_o)
        s2_mask = self.pool(s2_mask_o)
        s3_mask = self.pool(s3_mask_o)
        s_mask = torch.cat((s1_mask, s2_mask, s3_mask), dim=1)
        fusion_mask = self.fusion(s_mask)
        out4 = self.layer4(out3)

        out4 = out4 + torch.multiply(out4, fusion_mask)

        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5, slice_mask = self.slice_attention.forward(out4)
        out6 = self.fc(out5)

        return out6

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._forward_impl(x, y)


class ResNetImageAttention(nn.Module):
    def __init__(
            self,
            # num_classes_dict,
            # task_filters,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 2,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        self.is_layer_built = False

        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.conv1_s = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_s = norm_layer(self.inplanes)
        self.relu_s = nn.ReLU(inplace=True)
        self.maxpool_s = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_s = self._make_layer(block, 64, layers[0])
        self.layer2_s = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3_s = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4_s = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.att_1 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.att_2 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)
        self.att_3 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.slice_attention = SliceAttention(
            num_hidden_layers=1, num_hidden_units=64, dropout_rate=0.1
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fusion = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def build(self, input_shape):
        """
        Builds network whilst automatically inferring shapes of layers.
        """

        original_shape = input_shape
        out = torch.zeros(input_shape)

        c, h, w = out.shape[-3:]
        out = out.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        #
        out1 = self.conv1(out)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        out1 = self.layer1(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)
        out6 = self.fc(out5)

        self.is_layer_built = True

    def _forward_impl(self, x: Tensor) -> Tensor:
        input = x

        # print('input', input.shape)
        if not self.is_layer_built:
            self.build(input_shape=input.shape)
            self.to(x.device)

        original_shape = input.shape

        c, h, w = input.shape[-3:]
        input = input.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)

        out1 = self.conv1(input)

        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)

        out1 = self.layer1(out1)
        res1 = out1
        s1_mask = self.att_1(out1)
        out1 = res1 + torch.multiply(res1, s1_mask)

        out2 = self.layer2(out1)
        res2 = out2

        s2_mask = self.att_2(out2)
        out2 = res2 + torch.multiply(res2, s2_mask)

        out3 = self.layer3(out2)

        res3 = out3

        s3_mask = self.att_3(out3)
        out3 = res3 + torch.multiply(res3, s3_mask)

        out4 = self.layer4(out3)

        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5, slice_mask = self.slice_attention.forward(out4)
        out6 = self.fc(out5)

        return out6

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetSMIC(nn.Module):
    def __init__(
            self,
            # num_classes_dict,
            # task_filters,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 2,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        self.is_layer_built = False

        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.conv1_s = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_s = norm_layer(self.inplanes)
        self.relu_s = nn.ReLU(inplace=True)
        self.maxpool_s = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_s = self._make_layer(block, 64, layers[0])
        self.layer2_s = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3_s = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4_s = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.att_1 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.att_2 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)
        self.att_3 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.slice_attention = SliceAttention(
            num_hidden_layers=1, num_hidden_units=64, dropout_rate=0.1
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fusion = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def build(self, input_shape):
        """
        Builds network whilst automatically inferring shapes of layers.
        """

        original_shape = input_shape
        out = torch.zeros(input_shape)

        c, h, w = out.shape[-3:]
        out = out.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        #
        out1 = self.conv1(out)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        out1 = self.layer1(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)
        out6 = self.fc(out5)

        self.is_layer_built = True

    def _forward_impl(self, x: Tensor, y: Tensor) -> Tensor:

        input = x
        input_sal = y

        if not self.is_layer_built:
            self.build(input_shape=input.shape)
            self.to(x.device)

        original_shape = input.shape

        c, h, w = input.shape[-3:]
        input = input.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)
        input_sal = input_sal.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            input_sal = input_sal.repeat(1, 3, 1, 1)

        out1 = self.conv1(input)
        s1 = self.conv1_s(input_sal)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        s1 = self.bn1_s(s1)
        s1 = self.relu_s(s1)
        s1 = self.maxpool_s(s1)

        out1 = self.layer1(out1)

        s1 = self.layer1_s(s1)

        out2 = self.layer2(out1)
        res2 = out2
        s2 = self.layer2_s(s1)
        s2 = torch.sigmoid(s2)
        out2 = res2 + torch.multiply(res2, s2)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)

        out6 = self.fc(out5)

        return out6

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._forward_impl(x, y)


class ResNetHSM(nn.Module):
    def __init__(
            self,
            # num_classes_dict,
            # task_filters,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 2,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        self.is_layer_built = False

        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.conv1_s = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_s = norm_layer(self.inplanes)
        self.relu_s = nn.ReLU(inplace=True)
        self.maxpool_s = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_s = self._make_layer(block, 64, layers[0])
        self.layer2_s = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3_s = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4_s = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.att_1 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.att_2 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)
        self.att_3 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.slice_attention = SliceAttention(
            num_hidden_layers=1, num_hidden_units=64, dropout_rate=0.1
        )

        self.sal_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def build(self, input_shape):
        """
        Builds network whilst automatically inferring shapes of layers.
        """

        original_shape = input_shape
        out = torch.zeros(input_shape)

        c, h, w = out.shape[-3:]
        out = out.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        #
        out1 = self.conv1(out)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        out1 = self.layer1(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)
        out6 = self.fc(out5)

        self.is_layer_built = True

    def _forward_impl(self, x: Tensor, y: Tensor) -> Tensor:

        input = x
        input_sal = y
        if not self.is_layer_built:
            self.build(input_shape=input.shape)
            self.to(x.device)

        original_shape = input.shape

        c, h, w = input.shape[-3:]
        input = input.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)
        input_sal = input_sal.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            input_sal = input_sal.repeat(1, 3, 1, 1)

        out1 = self.conv1(input)
        s1 = self.conv1_s(input_sal)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        s1 = self.bn1_s(s1)
        s1 = self.relu_s(s1)
        s1 = self.maxpool_s(s1)

        out1 = self.layer1(out1)
        s1 = self.layer1_s(s1)

        out2 = self.layer2(out1)
        s2 = self.layer2_s(s1)

        out3 = self.layer3(out2)
        s3 = self.layer3_s(s2)
        out4 = self.layer4(out3)
        res4 = out4
        s4 = self.layer4_s(s3)
        s4 = F.relu(s4)
        out4 = res4 + torch.multiply(res4, s4)

        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)

        out6 = self.fc(out5)

        return out6

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._forward_impl(x, y)


class ResNetSalClassNet(nn.Module):
    def __init__(
            self,
            # num_classes_dict,
            # task_filters,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 2,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        self.is_layer_built = False

        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.conv1_s = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_s = norm_layer(self.inplanes)
        self.relu_s = nn.ReLU(inplace=True)
        self.maxpool_s = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_s = self._make_layer(block, 64, layers[0])
        self.layer2_s = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3_s = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4_s = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.att_1 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.att_2 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)
        self.att_3 = Spatial_attention(nf_i=1, nf_o=1, kernel=3, stride=1, padding=1, dilation=1, bias=False,
                                       dropout_rate=0.1)

        self.slice_attention = SliceAttention(
            num_hidden_layers=1, num_hidden_units=64, dropout_rate=0.1
        )

        self.sal_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def build(self, input_shape):
        """
        Builds network whilst automatically inferring shapes of layers.
        """

        original_shape = input_shape
        out = torch.zeros(input_shape)

        c, h, w = out.shape[-3:]
        out = out.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if out.shape[1] == 1:
            out = out.repeat(1, 6, 1, 1)
        #
        out1 = self.conv1(out)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        out1 = self.layer1(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)
        out6 = self.fc(out5)

        self.is_layer_built = True

    def _forward_impl(self, x: Tensor, y: Tensor) -> Tensor:
        input = x
        input_sal = y
        if not self.is_layer_built:
            self.build(input_shape=input.shape)
            self.to(x.device)

        original_shape = input.shape

        c, h, w = input.shape[-3:]
        input = input.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)
        input_sal = input_sal.reshape(-1, c, h, w)  # out (b*half*slices, c, h, w)

        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            input_sal = input_sal.repeat(1, 3, 1, 1)

        new_input = torch.cat((input, input_sal), -3)

        out1 = self.conv1(new_input)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)

        out1 = self.layer1(out1)

        out2 = self.layer2(out1)

        out3 = self.layer3(out2)

        out4 = self.layer4(out3)

        out4 = self.avgpool(out4)

        out4 = out4.reshape(original_shape[0], original_shape[1],  # b * half
                            out4.shape[1])

        out5 = torch.mean(out4, dim=1)

        out6 = self.fc(out5)

        return out6

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._forward_impl(x, y)


def resnet50_baseline(

        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    return model


def resnet50_SMIC(

        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],

        **kwargs: Any,
) -> ResNet:
    model = ResNetSMIC(block, layers, **kwargs)

    return model


def resnet50_HSM(

        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs: Any,
) -> ResNet:
    model = ResNetHSM(block, layers, **kwargs)

    return model


def resnet50_SalClassNet(

        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs: Any,
) -> ResNet:
    model = ResNetSalClassNet(block, layers, **kwargs)

    return model


def resnet50_ACAT(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs: Any,
) -> ResNet:
    model = ResnetACAT(block, layers, **kwargs)

    return model


def resnet50_image_attention(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs: Any,
) -> ResNet:
    model = ResNetImageAttention(block, layers, **kwargs)

    return model
