import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalization_values(batch, dataset, normalized, manifold=False, manifold_factor=1):

    if manifold:
        mean = torch.mean(batch, dim=(0, 2, 3), keepdim=True).to(device)
        std = torch.std(batch, dim=(0, 2, 3), keepdim=True).to(device)
        mean = mean.view(1, batch.size(1), 1, 1)
        std = ((1 / std) / manifold_factor).view(1, batch.size(1), 1, 1)
    elif normalized:
        if dataset == "CIFAR10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1).to(device)
        elif dataset == "CIFAR100":
            mean = (
                torch.tensor([0.50707516, 0.48654887, 0.44091784])
                .view(1, 3, 1, 1)
                .to(device)
            )
            std = (
                torch.tensor([0.26733429, 0.25643846, 0.27615047])
                .view(1, 3, 1, 1)
                .to(device)
            )
        elif dataset == "ImageNet" or dataset == "TinyImageNet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        else:
            print("no normalization values set for this dataset")
    else:
        mean = 0
        std = 1

    return mean, std


class CtModel(nn.Module):

    def __init__(self, dataset, normalized, num_classes):
        super(CtModel, self).__init__()
        self.normalized = normalized
        self.num_classes = num_classes
        self.dataset = dataset
        if normalized:
            mean, std = normalization_values(
                batch=None,
                dataset=dataset,
                normalized=normalized,
                manifold=False,
                manifold_factor=1,
            )
            self.register_buffer("mu", mean)
            self.register_buffer("sigma", std)

    def forward_normalize(self, x):
        if self.normalized:
            x = (x - self.mu) / self.sigma
        return x

    def forward_noise_mixup(self, out, targets):

        out = self.blocks[0](out)

        for i, ResidualBlock in enumerate(self.blocks[1:]):
            out = ResidualBlock(out)
        return out, targets


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(
        self, in_planes, planes, dropout_rate, stride=1, activation_function=F.relu
    ):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.activation_function = activation_function

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.activation_function(self.bn1(x))))
        out = self.conv2(self.activation_function(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation_function=F.relu):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.activation_function = activation_function

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.activation_function(self.bn1(self.conv1(x)))
        out = self.activation_function(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation_function(out)
        return out


class WideResNet(CtModel):
    activation_function: object

    def __init__(
        self,
        depth,
        widen_factor,
        dataset,
        normalized,
        dropout_rate=0.0,
        num_classes=10,
        factor=1,
        block=WideBasic,
        activation_function="relu",
    ):
        super(WideResNet, self).__init__(
            dataset=dataset, normalized=normalized, num_classes=num_classes
        )
        self.in_planes = 16
        self.activation_function = getattr(F, activation_function)

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (int)((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0], stride=1)
        self.layer1 = self._wide_layer(
            block,
            nStages[1],
            n,
            dropout_rate,
            stride=factor,
            activation_function=self.activation_function,
        )
        self.layer2 = self._wide_layer(
            block,
            nStages[2],
            n,
            dropout_rate,
            stride=2,
            activation_function=self.activation_function,
        )
        self.layer3 = self._wide_layer(
            block,
            nStages[3],
            n,
            dropout_rate,
            stride=2,
            activation_function=self.activation_function,
        )
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.blocks = [self.conv1, self.layer1, self.layer2, self.layer3]

    def _wide_layer(
        self, block, planes, num_blocks, dropout_rate, stride, activation_function
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(self.in_planes, planes, dropout_rate, stride, activation_function)
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, targets=None):

        out = super(WideResNet, self).forward_normalize(x)
        out, mixed_targets = super(WideResNet, self).forward_noise_mixup(out, targets)
        out = self.activation_function(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def WideResNet_28_4(
    num_classes,
    dataset,
    normalized,
    factor=1,
    block=WideBasic,
    dropout_rate=0.0,
    activation_function="relu",
):
    return WideResNet(
        depth=28,
        widen_factor=4,
        dataset=dataset,
        normalized=normalized,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        factor=factor,
        block=block,
        activation_function=activation_function,
    )
