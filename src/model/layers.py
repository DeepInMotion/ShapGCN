import logging
import sys

import torch
from torch import nn


from src.model.attention_layers import AttentionLayer
from src.model.activation_layers import Activations


# adopted from
# https://github.com/yfsong0709/EfficientGCNv1


class Stream(nn.Sequential):

    def __init__(self, actions, args, **kwargs):
        super(Stream, self).__init__()

        self.args = args

        # self.act_poss = {
        #     "relu": nn.ReLU(inplace=False),
        #     "relu6": nn.ReLU6(inplace=False),
        #     "silu": nn.SiLU(inplace=False),
        #     "hardswish": nn.Hardswish(inplace=False),
        #     'sigmoid': nn.Sigmoid(),
        #     'tanh': nn.Tanh(),
        #     'swish': Swish(inplace=False),
        #     'acon': AconC(channel=8),
        #     'meta': MetaAconC(channel=8)
        # }

        if self.args.old_sp:
            # old SP used in AutoGCN publication
            # common
            self.init_lay = actions.get("init_lay")
            self.conv_layer = actions.get("conv_lay")
            self.expand_ratio = 0
            self.att_lay = actions.get("att_lay")
            self.last_channel = self.init_lay
            self.temporal_layer = None

            # input stream
            self.blocks_in = actions.get("blocks_in")
            self.depth_in = actions.get("depth_in")
            self.temp_win_in = actions.get("temp_win_in")
            self.graph_dist_in = actions.get("graph_dist_in")
            self.stride_in = actions.get("stride_in")
            self.reduct_ratio_in = actions.get("reduct_ratio_in")

            # mainstream
            self.blocks_main = actions.get("blocks_main")
            self.depth_main = actions.get("depth_main")
            self.graph_dist_main = actions.get("graph_dist_main")
            self.shrinkage_main = actions.get("shrinkage_main")
            self.residual_main = actions.get("residual_main")
            self.adaptive_main = actions.get("adaptive_main")
        else:
            # common
            self.init_lay = actions.get("init_lay")
            self.act = actions.get("act")
            self.att_lay = actions.get("att_lay")
            self.conv_layer = actions.get("conv_lay")
            # drop_prob given in student
            self.multi = actions.get("multi")
            self.expand_ratio = actions.get("expand_ratio")
            self.reduct_ratio = actions.get("reduct_ratio")
            self.temporal_layer = None
            self.last_channel = self.init_lay

            # input stream
            self.blocks_in = actions.get("blocks_in")
            self.depth_in = actions.get("depth_in")
            self.scale_in = actions.get("scale_in")
            self.stride_in = actions.get("stride_in")
            self.temp_win_in = actions.get("temp_win_in")
            self.graph_dist_in = actions.get("graph_dist_in")

            # mainstream
            self.blocks_main = actions.get("blocks_main")
            self.depth_main = actions.get("depth_main")
            self.scale_main = actions.get("scale_main")
            self.stride_main = actions.get("stride_main")
            self.temp_win_main = actions.get("temp_win_main")
            self.graph_dist_main = actions.get("graph_dist_main")

        try:
            # import temporal layer class
            self.temporal_layer = getattr(sys.modules[__name__], f'Temporal_{self.conv_layer}_Layer')
        except:
            logging.error("The conv layer: {} is not known!".format(self.conv_layer))

        self.kwargs = kwargs
        # self.kwargs["act"] = self.act_poss.get(self.act)
        self.kwargs["act"] = self.act
        self.kwargs["expand_ratio"] = self.expand_ratio
        self.kwargs["reduct_ratio"] = self.reduct_ratio

        # add channel scaler make it working with prev. impl.
        try:
            self.scale_in = actions.get("scale_in")
            self.scale_main = actions.get("scale_main")
        except:
            self.scale_in = 0.5
            self.scale_main = 2
            logging.info("Scaling not given in search space -> setting value to: {}/{}".format(self.scale_in,
                                                                                               self.scale_main))


class InputStream(Stream):

    def __init__(self, actions, num_channel, args, **kwargs):
        super(InputStream, self).__init__(actions, args, **kwargs)

        self.args = args

        # fixed starting layer
        self.add_module('init_bn', nn.BatchNorm2d(num_channel))
        self.add_module('stem_scn', Spatial_Graph_Layer(num_channel, self.init_lay, self.graph_dist_in, **self.kwargs))
        self.add_module('stem_tcn', Temporal_Basic_Layer(self.init_lay, self.temp_win_in, **self.kwargs))

        num_subsets = self.kwargs['A'].shape[0]

        for i in range(self.blocks_in):
            # min to 8? or bigger
            channel = max(int(round(self.last_channel * self.scale_in / 16)) * 16, 32)
            # channel = round(int(self.last_channel / self.reduct_ratio))
            if self.multi:
                self.add_module(f'block-{i}_scn_main',
                                Spatial_Multi_Layer(self.last_channel, channel, num_subsets,
                                                    self.graph_dist_in, **self.kwargs))
            else:
                self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(self.last_channel, channel, self.graph_dist_in,
                                                                      **self.kwargs))
            for j in range(self.depth_in):
                s = self.stride_in if j == 0 else 1
                self.add_module(f'block-{i}_tcn-{j}', self.temporal_layer(channel, self.temp_win_in, stride=s,
                                                                          **self.kwargs))
            self.add_module(f'block-{i}_att', AttentionLayer(channel, self.att_lay, **self.kwargs))
            self.last_channel = channel

    @property
    def last_channel(self):
        return self._last_channel

    @last_channel.setter
    def last_channel(self, val):
        self._last_channel = val


class MainStream(Stream):

    def __init__(self, actions, input_main, args, **kwargs):
        super(MainStream, self).__init__(actions, args, **kwargs)
        self.args = args
        self.last_channel = input_main
        assert self.blocks_main != 0

        # calculate channels
        if self.args.dataset in ['cp19', 'cp29']:
            # channels_in = [self.last_channel, 64, 48, 32]
            # channels_out = [64, 64, 48, 32]
            max_channel = 128
            finish_channel = 48
        else:
            # bigger model for ntu or kinetics data
            # channels_in = [self.last_channel, 128, 128, 256, 256, 256]
            # channels_out = [128, 128, 256, 256, 256, 320]
            max_channel = 128 # 320
            finish_channel = 48 # 320
        num_subsets = self.kwargs['A'].shape[0]

        for idx, i in enumerate(range(0, self.blocks_main)):
            # cur_channel_in = channels_in[idx]
            # cur_channel_out = channels_out[idx]
            if idx == self.blocks_main - 1:
                # scale down at end for CP for NTU leave the same
                channel = finish_channel
            else:
                channel = min(int(round(self.last_channel * self.scale_main / 16)) * 16, max_channel)

            if self.multi:
                self.add_module(f'block-{i}_scn_main', Spatial_Multi_Layer(self.last_channel, channel, num_subsets,
                                                                           self.graph_dist_main, **self.kwargs))
            else:
                self.add_module(f'block-{i}_scn_main', Spatial_Graph_Layer(self.last_channel, channel,
                                                                           self.graph_dist_main, **self.kwargs))
            for j in range(self.depth_main):
                s = self.stride_main if j == 0 else 1
                self.add_module(f'block-{i}_tcn_main', Temporal_Basic_Layer(channel, self.temp_win_main, stride=s,
                                                                          **self.kwargs))
            self.add_module(f'block-{i}_att_main', AttentionLayer(channel, self.att_lay, **self.kwargs))
            self.last_channel = channel

    @property
    def last_channel(self):
        return self._last_channel

    @last_channel.setter
    def last_channel(self, val):
        self._last_channel = val


class BasicLayer(nn.Module):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(BasicLayer, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.residual = nn.Identity() if residual else ZeroLayer()
        self.activation = Activations.get_activation(act, self.__repr__())

    def forward(self, x):
        res = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x + res
        return x


class Spatial_Multi_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, num_subsets, max_graph_distance, bias, A, act, residual=True, **kwargs):
        super(Spatial_Multi_Layer, self).__init__()

        self.conv = nn.ModuleList([
            MultiscaleSpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, act, **kwargs)
            for _ in range(num_subsets)
        ])

        if not residual:
            self.residual = ZeroLayer()
        elif in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.residual = nn.Identity()
        self.bn = nn.BatchNorm2d(out_channel)
        self.A = nn.Parameter(A, requires_grad=False)
        self.activation = Activations.get_activation(act, self.__repr__())

    def forward(self, x):
        y = 0
        res = self.residual(x)

        for i, multiconv in enumerate(self.conv):
            y += multiconv(x, self.A[i])
        y = self.activation(y) + self.bn(y) + res
        return y


class Spatial_Graph_Layer(BasicLayer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )


class Temporal_Basic_Layer(BasicLayer):
    def __init__(self, channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual, bias, **kwargs)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2d(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )


class Temporal_Bottleneck_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()

        inner_channel = int(channel // reduct_ratio)
        padding = (temporal_window_size - 1) // 2

        self.reduct_conv = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0), bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )

        self.activation_1 = Activations.get_activation(act, self.__repr__() + "_1")
        self.activation_2 = Activations.get_activation(act, self.__repr__() + "_2")
        self.activation_3 = Activations.get_activation(act, self.__repr__() + "_3")

    def forward(self, x):
        res = self.residual(x)
        x = self.activation_1(self.reduct_conv(x))
        x = self.activation_2(self.conv(x))
        x = self.activation_3(self.expand_conv(x) + res)
        return x


class Temporal_Sep_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2

        if expand_ratio > 0:
            inner_channel = int(channel * expand_ratio)
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
            self.activation_1 = Activations.get_activation(act, self.__repr__() + "_1")
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.activation_2 = Activations.get_activation(act, self.__repr__() + "_2")

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.activation_1(self.expand_conv(x))
        x = self.activation_2(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Temporal_SG_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = int(channel // reduct_ratio)

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size, 1), 1, (padding, 0), groups=channel, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=channel,
                      bias=bias),
            nn.BatchNorm2d(channel),
        )

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )
        self.activation_1 = Activations.get_activation(act, self.__repr__() + "_1")
        self.activation_2 = Activations.get_activation(act, self.__repr__() + "_2")

    def forward(self, x):
        res = self.residual(x)
        x = self.activation_1(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.activation_2(self.point_conv2(x))
        x = self.depth_conv2(x)
        x = x + res
        return x


class Temporal_V3_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True,
                 squeez_excite=True, reduct=4, **kwargs):
        super(Temporal_V3_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        # self.act = act
        # self.act = nn.Hardswish(inplace=False)

        if expand_ratio > 0:
            inner_channel = int(channel * expand_ratio)
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.activation_1 = Activations.get_activation(act, self.__repr__() + "_1")

        if squeez_excite:
            self.squeez_excite = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inner_channel, inner_channel // reduct, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inner_channel // reduct),
                nn.ReLU(inplace=False),
                nn.Conv2d(inner_channel // reduct, inner_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inner_channel),
                nn.ReLU(inplace=False)
            )
        else:
            self.squeez_excite = nn.Identity()

        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )

        self.activation_2 = Activations.get_activation(act, self.__repr__() + "_2")

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.activation_1(self.expand_conv(x))
        x = self.depth_conv(x)
        x = self.squeez_excite(x)
        x = self.point_conv(x)
        x = self.activation_2(x)
        x = x + res
        return x


class Temporal_Shuffle_Layer(nn.Module):
    """
    ShuffleNet with pointwise group
    Point group conv + Channel shuffle + 3x3 depth conv + point group conv and residual
    """

    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, combine=False,
                 **kwargs):
        super(Temporal_Shuffle_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        # in paper // 4
        inner_channel = int(channel // reduct_ratio)
        self.activation = Activations.get_activation(act, self.__repr__())
        self.combine = combine

        self.groups = inner_channel

        # no group conv
        self.point_conv = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # in paper 3x3 kernel and stride == 2
        # try with other depth
        self.depth_conv = nn.Sequential(
            # nn.Conv2d(inner_channel, inner_channel, (3, 3), (stride, 1), (padding, 0),
            #            groups=inner_channel, bias=bias),
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )

        self.point_conv_expand = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel)
        )

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    @staticmethod
    def channel_shuffle(x, groups):
        batchsize, num_channels, time, vertices = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, time, vertices)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, time, vertices)
        return x

    def forward(self, x):
        res = self.residual(x)
        if self.combine:
            res = 0
            # res = torch.nn.functional.avg_pool3d(x, kernel_size=3, stride=2, padding=1)

        x = self.point_conv(x)
        x = self.channel_shuffle(x, self.groups)
        x = self.depth_conv(x)
        x = self.point_conv_expand(x)
        if self.combine:
            x = torch.cat((res, x), -1)
        else:
            x = x + res
        x = self.activation(x)
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Temporal_Multi_Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, dilations=[1, 2], residual=True, **kwargs):
        super(Temporal_Multi_Layer, self).__init__()
        assert in_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = in_channels // self.num_branches
        if type(kernel_size) is list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=False),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation)
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, (stride, 1)),
                nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class SpatialGraphConv(nn.Module):
    """
    https://github.com/yysijie/st-gcn
    """
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channel, out_channel * self.s_kernel_size, 1, bias=bias)
        self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1

        # self.mask = torch.zeros_like(self.A)

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        # A_masked = self.A * self.edge * self.mask
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
        return x


class MultiscaleSpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, act, edge, rel_reduction=2, mid_reduction=1,
                 **kwargs):
        super(MultiscaleSpatialGraphConv, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        if in_channel in (3, 9):
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channel // rel_reduction
            self.mid_channels = in_channel // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=bias)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1, bias=bias)

        self.activation = Activations.get_activation(act, self.__repr__())
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, A):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.activation(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * self.alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class Classifier(nn.Sequential):
    def __init__(self, curr_channel, drop_prob, old_sp, num_class, **kwargs):
        super(Classifier, self).__init__()

        if not old_sp:
            self.add_module('gap', nn.AdaptiveAvgPool3d(1))
            self.add_module('dropout', nn.Dropout(drop_prob, inplace=False))
            self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))
        else:
            self.add_module('dropout', nn.Dropout(drop_prob, inplace=False))
            self.add_module('fc', nn.Linear(curr_channel, num_class))


class ZeroLayer(nn.Module):
    def __init__(self):
        super(ZeroLayer, self).__init__()

    def forward(self, x):
        return 0
