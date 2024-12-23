import logging
import sys

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

from src.model.layers import Classifier, InputStream, MainStream


class Student(nn.Module):
    def __init__(self, actions_arch: list, arch_choices, actions_hyper: list, hyper_choices, student_id: int, args,
                 **kwargs):
        """
        Student initializer
        :param actions_arch: list of architecture action indices [1,2,3,2,1....] or dict with chosen config
        :param arch_choices: OrderedDict of architecture choices from config
        :param actions_hyper: list of hyperparameter action indices [1,2,3,2,1....] or dict with chosen config
        :param hyper_choices: OrderedDict of architecture choices from config
        :param student_id: number of student
        :param args: args from config
        :param kwargs: additional args
        """
        super(Student, self).__init__()
        self.student_id = student_id
        self.args = args
        self.kwargs = kwargs
        self.num_input, self.num_channel, _, _, _ = kwargs.get('data_shape')

        # skeleton
        self.input_stream = None
        self.main_stream = None
        self.classifier = None

        # set architecture actions if only indices provided
        if all(isinstance(item, int) for item in actions_arch):
            self.action_arch_dict = dict.fromkeys(arch_choices)
            computations_arch = list(arch_choices.keys())
            for comp, action in zip(computations_arch, actions_arch):
                self.action_arch_dict[comp] = arch_choices[comp][action]
        else:
            self.action_arch_dict = actions_arch

        # set hyperparameter actions here to have them bound to the student
        if all(isinstance(item, int) for item in actions_hyper):
            self.actions_hyper_dict = dict.fromkeys(hyper_choices)
            computations_hyper = list(hyper_choices.keys())
            for comp, action in zip(computations_hyper, actions_hyper):
                self.actions_hyper_dict[comp] = hyper_choices[comp][action]
        else:
            self.actions_hyper_dict = actions_hyper

        # GradCAM bug to get features after activation
        self.feat_main = None
        self.feat_j = None
        self.feat_v = None
        self.feat_b = None
        self.feat_a = None

        # make sure all actions are set
        assert None not in self.action_arch_dict
        assert None not in self.actions_hyper_dict

        # set property
        self.hyper_info = self.actions_hyper_dict
        self.arch_info = self.action_arch_dict

        self.__build_student()

    @property
    def arch_info(self) -> dict:
        return self._arch_info

    @arch_info.setter
    def arch_info(self, info: dict):
        self._arch_info = info

    @property
    def hyper_info(self) -> dict:
        return self._hyper_info

    @hyper_info.setter
    def hyper_info(self, info: dict):
        self._hyper_info = info

    def __build_student(self):
        # build input stream
        # depending on the input size -> JVB == 3 --> JVBA == 4
        # bug in torch so this has to be initialized before!
        input_stream_helper = InputStream(self.action_arch_dict, self.num_channel, self.args, **self.kwargs)

        self.input_stream = nn.ModuleList(
            [InputStream(self.action_arch_dict, self.num_channel, self.args, **self.kwargs)
             for _ in range(self.num_input)])

        # build mainstream
        input_main = input_stream_helper.last_channel * self.num_input
        self.main_stream = MainStream(self.action_arch_dict, input_main, self.args, **self.kwargs)

        # build classifier
        input_classifier = self.main_stream.last_channel
        drop_prob = self.action_arch_dict.get("drop_prob")
        self.classifier = Classifier(input_classifier, drop_prob, self.args.old_sp, **self.kwargs)

        # init parameters
        self.__init_param(self.modules())

    def forward(self, x):
        """
        Forward pass through the input, main stream and classifier.
        N = number of samples
        I = input modalities (JVBA) == number of channels
        C = number of modalities -> num(JVBA * 2)
        T = time step
        V = vertices in graph
        M = number of skeletons
        :param x:
        :return:
        """
        N, I, C, T, V, M = x.size()
        # put number of channels first
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N * M, C, T, V)

        # Separate the input stream by modality (assuming JVBA order)!
        x_j = x[0]  # Joint data
        x_v = x[1]  # Velocity data
        x_b = x[2]  # Bone data
        x_a = x[3]  # Acceleration data

        # Apply each branch separately
        self.feat_j = self.input_stream[0](x_j)
        self.feat_v = self.input_stream[1](x_v)
        self.feat_b = self.input_stream[2](x_b)
        self.feat_a = self.input_stream[3](x_a)

        # Fuse the output from the separate branches
        x = torch.cat((self.feat_j, self.feat_v, self.feat_b, self.feat_a), dim=1)

        # input branches
        # iterate over the number of branches (JVBA) and fuse the output tensor
        # x = [branch(x[i]) for i, branch in enumerate(self.input_stream)]
        # x = torch.cat(x, dim=1)

        # Main stream
        x = self.main_stream(x)

        # output
        # TODO: check pooling option
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        if not self.args.old_sp:
            out = self.classifier(feature).view(N, -1)
        else:
            c_new = x.size(1)
            x = x.view(N, M, c_new, -1)
            x = x.mean(3).mean(1)
            out = self.classifier(x)
        return out

    def forward_before_global_avg_pool(self, x):
        outputs = []

        def hook_fn(module, input_t, output_t):
            outputs.append(output_t)

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)
        self.forward(x)

        assert len(outputs) == 1, f"Expected 1 AdaptiveAvgPool2d, got {len(outputs)}"
        return outputs[0]

    @staticmethod
    def __init_param(modules):
        for m in modules:
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
