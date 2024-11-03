# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

import ast
from functools import reduce
from operator import mul

import torch
from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Module
from torch.nn import ModuleList

from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver

from quants import CommonActQuant, CommonWeightQuant
from tensors import TensorNorm


class FC(Module):

    DROPOUT = 0.2

    def __init__(
        self,
        num_classes,
        weight_bit_width,
        act_bit_width,
        in_bit_width,
        out_features,
        in_features=(28, 28),
    ):
        super(FC, self).__init__()

        self.num_classes = num_classes

        self.features = ModuleList()
        self.features.append(
            QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width)
        )
        self.features.append(Dropout(p=self.DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in out_features:
            self.features.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_bit_width=weight_bit_width,
                    weight_quant=CommonWeightQuant,
                )
            )
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )
            self.features.append(Dropout(p=self.DROPOUT))
        self.features.append(
            QuantLinear(
                in_features=in_features,
                out_features=num_classes,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant,
            )
        )
        self.features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.features:
            x = mod(x)
        return x

    def load_checkpoint(self, checkpoint):
        package = torch.load(checkpoint, map_location='cpu')
        model_state_dict = package['state_dict'] if package.get('state_dict') else package
        self.load_state_dict(model_state_dict, strict=True)

def fc(cfg=None):
    weight_bit_width = 1  #> cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = 1  #> cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = 1  #> cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = 10  #> cfg.getint('MODEL', 'NUM_CLASSES')
    out_features = [256, 256, 256]  #> ast.literal_eval(cfg.get('MODEL', 'OUT_FEATURES'))
    model = FC(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        out_features=out_features,
        num_classes=num_classes,
    )
    return model
