# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Define the PINNs network for the Schrodinger equation."""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import TruncatedNormal, Zero, initializer


class neural_net(nn.Cell):
    """
    Neural net to fit the wave function

    Args:
        layers (list(int)): num of neurons for each layer
        lb (np.array): lower bound (x, t) of domain
        ub (np.array): upper bound (x, t) of domain
    """
    def __init__(self, layers, lb, ub):
        super(neural_net, self).__init__()
        self.layers = layers
        self.lb = Tensor(lb, mstype.float32)
        self.ub = Tensor(ub, mstype.float32)

        self.dense_layers = self._make_layers()

    def _make_layers(self):
        layers = []
        num_layers = len(self.layers) - 1
        for i in range(num_layers):
            in_dim = self.layers[i]
            out_dim = self.layers[i+1]
            std = np.sqrt(2/(in_dim + out_dim))
            weight_init = TruncatedNormal(std)
            if i == num_layers - 1:
                activation = None
            else:
                activation = 'tanh'
            layers.append(nn.Dense(self.layers[i], self.layers[i+1],
                                   weight_init, activation=activation))
        return nn.SequentialCell(layers)

    def construct(self, x, t):
        """Forward propagation"""
        X = ops.concat((x, t), 1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        X = self.dense_layers(X)

        return X[:, 0:1], X[:, 1:2]


class PINNs(nn.Cell):
    """
    PINNs for the Schrodinger equation.
    """
    def __init__(self, layers, lb, ub):
        super(PINNs, self).__init__()
        self.nn = neural_net(layers, lb, ub)

    def construct(self, X):
        """forward propagation"""
        def fn_u(x, t):
            return self.nn(x, t)[0]

        def fn_v(x, t):
            return self.nn(x, t)[1]

        x = X[:, 0:1]
        t = X[:, 1:2]
        u, v = self.nn(x, t)
        du = ops.grad(fn_u, (0, 1))
        ddu = ops.grad(du, (0, 1))
        ux, ut = du(x, t)
        uxx, _ = ddu(x, t)

        dv = ops.grad(fn_v, (0, 1))
        ddv = ops.grad(dv, (0, 1))
        vx, vt = dv(x, t)
        vxx, _ = ddv(x, t)
        
        square_sum = ops.add(ops.pow(u, 2), ops.pow(v, 2))

        fu1 = ops.mul(vxx, 0.5)
        fu2 = ops.mul(square_sum, v)
        fu = ops.add(ops.add(ut, fu1), fu2)

        fv1 = ops.mul(uxx, -0.5)
        fv2 = ops.mul(square_sum, u)
        fv2 = ops.mul(fv2, -1.0)
        fv = ops.add(ops.add(vt, fv1), fv2)

        return u, v, ux, vx, fu, fv
