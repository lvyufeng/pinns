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
"""Train PINNs for Schrodinger equation scenario"""
import time
import numpy as np
from mindspore.common import set_seed
from mindspore import nn, ops, save_checkpoint, ms_function
from src.Schrodinger.dataset import generate_PINNs_training_set
from src.Schrodinger.net import PINNs
from src.Schrodinger.loss import Loss


def train_sch(epochs=50000, lr=0.0001, N0=50, Nb=50, Nf=20000, num_neuron=100, seed=None,
              path='./Data/NLS.mat', ck_path='./ckpoints/'):
    """
    Train PINNs network for Schrodinger equation

    Args:
        epoch (int): number of epochs
        lr (float): learning rate
        N0 (int): number of data points sampled from the initial condition,
            0<N0<=256 for the default NLS dataset
        Nb (int): number of data points sampled from the boundary condition,
            0<Nb<=201 for the default NLS dataset. Size of training set = N0+2*Nb
        Nf (int): number of collocation points, collocation points are used
            to calculate regularizer for the network from Schoringer equation.
            0<Nf<=51456 for the default NLS dataset
        num_neuron (int): number of neurons for fully connected layer in the network
        seed (int): random seed
        path (str): path of the dataset for Schrodinger equation
        ck_path (str): path to store checkpoint files (.ckpt)
    """
    if seed is not None:
        np.random.seed(seed)
        set_seed(seed)

    layers = [2, num_neuron, num_neuron, num_neuron, num_neuron, 2]

    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    dataset = generate_PINNs_training_set(N0, Nb, Nf, lb, ub, path=path)
    dataset = dataset.repeat(epochs)

    model = PINNs(layers, lb, ub)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
    loss_fn = Loss(N0, Nb, Nf)

    print_per_iter = 10
    save_ckpt_steps = 1000

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    # Define function of one-step training
    @ms_function
    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        start_time = time.time()
        loss = train_step(data, label)
        cost_time = time.time() - start_time

        if batch % print_per_iter == 0:
            loss, current = loss.asnumpy(), batch + 1
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}], time: {(cost_time * 1000):.3f} ms/step")

        if batch % (save_ckpt_steps - 1) == 0:
            save_checkpoint(model, f'ckpoints/PINNs_Schrodinger_{batch}.ckpt')