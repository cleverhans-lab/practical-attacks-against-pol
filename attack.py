import copy
import torch
import numpy as np
import utils


def init_w_0(final_model, architecture):
    """
    initialize the weights follow the distribution and minimizes d(w_0, w_T)
    :param final_model: the model that contains the final weights, can be a path, parameters, or a torch model
    :param architecture: model architecture used as in model.py
    :return: initial model
    """
    initial_model = architecture()
    initial_weight = utils.consistent_type(initial_model, architecture, squeeze=False, trainable_only=False)
    final_weight = utils.consistent_type(final_model, architecture, squeeze=False, trainable_only=False)
    # match the weights layer by layer
    for w0, wt in zip(initial_weight, final_weight):
        indices = wt.flatten().argsort()
        w0.data = w0.data.flatten()[indices].reshape(w0.shape)
    initial_model = utils.set_parameters(initial_model, initial_weight, verbose=True)
    return initial_model


class sample_w_t():
    def __init__(self, initial_model, final_model, delta, num_steps, save_freq, architecture, order,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """
        :param initial_model: initial weights w_0
        :param final_model: w_T
        :param delta: delta such that d(w_t, w_{t-k}) <= delta
        :param num_steps: total number of training steps T
        :param save_freq: checkpoint interval k
        :param order: distance (l1, l2, inf, or cos)
        :param architecture:
        :param half:
        :return: list of checkpoints
        """
        self.device = device
        self.save_freq = save_freq
        self.architecture = architecture
        self.initial = utils.consistent_type(initial_model, architecture, squeeze=False, trainable_only=False)
        self.prev = copy.deepcopy(self.initial)
        self.final = utils.consistent_type(final_model, architecture, squeeze=False, trainable_only=False)
        self.t = 0
        self.num_steps = num_steps
        # self.diff = [(i.data.double() - j.data.double()) / (num_steps) for i, j in zip(final_weight, self.prev)]
        if not isinstance(order, list):
            self.order = [order]
        else:
            self.order = order
        if not isinstance(delta, list):
            self.delta = [delta]
        else:
            self.delta = delta

        if len(self.delta) != len(self.order):
            if len(self.delta) == 1:
                self.delta = self.delta * len(self.order)
            else:
                raise ValueError("give list of delta doesn't match the number of orders")

    def next(self):
        self.t += self.save_freq
        checkpoint = [j.data + self.t * (i.data - j.data) / (self.num_steps) for i, j in zip(self.final, self.initial)]
        dist = utils.parameter_distance(self.prev, checkpoint, order=self.order, architecture=self.architecture)
        for o, d, th in zip(self.order, dist, self.delta):
            if d > th:
                raise ValueError(f"Distance in between checkpoints exceed the {o} threshold, "
                                 f"larger num_steps is needed")
        self.prev = utils.set_parameters(self.architecture().to(self.device), checkpoint)
        return self.prev
