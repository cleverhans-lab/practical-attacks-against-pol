import torch
import os
import numpy as np
import pandas as pd
import shutil
from scipy import stats
import collections
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

import model


def get_parameters(net, numpy=False, squeeze=True, trainable_only=True):
    trainable = []
    non_trainable = []
    trainable_name = [name for (name, param) in net.named_parameters()]
    state = net.state_dict()
    for i, name in enumerate(state.keys()):
        if name in trainable_name:
            trainable.append(state[name])
        else:
            non_trainable.append(state[name])

    if squeeze:
        trainable = torch.cat([i.reshape([-1]) for i in trainable])
        non_trainable = torch.cat([i.reshape([-1]) for i in non_trainable])
        if numpy:
            trainable = trainable.cpu().numpy()
            non_trainable = non_trainable.cpu().numpy()

    if trainable_only:
        parameter = trainable
    else:
        parameter = trainable + non_trainable

    return parameter


def set_parameters(net, parameters, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                   verbose=False):
    net.load_state_dict(to_state_dict(net, parameters, device, verbose))
    return net


def to_state_dict(net, parameters, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                  verbose=False):
    state_dict = OrderedDict()
    trainable_name = [name for (name, param) in net.named_parameters()]
    if len(trainable_name) < len(parameters):
        if verbose:
            print("Setting trainable and non-trainable parameters")
        i, j = 0, 0
        for name in net.state_dict().keys():
            if name in trainable_name:
                if isinstance(parameters[i], torch.Tensor):
                    state_dict[name] = parameters[i].to(device)
                else:
                    state_dict[name] = torch.Tensor(parameters[i]).to(device)
                i += 1
            else:
                if isinstance(parameters[len(trainable_name) + j], torch.Tensor):
                    state_dict[name] = parameters[len(trainable_name) + j].to(device)
                else:
                    state_dict[name] = torch.Tensor(parameters[len(trainable_name) + j]).to(device)
                j += 1
    else:
        if verbose:
            print("Setting trainable parameters only")
        i = 0
        for name in net.state_dict().keys():
            if name in trainable_name:
                if isinstance(parameters[i], torch.Tensor):
                    state_dict[name] = parameters[i].to(device)
                else:
                    state_dict[name] = torch.Tensor(parameters[i]).to(device)
                i += 1
            else:
                state_dict[name] = net.state_dict()[name]
    return state_dict


def consistent_type(model, architecture=None,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), half=False,
                    squeeze=True, torch_model=False, trainable_only=True):
    # this function takes in directory to where model is saved, model weights as a list of numpy array,
    # or a torch model and outputs model weights as a list of numpy array
    if torch_model:
        net = architecture().to(device)
    if isinstance(model, str):
        assert architecture is not None
        state = torch.load(model, map_location=device)
        net = architecture()
        net.load_state_dict(state['net'])
        if torch_model:
            return net.to(device)
        weights = get_parameters(net, squeeze=squeeze, trainable_only=trainable_only)
    elif isinstance(model, list):
        if torch_model:
            return set_parameters(net, model)
        if trainable_only:
            if architecture is None:
                raise NotImplementedError("My Brain is Not Implemented")
            if not torch_model:
                net = architecture()
            trainable_name = [name for (name, param) in net.named_parameters()]
            model = model[:len(trainable_name)]
        if squeeze:
            weights = torch.cat([i.data.reshape([-1]) for i in model])
        else:
            weights = model
    elif isinstance(model, np.ndarray):
        if torch_model:
            raise NotImplementedError
        weights = torch.tensor(model)
    elif isinstance(model, collections.OrderedDict):
        if not torch_model:
            net = architecture()
        net.load_state_dict(model)
        if torch_model:
            return net.to(device)
        else:
            weights = get_parameters(net, squeeze=squeeze, trainable_only=trainable_only)
    elif not isinstance(model, torch.Tensor):
        if torch_model:
            return model
        weights = get_parameters(model, squeeze=squeeze, trainable_only=trainable_only)
    else:
        if torch_model:
            return set_parameters(net, model)
        weights = model
    if half:
        if half == 2:
            weights = weights.type(torch.IntTensor).type(torch.FloatTensor)
        else:
            weights = weights.half()
    if not isinstance(weights, list):
        weights = weights.to(device)
    else:
        weights = [w.to(device) for w in weights]
    return weights


def compute_distance(a, b, order, numpy=True):
    if order == 'inf':
        order = np.inf
    if order == 'cos' or order == 'cosine':
        dist = (1 - torch.dot(a, b) / (torch.norm(a, p=2) * torch.norm(b, p=2)))
        if numpy:
            dist = dist.cpu().numpy()
        return dist
    else:
        if order != np.inf:
            try:
                order = int(order)
            except:
                raise TypeError("input metric for distance is not understandable")
        dist = torch.norm(a - b, p=order)
        if numpy:
            dist = dist.cpu().numpy()
        return dist


def parameter_distance(model1, model2, order=2, architecture=None, half=False, trainable_only=True):
    # compute the difference between 2 checkpoints
    weights1 = consistent_type(model1, architecture=architecture, half=half, trainable_only=trainable_only)
    weights2 = consistent_type(model2, architecture=architecture, half=half, trainable_only=trainable_only)
    if not isinstance(order, list):
        orders = [order]
    else:
        orders = order
    res_list = []
    for o in orders:
        res = compute_distance(weights1, weights2, o)
        if isinstance(res, np.ndarray):
            res = float(res)
        res_list.append(res)
    return res_list


def load_dataset(dataset, train, download=False, numpy_data=None):
    try:
        dataset_class = eval(f"torchvision.datasets.{dataset}")
    except:
        raise NotImplementedError(f"Dataset {dataset} is not implemented by pytorch.")

    if dataset == "CIFAR100":
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
    if dataset == 'CIFAR10':
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = dataset_class(root='./data', train=train, download=download, transform=transform)

    return data


def unsqueeze(architecture, parameter):
    unsqueezed = []
    net = architecture()
    reference = get_parameters(net, squeeze=False)
    for layer in reference:
        layer_shape = layer.shape
        layer_size = layer.reshape(-1).shape[0]
        unsqueezed.append(parameter[:layer_size].reshape(layer_shape))
        parameter = parameter[layer_size:]
    return unsqueezed


def add_states(state1, state2, a, b):
    return [a * i + b * j for i, j in zip(state1, state2)]
