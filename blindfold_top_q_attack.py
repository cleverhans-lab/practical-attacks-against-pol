import torch
import torchvision
import os
import sys
import argparse
import numpy as np
import attack
import utils
import model
import shutil
from os.path import dirname
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import pandas as pd
import copy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--dist', type=str, default='2',
                        help='metric for computing distance, cos, 1, 2, or inf')
    parser.add_argument('--save-freq', type=int, default=10, help='frequence of saving checkpoints')
    parser.add_argument('--q', type=int, default=5)
    parser.add_argument('--save-dir', type=str, default='blindfold_top_q')
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--model', type=str, help="Simple_Conv or any torchvision model", default="resnet20")
    parser.add_argument('--final-model', type=str, default="path/to/model")
    parser.add_argument('--delta', type=float, default=0.008, help='threshold for pol, 0 to 1')
    arg = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    architecture = eval(f"model.{arg.model}")

    norm_denominator = [71.472] if arg.model == 'resnet20' else [58.056]
    arg.delta *= norm_denominator[0]

    try:
        arg.dist = eval(arg.dist)
    except:
        arg.dist = np.inf

    save_dir = arg.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    t = 0

    # the stolen final model states
    w_final = utils.consistent_type(arg.final_model, architecture=architecture)
    w = architecture().to(device)
    d0 = utils.compute_distance(utils.consistent_type(w), w_final, arg.dist)
    torch.save({'net': w.state_dict()}, f"{save_dir}/model_{t}.pt")
    sample_checkpoints_epoch = attack.sample_w_t(w, arg.final_model, np.inf, arg.epochs, 1, architecture,
                                                 order=arg.dist)

    optimizer = optim.SGD(w.parameters(), lr=arg.lr)
    try:
        trainset = utils.load_dataset(arg.dataset, True)
    except:
        trainset = utils.load_dataset(arg.dataset, True, download=True)
    trainloader = DataLoader(trainset, batch_size=arg.batch_size, shuffle=True)

    w_next_param = utils.consistent_type(w)
    for e in range(arg.epochs):
        print(f"Epoch {e}")
        if arg.dataset == "CIFAR100":
            torch.cuda.empty_cache()
        w_t_param = w_next_param.clone()
        if e > 0:
            w.load_state_dict(w_next.state_dict())
        w_next = sample_checkpoints_epoch.next()
        w_next_param = utils.consistent_type(w_next)
        t = e * (arg.steps + arg.q * arg.save_freq)
        for i, data in enumerate(trainloader):
            if i < arg.q * arg.save_freq:
                t += 1
                w.train()
                w_cur_temp = copy.deepcopy(utils.consistent_type(w, squeeze=False))
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = w(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # above is a legitimate update
                w_next_temp = utils.consistent_type(w, squeeze=False)
                update_size = utils.compute_distance(torch.concat([i.reshape(-1) for i in w_cur_temp]),
                                                     torch.concat([i.reshape(-1) for i in w_next_temp]), arg.dist)
                scaling_factor = d0 / arg.epochs / update_size
                # this can be thought as the learning rate
                update = utils.add_states(w_next_temp, w_cur_temp, 1, -1)
                w.load_state_dict(utils.to_state_dict(w, utils.add_states(w_cur_temp, update, 1, scaling_factor)))

                if t % arg.save_freq == 0:
                    print(f"saving honestly computed step {t - arg.save_freq} to step {t}")
                    torch.save({'net': w.state_dict()}, f"{save_dir}/model_{t}.pt")
            else:
                # the epoch is not finished, but we don't need real data after this point
                break

        sample_checkpoints_step = attack.sample_w_t(w, w_next, arg.delta, arg.steps, arg.save_freq,
                                                     architecture, order=arg.dist)
        for j in range(arg.steps // arg.save_freq):
            t += arg.save_freq
            print(f"saving spoofed step {t - arg.save_freq} to step {t}")
            torch.save({'net': sample_checkpoints_step.next().state_dict()}, f"{save_dir}/model_{t}.pt")
