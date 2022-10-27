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

sys.path.append(dirname(__file__))


def attack_inf(architecture, final_model, num_steps, delta, order, save_freq, dataset, save_dir,
               batch_size=128,):
    """
    this function stores the adversary inputs and checkpoints to the save_dir
    Note: the final checkpoint is the stolen model (final_model).
    :param architecture: model architecture. (e.g., resnet20)
    :param final_model: the path to the final model
    :param num_steps: number of training steps to fake the PoL for the final model
    :param delta: delta such that d(w_t, w_{t-k}) <= delta
    :param order: dist order (1,2,inf)
    :param save_freq: frequency to fake the checkpoints
    :param dataset: name of the dataset
    :param save_dir: dir to save the noised inputs and checkpoints
    :param batch_size:
    :return:
    """
    trainset = utils.load_dataset(dataset, True, download=True)

    train_size = trainset.__len__()
    sequence = np.concatenate([np.random.default_rng().choice(train_size, size=train_size, replace=False)
                               for _ in range(round(num_steps * batch_size / train_size) + 1)])[:num_steps * batch_size]

    print("Start")
    w = attack.init_w_0(final_model, architecture)
    torch.save({'net': w.state_dict()}, f"{save_dir}/model_0.pt")
    sample_checkpoints = attack.sample_w_t(w, final_model, delta, num_steps, save_freq, architecture,
                                           order=order)
    t = 0

    while True:
        # save the checkpoint
        start = t * save_freq
        end = (t + 1) * save_freq
        if end > num_steps:
            end = num_steps
        print(f"Making points for step {start} to step {end}")
        subset = torch.utils.data.Subset(trainset, sequence[start * batch_size:end * batch_size])
        trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=0, pin_memory=True)
        list_adv_inputs = []
        list_labels = []

        # iterate from the checkpoint to the next
        for i, data in enumerate(trainloader, 0):
            list_adv_inputs.append(data[0].cpu().numpy())
            list_labels.append(data[1].cpu().numpy())

        # save the adv inputs to verify this checkpoint
        np.save(f"{save_dir}/data_{t * save_freq}.npy", np.stack(list_adv_inputs))
        np.save(f"{save_dir}/label_{t * save_freq}.npy", np.stack(list_labels))

        if end < num_steps:
            w = sample_checkpoints.next()
            torch.save({'net': w.state_dict()},
                       f"{save_dir}/model_{(t + 1) * save_freq}.pt")
        else:
            shutil.copyfile(final_model, f"{save_dir}/model_{num_steps}.pt")
            break
        t += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--dist', type=str, default='2',
                        help='metric for computing distance, cos, 1, 2, or inf')
    parser.add_argument('--save-freq', type=int, default=10, help='frequence of saving checkpoints')
    parser.add_argument('--save-dir', type=str, default="inf_update_CIFAR10/")
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--model', type=str, help="Simple_Conv or any torchvision model", default="resnet20")
    parser.add_argument('--final-model', type=str, default="path/to/model")
    parser.add_argument('--delta', type=float, default=0.008, help='threshold for pol, 0 to 1')
    arg = parser.parse_args()

    architecture = eval(f"model.{arg.model}")

    norm_denominator = [71.472] if arg.model == 'resnet20' else [58.056]
    arg.delta *= norm_denominator[0]

    try:
        arg.dist = eval(arg.dist)
    except:
        arg.dist = np.inf

    if not os.path.exists(arg.save_dir):
        os.mkdir(arg.save_dir)
    attack_inf(architecture, arg.final_model, arg.steps, arg.delta, arg.dist, arg.save_freq, arg.dataset,
               arg.save_dir, batch_size=arg.batch_size,)
