import sys
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
plt.ion()


## Example line of log.txt files:
# {"train_lr": 0.0003969463130731183, "train_loss": 0.03189541945527236, "epoch": 30, "test_loss": 0.03149713755660092}


def plot(args):
    # Read in file and convert lines to dicts
    with open(args.log_filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    dicts = [json.loads(l) for l in lines]


    # Plot the data
    keys = dicts[0].keys()
    n_keys = len(keys)
    plt.figure(1, figsize=(8, 1.5*n_keys))
    for i, k in enumerate(keys):
        values = [d[k] for d in dicts]
        plt.subplot(n_keys, 1, i+1)
        plt.plot(values)
        plt.ylabel(k)
        plt.grid(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot info from eval_linear_scale.py log file.')
    parser.add_argument('--log_filename', default='log.txt', type=str, help='Architecture')

    args = parser.parse_args()
    plot(args)
