
# Standard library
import math
import os
import sys

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


def open_file(filename):
    try:
        f = open(filename)
    except IOError:
        text = 'Couldn\'t open \"%s\" for reading\n' % filename
        sys.stderr.write(text)
        return None
    return f


def main():
    if len(sys.argv) != 3:
        print('usage: python plot.py <file1> <file2>')
        exit(1)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))

    file1 = str(sys.argv[1])
    data_len = create_plot1(file1, ax[0])

    file2 = str(sys.argv[2])
    create_plot2(file2, ax[1])

    plt.suptitle("Average Reward Softmax Actor-Critic ({} Runs)".format(data_len),fontsize=16, fontweight='bold', y=1.03)

    plt.show()


def create_plot1(filename, ax):
    """Load the data from the file ``filename``, and generate the
    corresponding plot.
    """
    plt_xticks = [0, 4999, 9999, 14999, 19999]
    plt_xlabels = [1, 5000, 10000, 15000, 20000]
    plt1_yticks = range(0, -6001, -2000)

    actor_ss, critic_ss, avg_reward_ss = 0, 0, 0

    f = open_file(filename)
    if not f:
        return

    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        cols = line.split(' ')
        val = float(cols[0])
        data.append(val)

    data = np.array(data)
    data = np.reshape(data, (25, 20000))

    data_mean = np.mean(data, axis=0)
    data_std_err = np.std(data, axis=0) / np.sqrt(len(data))

    plt_x_legend = range(len(data_mean))

    ax.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    ax.plot(plt_x_legend, data_mean, linewidth=1.0,
            label="actor_ss: {}/32, critic_ss: {}/32, avg reward step_size{}".
            format(actor_ss, critic_ss, avg_reward_ss))

    # ax.legend()
    ax.set_xticks(plt_xticks)
    ax.set_yticks(plt1_yticks)
    ax.set_xticklabels(plt_xlabels)
    ax.set_yticklabels(plt1_yticks)
                        
    ax.set_title("Return per Step")
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Total Return', rotation=90)
    ax.set_xlim([0, 20000])
    ax.grid(True)

    return len(data_mean)


def create_plot2(filename, ax):
    """Load the data from the file ``filename``, and generate the
    corresponding plot.
    """
    x_range = 20000
    plt_xticks = [0, 4999, 9999, 14999, 19999]
    plt_xlabels = [1, 5000, 10000, 15000, 20000]
    plt2_yticks = range(-3, 1, 1)

    actor_ss, critic_ss, avg_reward_ss = 0, 0, 0

    f = open_file(filename)
    if not f:
        return

    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        cols = line.split(' ')
        val = float(cols[0])
        data.append(val)

    data = np.array(data)
    data = np.reshape(data, (25, 20000))

    data_mean = np.mean(data, axis=0)
    data_std_err = np.std(data, axis=0) / np.sqrt(len(data))

    plt_x_legend = range(1, len(data_mean) + 1)

    ax.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    ax.plot(plt_x_legend, data_mean, linewidth=1.0,
            label="actor: {}/32, critic: {}/32, avg reward: {}".format(actor_ss, critic_ss, avg_reward_ss))

    # ax.legend()
    ax.set_xticks(plt_xticks)
    ax.set_yticks(plt2_yticks)
    ax.set_xticklabels(plt_xlabels)
    ax.set_yticklabels(plt2_yticks)

    ax.set_title("Exponential Average Reward per Step")
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Exponential Average Reward', rotation=90)

    ax.set_xlim([0, 20000])
    ax.set_ylim([-3, 0.16])
    ax.grid(True)
    

if __name__ == "__main__":
    main()
