
# Standard library
import math
import os
import sys

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


def openFile(filename):
  try:
    f = open(filename)
  except IOError:
    text = 'Couldn\'t open \"%s\" for reading\n' % filename
    sys.stderr.write(text)
    return None
  except:
    sys.stderr.write('unknown error\n')
    return None

  return f

def main():

  if len(sys.argv) != 2:
    print('usage: python plot.py <file>')
    exit(1)
        
  file = str(sys.argv[1])
  create_plot(file)

def create_plot(filename):
    """Load the data from the file ``filename``, and generate the
    corresponding plot.

    """

    f = openFile(filename)
    if not f:
      return

    lines = f.readlines()
    f.close()

    avg_steps = []
    for line in lines:
        cols = line.split(' ')
        cols = cols[0:-1]
        [val0, val1, val2] = [float(x.strip()) for x in cols]
        avg_steps.append([val0, val1, val2])

    alphas = [0.5]
    agent_info_options = [{"num_tiles": 16, "num_tilings": 2, "alpha": 0.5},
                          {"num_tiles": 4, "num_tilings": 32, "alpha": 0.5},
                          {"num_tiles": 8, "num_tilings": 8, "alpha": 0.5}]
    agent_info_options = [{"num_tiles": agent["num_tiles"],
                           "num_tilings": agent["num_tilings"],
                           "alpha": alpha} for agent in agent_info_options for alpha in alphas]


    plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(np.array(avg_steps))
    plt.xlabel("Episode")
    plt.ylabel("Steps Per Episode")
    plt.yscale("linear")
    plt.ylim(0, 1000)
    plt.legend(["num_tiles: {}, num_tilings: {}, alpha: {}".format(agent_info["num_tiles"],
                                                               agent_info["num_tilings"],
                                                               agent_info["alpha"])
            for agent_info in agent_info_options])
    plt.grid(True)
    plt.savefig('fig1.pdf')
    plt.show()

if __name__ == "__main__":
    main()

