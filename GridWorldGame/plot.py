
# Standard library
import math
import os
import sys

# Third-party libraries
import matplotlib.pyplot as plt


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

    avg_returns = {"Expected Sarsa": [], "Q-Learning": []}
    for line in lines:
        cols = line.split(' ')
        cols = cols[0:-1]
        [val0, val1] = [float(x.strip()) for x in cols]
        avg_returns["Expected Sarsa"].append(val0)
        avg_returns["Q-Learning"].append(val1)

    for algorithm in avg_returns.keys():
        plt.plot(avg_returns[algorithm], label=algorithm)

    plt.xlabel("Episodes")
    plt.ylabel("Sum of\n rewards\n during\n episode", rotation=0, labelpad=25)
    plt.xlim(0, 250)
    plt.ylim(-400, 1200)
    plt.legend()
    plt.grid(True)
    plt.savefig('fig1.pdf')
    plt.show()

if __name__ == "__main__":
    main()

