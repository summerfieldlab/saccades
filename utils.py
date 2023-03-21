from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

class Timer():
    """A class for timing code execution.
    Copied from HRS.
    """
    def __init__(self):
        self.start = datetime.now()
        self.end = None
        self.elapsed_time = None

    def stop_timer(self):
        self.end = datetime.now()
        self.elapsed_time = self.end - self.start
        print('Execution time: {}'.format(self.elapsed_time))
