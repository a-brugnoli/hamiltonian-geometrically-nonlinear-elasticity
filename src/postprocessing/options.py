import matplotlib.pyplot as plt
from matplotlib import rcParams

styles = {
    'markers': ['o', '+', '^', 's'],  # circle, square, triangle, diamond
    'linestyles': ['-', '--', ':', '-.'],  # solid, dashed, dotted, dash-dot
    'colors': ['blue', 'red', 'green', 'purple']
}

def configure_matplotib():
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    rcParams.update({'figure.autolayout': True})
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble']=r"\usepackage{amsmath}\usepackage{bm}"
    rcParams["legend.loc"] = 'best'
    rcParams["figure.figsize"] = (8, 6)


    rcParams['figure.constrained_layout.use'] = True

    return