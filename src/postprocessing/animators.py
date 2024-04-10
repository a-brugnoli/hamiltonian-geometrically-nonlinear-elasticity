import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import firedrake as fdrk
import src.postprocessing.options
from mpl_toolkits.mplot3d import Axes3D

def animate_vector_displacement(t_frames, list_frames, interval=10, \
                        lim_x=None, lim_y=None, \
                        xlabel=None, ylabel=None, title=None):

    fig, axes = plt.subplots()

    fdrk.triplot(list_frames[0], axes=axes)

    axes.set_aspect("equal")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    
    def update_plot(frame_number):
        # axes.clear()
        axes.cla()
        # plt.clf()

        label_t = 'Time =' + '{0:.2e}'.format(t_frames[frame_number])
        fdrk.triplot(list_frames[frame_number], axes=axes)

        axes.set_xlim(lim_x)
        axes.set_ylim(lim_y)

    axes.set_title(title, loc='center')

    anim = animation.FuncAnimation(fig, update_plot, frames=len(list_frames), interval = interval)

    return anim


def animate_scalar_displacement(domain, list_frames, interval):

    nsp = 16
    fn_plotter = fdrk.FunctionPlotter(domain, num_sample_points=nsp)

    # Displacement animation
    fig, axes = plt.subplots()
    axes.set_aspect('equal')

    colors = fdrk.tripcolor(list_frames[0], num_sample_points=nsp, axes=axes)
    fig.colorbar(colors)
    def animate(q):
        colors.set_array(fn_plotter(q))

    anim = FuncAnimation(fig, animate, frames=list_frames, interval=interval)

    return anim
