import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import firedrake as fdrk
import src.postprocessing.options
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def animate_vector_triplot(list_frames, interval=10, \
                        lim_x=None, lim_y=None, \
                        xlabel=None, ylabel=None, title=None):

    fig, axes = plt.subplots()

    fdrk.triplot(list_frames[0], axes=axes)

    axes.set_aspect("equal")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    axes.set_xlim(lim_x)
    axes.set_ylim(lim_y)
    
    def update_plot(frame_number):
        # axes.clear()
        axes.cla()
        # plt.clf()

        fdrk.triplot(list_frames[frame_number], axes=axes)

    axes.set_title(title, loc='center')

    anim = FuncAnimation(fig, update_plot, frames=len(list_frames), interval = interval)

    return anim


def animate_scalar_tripcolor(domain, list_frames, interval):

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


def animate_scalar_trisurf(time_frames, list_frames, interval, lim_z = None):

    # Displacement animation
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.set_aspect('equal')

    def animate(frame_number):
        axes.clear()
        time = time_frames[frame_number]
        time_label = f'Time = {time:.2f} [ms]'
        fdrk.trisurf(list_frames[frame_number], axes=axes, label=time_label, cmap=cm.jet)
        axes.set_zlim(lim_z)
        axes.legend()

    
    fdrk.trisurf(list_frames[0], axes=axes, cmap=cm.jet)
    # fig.colorbar(surf_plot)
    anim = FuncAnimation(fig, animate, frames=len(list_frames), interval=interval)

    return anim
