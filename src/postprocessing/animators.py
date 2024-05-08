import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import firedrake as fdrk
import src.postprocessing.options
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def animate_vector_triplot(list_frames, interval=10, three_dim = False, \
                    lim_x=None, lim_y=None, lim_z=None, \
                    xlabel=None, ylabel=None, zlabel=None, title=None):

    fig = plt.figure()
    if three_dim:
        axes = fig.add_subplot(111, projection='3d')
    else:
        axes = fig.add_subplot(111)

    axes.set_aspect("equal")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title, loc='center')

    axes.set_xlim(lim_x)
    axes.set_ylim(lim_y)
    if three_dim:
        axes.set_zlabel(zlabel)

    fdrk.triplot(list_frames[0], axes=axes)

    
    def update_plot(frame_number):
        axes.clear()
        # axes.cla()
        # plt.clf()
        axes.set_xlim(lim_x)
        axes.set_ylim(lim_y)
        if three_dim:
            axes.set_zlim(lim_z)
        
        fdrk.triplot(list_frames[frame_number], axes=axes)
        
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


def animate_scalar_trisurf(time_frames, list_frames, interval,\
                           title=None, xlabel=None, ylabel=None, lim_z = None):
    vmin, vmax = lim_z
    # Displacement animation
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.set_aspect('equal')
    
    def animate(frame_number):
        axes.clear()
        time = time_frames[frame_number]
        time_label = f'Time = {time:.2f}' + r'$\; \mathrm{[ms]}$'
        fdrk.trisurf(list_frames[frame_number], axes=axes, \
                     label=time_label, cmap=cm.jet, vmin=vmin, vmax = vmax)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title, loc='center')
        axes.legend()
        axes.set_zlim(lim_z)


    
    fdrk.trisurf(list_frames[0], axes=axes, cmap=cm.jet, \
                 vmin=vmin, vmax = vmax)
    # fig.colorbar(surf_plot)
    anim = FuncAnimation(fig, animate, frames=len(list_frames), interval=interval)

    return anim
