import matplotlib.pyplot as plt
import matplotlib.animation as animation
import firedrake as fdrk
import src.postprocessing.options

def animate_displacement(t_frames, list_frames, interval=10, lim_x=None, lim_y=None, xlabel=None, ylabel=None, title=None):
    fntsize = 20

    fig, axes = plt.subplots()
    axes.set_aspect("equal")

    def update_plot(frame_number):
        axes.collections.clear()

        label_t = 'Time =' + '{0:.2e}'.format(t_frames[frame_number])
        fdrk.triplot(list_frames[frame_number], axes=axes)
        

    axes.set_xlabel(xlabel, fontsize=fntsize)
    axes.set_ylabel(ylabel, fontsize=fntsize)

    if lim_x is not None:
        axes.set_xlim(lim_x)
    if lim_y is not None:
        axes.set_ylim(lim_y)

    axes.set_title(title, fontsize=fntsize, loc='center')

    label_t = 'Time =' + '{0:.2e}'.format(t_frames[0])

    fdrk.triplot(list_frames[0], axes=axes)

    anim = animation.FuncAnimation(fig, update_plot, frames=len(list_frames), interval = interval)

    return anim

