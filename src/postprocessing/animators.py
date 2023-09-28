import matplotlib.pyplot as plt
import matplotlib.animation as animation
import firedrake as fdrk
import src.postprocessing.options

def animate_displacement(t_frames, list_frames, interval=10, lim_x=None, lim_y=None, xlabel=None, ylabel=None, title=None):

    fig, axes = plt.subplots()

    def update_plot(frame_number):
        # axes.clear()
        axes.collections.clear()

        label_t = 'Time =' + '{0:.2e}'.format(t_frames[frame_number])
        fdrk.triplot(list_frames[frame_number], axes=axes)
        
    fdrk.triplot(list_frames[0], axes=axes)

    axes.set_aspect("equal")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    axes.set_xlim(lim_x)
    axes.set_ylim(lim_y)

    axes.set_title(title, loc='center')

    anim = animation.FuncAnimation(fig, update_plot, frames=len(list_frames), interval = interval)

    return anim

