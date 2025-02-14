import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from src.postprocessing.options import configure_matplotib
configure_matplotib()


def create_1d_line_animation(t_vec, x_vec, y_matrix, interval=100, **kwargs):
   
    n_times = len(t_vec)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs["xlabel"])
    else:
        ax.set_xlabel('$x$')
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs["ylabel"])
    else:
        ax.set_ylabel('$f(x)$')
    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    else:
        ax.set_title("Line Animation")
    
    # Find global min and max for consistent y-axis limits
    y_min = float('inf')
    y_max = float('-inf')

    y_min = np.min(y_matrix)
    y_max = np.max(y_matrix)
    
    # Add some padding to the limits
    padding = 0.1 * (y_max - y_min)
    ax.set_ylim([y_min - padding, y_max + padding])
    ax.set_xlim([x_vec[0], x_vec[-1]])
    
    # Initialize the line
    line, = ax.plot([], [], 'b-', lw=2)
    
    # Animation initialization function
    def init():
        line.set_data([], [])
        return line,
    
    # Animation update function
    def update(frame):
        line.set_data(x_vec, y_matrix[frame])
        line.set_label(f'Time {t_vec[frame]:.1f} [ms]')
        leg = ax.legend()
        return line, leg
        

    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=n_times, interval=interval,
                                 blit=True)
    
    # Save if filename is provided
    if "filename" in kwargs:
        filename = kwargs["filename"]
        anim.save(filename, writer='ffmpeg')
        print(f"Animation saved as {filename}")

    plt.close()
    
    return 
