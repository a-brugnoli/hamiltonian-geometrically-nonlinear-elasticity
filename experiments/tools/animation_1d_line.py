import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_1d_line_animation(y_matrix, x_vec, filename=None, 
                            interval=100, title="Line Animation",
                            display=True):
   
    n_times = y_matrix.shape[0]
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title(title)
    
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
        line.set_label(f'Frame num {frame}')
        leg = ax.legend()
        return line, leg
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=n_times, interval=interval,
                                 blit=True)
    
    # Save if filename is provided
    if filename:
        anim.save(filename, writer='ffmpeg')
        print(f"Animation saved as {filename}")
    
    # Display if requested
    if display:
        plt.show()
    # else:
    #     plt.close()
    
    return anim
