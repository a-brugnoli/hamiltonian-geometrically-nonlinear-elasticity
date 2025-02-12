import firedrake as fdrk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def is_hermite(f):
    # Get the function space from the function
    V = f.function_space()
    # Get the element
    element = V.ufl_element()
    # Check if it's Hermite
    return 'Hermite' in element.family()


def create_1d_function_animation(functions, mesh, filename=None, 
                            interval=100, title="Function Animation",
                            display=True):
    """
    Create and display/save an animation from a list of Firedrake 1D functions.
    
    Parameters:
    -----------
    functions : list of firedrake.Function
        List of Firedrake functions to animate
    mesh : firedrake.Mesh
        The 1D mesh used for the functions
    filename : str, optional
        Output filename (must end in .mp4). If None, animation is only displayed
    interval : int
        Time interval between frames in milliseconds
    title : str
        Title for the animation
    display : bool
        Whether to display the animation in a window
    
    Returns:
    --------
    anim : matplotlib.animation.Animation
        The animation object
    """
    
    # Get coordinates of mesh points
    x = mesh.coordinates.dat.data_ro[:]
    x_sorted_idx = np.argsort(x)
    x_sorted = x[x_sorted_idx]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title(title)
    
    # Find global min and max for consistent y-axis limits
    y_min = float('inf')
    y_max = float('-inf')

    for f in functions:
        if is_hermite(f):
            y = f.dat.data_ro[::2]
        else:
            y = f.dat.data_ro[:]


        y_min = min(y_min, y.min())
        y_max = max(y_max, y.max())
    
    # print(f"Minimum y: {y_min}")
    # print(f"Maximum y: {y_max}")
    
    # Add some padding to the limits
    padding = 0.1 * (y_max - y_min)
    ax.set_ylim([y_min - padding, y_max + padding])
    ax.set_xlim([x_sorted[0], x_sorted[-1]])
    
    # Initialize the line
    line, = ax.plot([], [], 'b-', lw=2)
    
    # Animation initialization function
    def init():
        line.set_data([], [])
        return line,
    
    # Animation update function
    def update(frame):
        f_frame = functions[frame]
        if is_hermite(f_frame):
            y = f_frame.dat.data_ro[::2]
        else:
            y = f_frame.dat.data_ro[:]
        y_sorted = y[x_sorted_idx]
        line.set_data(x_sorted, y_sorted)
        line.set_label(f'Frame num {frame}')
        leg = ax.legend()
        return line, leg
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=len(functions), interval=interval,
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

# Example usage
if __name__ == "__main__":
    # Create a 1D mesh
    n_elements = 100
    mesh = fdrk.UnitIntervalMesh(n_elements)
    # V = fdrk.FunctionSpace(mesh, "CG", 1)
    V = fdrk.FunctionSpace(mesh, "Hermite", 3)
    f = fdrk.Function(V)
    

    x = fdrk.SpatialCoordinate(mesh)[0]
    
    # Create some example functions
    functions = []
    t_values = np.linspace(0, 2*np.pi, 60)
    # t_values = [0]
    
    for t in t_values:
        # Create a traveling wave
        exp = fdrk.sin(2*fdrk.pi*(x - t))
        if is_hermite(f):
            f.project(exp)            
        else:
            f.interpolate(exp)
        functions.append(f.copy(deepcopy=True))
    
    # Create and display animation
    # Option 1: Display only
    anim1 = create_1d_function_animation(functions, mesh, 
                                    title="Traveling Wave",
                                    interval=1)
    