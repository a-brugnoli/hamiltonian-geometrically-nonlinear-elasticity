import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.postprocessing.options import configure_matplotib
configure_matplotib()

def plot_surface_from_matrix(t_vec, x_vec, z_values, **kwargs):
    """
    Create a 3D surface plot from a grid of samples.
    
    Parameters:
    -----------
    z_values : 2D numpy array
        Grid of f(x,t) values with shape (n, m)
    x_range : array-like, optional
        Array of x values. If None, uses indices
    t_range : array-like, optional
        Array of t values. If None, uses indices
    title : str, optional
        Title for the plot
    """
    
    # Create coordinate matrices
    X, T = np.meshgrid(x_vec, t_vec)
    
    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        
    # Plot the surface
    surface = ax.plot_surface(X, T, z_values, 
                            cmap='viridis',
                            edgecolor='none',
                            alpha=0.8)
    
    # Add colorbar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    if "x_label" in kwargs:
        ax.set_xlabel(kwargs["x_label"])
    else:
        ax.set_xlabel('$x$')
    if "y_label" in kwargs:
        ax.set_ylabel(kwargs["y_label"])
    else:
        ax.set_ylabel('$t$')
    if "z_label" in kwargs:
        ax.set_zlabel(kwargs["z_label"])
    else:
        ax.set_zlabel('$f(x,t)$')
    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    else:   
        ax.set_title("Surface Plot of f(x,t)")   
    # Adjust the viewing angle
    ax.view_init(elev=30, azim=45)
    
    return fig, ax

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    n_t, n_x = 50, 40
    t_vec = np.linspace(0, 10, n_t)
    x_vec = np.linspace(-5, 5, n_x)
    
    # Create a sample function z = sin(x) * cos(t)
    X, T = np.meshgrid(x_vec, t_vec)
    Z = np.sin(X) * np.cos(T)

    print(X.shape, T.shape, Z.shape)
    
    # Plot the surface
    fig, ax = plot_surface_from_matrix(t_vec, x_vec, Z)
    plt.show()