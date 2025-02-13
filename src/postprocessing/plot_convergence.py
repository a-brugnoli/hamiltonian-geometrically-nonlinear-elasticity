import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(dt_vec, dict_error_vec, **options):
    
    plt.figure()
    for key_label, error_vec in dict_error_vec.items():

            plt.plot(np.log10(dt_vec), np.log10(error_vec), '-.+', label=key_label)

        # # Define the coordinates of the triangle's vertices
        # if "rate" in options:
        #     empirical_rate = deg + options["rate"][count]
        # else:
        #     empirical_rate = deg 

    empirical_rate = np.log10(error_vec[-1]/error_vec[-2])/np.log10(dt_vec[-1]/dt_vec[-2]) 

    base_triangle = 0.5*abs(np.log10(dt_vec[-2]) - np.log10(dt_vec[-1]))
    height_triangle = empirical_rate*base_triangle
    shift_down = 0.2*(abs(np.log10(error_vec[-2]) - np.log10(error_vec[-1])))

    point1 = (np.log10(dt_vec[-1]), np.log10(error_vec[-1])-shift_down)
    point2 = (point1[0] + base_triangle, point1[1])
    point3 = (point2[0], point2[1] + height_triangle)

    x_triangle = [point1[0], point2[0], point3[0], point1[0]]  
    y_triangle = [point1[1], point2[1], point3[1], point1[1]]

    # Plot the triangle
    plt.plot(x_triangle, y_triangle, 'k')  # 'k-' specifies a black solid line
    # plt.text(0.5*(point1[0] + point2[0]), point1[1], '1', va='top', ha='left')  # Write '1' below the base
    plt.text(point2[0] + 0.1*base_triangle, 0.5*(point2[1] + point3[1]), f'{empirical_rate:.1f}', ha='left', va='center')  # Write 'empirical_rate' next to the height

    # Add grid
    plt.grid(True)
    
    plt.legend()

    if "title" in options:
        plt.title(options["title"])
    if "xlabel" in options:
        plt.xlabel(options["xlabel"])
    if "ylabel" in options:
        plt.ylabel(options["ylabel"])
    if "savefig" in options:
        plt.savefig(options["savefig"]+".pdf", dpi='figure', format='pdf', bbox_inches="tight")

