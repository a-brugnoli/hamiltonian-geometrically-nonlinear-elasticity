from src.norm_computation import error_norm_time, error_norm_space_time
    

norm_type = "L2"




#  # # Compute error
# error_q_x_dis_gradient = error_norm_space_time(q_x_array_reference, \
#                                                 q_x_array_dis_gradient, dt_output, norm=norm_type)
# error_q_x_lin_implicit = error_norm_space_time(q_x_array_reference, \
#                                                 q_x_array_lin_implicit, dt_output, norm=norm_type)

# error_q_z_dis_gradient = error_norm_space_time(q_z_array_reference, \
#                                                 q_z_array_dis_gradient, dt_output, norm=norm_type)
# error_q_z_lin_implicit = error_norm_space_time(q_z_array_reference, \
#                                                 q_z_array_lin_implicit, dt_output, norm=norm_type)

# error_v_x_dis_gradient = error_norm_space_time(v_x_array_reference, \
#                                                 v_x_array_dis_gradient, dt_output, norm=norm_type)
# error_v_x_lin_implicit = error_norm_space_time(v_x_array_reference, \
#                                                 v_x_array_lin_implicit, dt_output, norm=norm_type)

# error_v_z_dis_gradient = error_norm_space_time(v_z_array_reference, \
#                                                 v_z_array_dis_gradient, dt_output, norm=norm_type)
# error_v_z_lin_implicit = error_norm_space_time(v_z_array_reference, \
#                                                 v_z_array_lin_implicit, dt_output, norm=norm_type)


# error_vec_q_x_dis_gradient[ii] = error_q_x_dis_gradient
# error_vec_q_z_dis_gradient[ii] = error_q_z_dis_gradient

# error_vec_v_x_dis_gradient[ii] = error_v_x_dis_gradient
# error_vec_v_z_dis_gradient[ii] = error_v_z_dis_gradient

# diff_E_vec_dis_gradient[ii] = np.mean(np.diff(energy_vec_dis_gradient))

# elapsed_vec_dis_gradient[ii] = elapsed_dis_gradient

# error_vec_q_x_lin_implicit[ii] = error_q_x_lin_implicit
# error_vec_q_z_lin_implicit[ii] = error_q_z_lin_implicit

# error_vec_v_x_lin_implicit[ii] = error_v_x_lin_implicit
# error_vec_v_z_lin_implicit[ii] = error_v_z_lin_implicit

# diff_E_vec_lin_implicit[ii] = np.mean(np.diff(energy_vec_lin_implicit))

# elapsed_vec_lin_implicit[ii] = elapsed_lin_implicit



# dict_hor_position = {"Discrete gradient": error_vec_q_x_dis_gradient,\
#                 "Linear implicit": error_vec_q_x_lin_implicit}

# dict_ver_position = {"Discrete gradient": error_vec_q_z_dis_gradient,\
#                 "Linear implicit": error_vec_q_z_lin_implicit}


# dict_hor_velocity = {"Discrete gradient": error_vec_v_x_dis_gradient,\
#                 "Linear implicit": error_vec_v_x_lin_implicit}

# dict_ver_velocity = {"Discrete gradient": error_vec_v_z_dis_gradient,\
#                 "Linear implicit": error_vec_v_z_lin_implicit}


# str_xlabel = '$\log \Delta t \; \mathrm{[s]}$'
# plot_convergence(time_step_vec, dict_hor_position, xlabel=str_xlabel, ylabel="$\log \Delta q$", \
#                 title='Position error', savefig=f"{directory_results}convergence_horizontal_position.pdf")
# plot_convergence(time_step_vec, dict_hor_velocity, xlabel=str_xlabel, ylabel="$\log \Delta v$",  \
#                  title='Velocity error', savefig=f"{directory_results}convergence_horizontal_velocity.pdf")

# plot_convergence(time_step_vec, dict_ver_position, xlabel=str_xlabel, ylabel="$\log \Delta q$", \
#                 title='Position error', savefig=f"{directory_results}convergence_vertical_position.pdf")
# plot_convergence(time_step_vec, dict_ver_velocity, xlabel=str_xlabel, ylabel="$\log \Delta v$",  \
#                  title='Velocity error', savefig=f"{directory_results}convergence_vertical_velocity.pdf")


# plt.figure()
# plt.loglog(time_step_vec, diff_E_vec_dis_gradient, 'o-', label='Discrete gradient')
# plt.loglog(time_step_vec, diff_E_vec_lin_implicit, '+-', label='Linear implicit')
# plt.grid(color='0.8', linestyle='-', linewidth=.5)
# plt.xlabel(str_xlabel)
# plt.ylabel("$\log \Delta H$")

# plt.legend()
# plt.grid(True)
# plt.title("Energy error")
# plt.savefig(f"{directory_results}energy_error.pdf", dpi='figure', format='pdf', bbox_inches="tight")

# plt.figure()
# plt.loglog(time_step_vec, elapsed_vec_dis_gradient, 'o-', label='Discrete gradient')
# plt.loglog(time_step_vec, elapsed_vec_lin_implicit, '+-', label='Linear implicit')
# plt.grid(color='0.8', linestyle='-', linewidth=.5)
# plt.xlabel(str_xlabel)
# plt.ylabel("$\log \\tau$")
# plt.legend()
# plt.grid(True)
# plt.title("Computational time [ms]")
# plt.savefig(f"{directory_results}computational_time.pdf", dpi='figure', format='pdf', bbox_inches="tight")

# plt.show()
