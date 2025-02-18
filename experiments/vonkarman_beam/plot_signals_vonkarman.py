#     plt.figure()
#     plt.plot(t_vec_output, energy_vec_dis_gradient, label="Dis gradient")
#     plt.plot(t_vec_output, energy_vec_lin_implicit, label="Linear implicit")
#     plt.legend()
#     plt.xlabel("Time [ms]")
#     plt.title("Energy")

#     hor_disp_at_point_dis_gradient = q_x_array_dis_gradient[:, index_point]
#     hor_disp_at_point_lin_implicit = q_x_array_lin_implicit[:, index_point]

#     plt.figure()
#     plt.plot(t_vec_output, hor_disp_at_point_reference, label="reference")
#     plt.plot(t_vec_output, hor_disp_at_point_dis_gradient, label="Dis gradient")
#     plt.plot(t_vec_output, hor_disp_at_point_lin_implicit, label="Linear implicit")
#     plt.legend()
#     plt.xlabel("Time [ms]")
#     plt.title("Horizontal displacement")

#     ver_disp_at_point_dis_gradient = q_z_array_dis_gradient[:, index_point]
#     ver_disp_at_point_lin_implicit = q_z_array_lin_implicit[:, index_point]
#     # ver_disp_at_point_leapfrog = q_z_array_leapfrog[:, index_point]

#     plt.figure()
#     plt.plot(t_vec_output, ver_disp_at_point_reference, label="reference")
#     plt.plot(t_vec_output, ver_disp_at_point_dis_gradient, label="Dis gradient")
#     plt.plot(t_vec_output, ver_disp_at_point_lin_implicit, label="Linear implicit")
# #     # plt.plot(t_vec_output, ver_disp_at_point_leapfrog, label="Leapfrog")

#     plt.legend()
#     plt.xlabel("Time [ms]")
#     plt.title("Vertical displacement")

#     # fig_x_imp_midpoint, ax_x_imp_midpoint = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                             q_x_array_imp_midpoint, \
#     #                                             x_label="$x$", y_label="$t$", z_label="$q_x$", \
#     #                                             title="Horizontal displacement discrete gradient")

    
#     # fig_z_imp_midpoint, ax_z_imp_midpoint = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_z_array_imp_midpoint, \
#     #                                         x_label="$x \; \mathrm{[m]}$", \
#     #                                         y_label="$t \; \mathrm{[ms]}$", \
#     #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
#     #                                         title="Vertical displacement discrete gradient")


    
#     # fig_x_lin_implicit, ax_x_lin_implicit = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_x_array_lin_implicit, \
#     #                                         x_label="$x$", y_label="$t$", z_label="$q_x$", \
#     #                                         title="Horizontal displacement linear implicit")

    
#     # fig_z_lin_implicit, ax_z_lin_implicit = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_z_array_lin_implicit, \
#     #                                         x_label="$x \; \mathrm{[m]}$", \
#     #                                         y_label="$t \; \mathrm{[ms]}$", \
#     #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
#     #                                         title="Vertical displacement lin implicit")
    


#     # fig_x_dis_gradient, ax_x_dis_gradient = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                             q_x_array_dis_gradient, \
#     #                                             x_label="$x$", y_label="$t$", z_label="$q_x$", \
#     #                                             title="Horizontal displacement discrete gradient")

    
#     # fig_z_dis_gradient, ax_z_dis_gradient = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_z_array_dis_gradient, \
#     #                                         x_label="$x \; \mathrm{[m]}$", \
#     #                                         y_label="$t \; \mathrm{[ms]}$", \
#     #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
#     #                                         title="Vertical displacement discrete gradient")


#     # fig_x_leapfrog, ax_x_leapfrog = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                             q_x_array_leapfrog, \
#     #                                             x_label="$x$", y_label="$t$", z_label="$q_x$", \
#     #                                             title="Horizontal displacement leapfrog")

    
#     # fig_z_leapfrog, ax_z_leapfrog = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_z_array_leapfrog, \
#     #                                         x_label="$x \; \mathrm{[m]}$", \
#     #                                         y_label="$t \; \mathrm{[ms]}$", \
#     #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
#     #                                         title="Vertical displacement leapfrog")    
    
#     # anim_x_imp_midpoint = create_1d_line_animation(t_vec_output, x_vec, \
#     #                         q_x_array_imp_midpoint, interval=interval,
#     #                         xlabel="x", ylabel="$q_x$", \
#     #                         title=f"Horizontal displacement $\\alpha={alpha}$", \
#     #                         filename=f"{directory_results}Hor_disp_impl_midpoint.mp4")

#     # anim_z_imp_midpoint = create_1d_line_animation(t_vec_output, x_vec, \
#     #                                             q_z_array_imp_midpoint, interval=interval/10, \
#     #                                             xlabel="x", ylabel="$q_z$", \
#     #                                             title=f"Vertical displacement $\\alpha={alpha}$", \
#     #                                             filename=f"{directory_results}Ver_disp_impl_midpoint.mp4")
