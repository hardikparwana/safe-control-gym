t = 0
jj = 0
z = cs.vertcat(sigma_points[:,jj], u_val[:,t])
pred = self.gaussian_process.casadi_predict(z=z)
mu = cs.reshape(pred['mean'], -1,1) + cs.reshape(self.prior_dynamics_func(x0=sigma_points[:,jj]-self.prior_ctrl.X_LIN[:,None], p=cs.reshape(u_val[:, t],2,1)-self.prior_ctrl.U_LIN[:,None])['xf'], -1,1) + cs.reshape(self.prior_ctrl.X_LIN[:,None], -1,1)
root_term = np.zeros((6,6))
new_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, cs.reshape(sigma_points[:,jj], -1,1), 1.0 )
new_weights = temp_weights * weights[:,jj]               


z = cs.vertcat(sigma_points[:,jj], u_val[:,t])
pred = self.gaussian_process.casadi_predict(z=z)
mu = cs.reshape(pred['mean'], -1,1) + cs.reshape(self.prior_dynamics_func(x0=sigma_points[:,jj]-self.prior_ctrl.X_LIN[:,None], p=cs.reshape(u_val[:, t],2,1)-self.prior_ctrl.U_LIN[:,None])['xf'], -1,1) + cs.reshape(self.prior_ctrl.X_LIN[:,None], -1,1)
root_term = np.zeros((6,6))
temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, cs.reshape(sigma_points[:,jj], -1,1), 1.0 )
new_points = cs.hcat([ new_points, temp_points ])
new_weights = cs.hcat([ new_weights, temp_weights * weights[:,jj]  ])  

weights[:,jj]
cd.sum2( temp_weights * weights[:,jj]     )