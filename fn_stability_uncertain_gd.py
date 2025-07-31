import numpy as np

import jax.numpy as jnp 
from jax import jit, random, vmap, grad
import jaxopt
import time

import matplotlib.pyplot as plt 
from functools import partial
import jax
from jaxopt import GaussNewton, LevenbergMarquardt, GradientDescent
from jaxopt.projection import projection_polyhedron


class optim_cem_uneven():

    def __init__(self, t_fin, P, Pdot, Pddot, num_unknowns):

        # self.v_max = 0.7
        self.t_fin = t_fin

        self.num_batch = 500    #50
        self.num_ellite = 50    #10
        
        self.num_unknowns = num_unknowns

        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(P), jnp.asarray(Pdot), jnp.asarray(Pddot)

        self.nvar = jnp.shape(self.P_jax)[1]
        ###############################

        self.A_eq_x = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.P_jax[-1], self.Pdot_jax[-1]   ))
        self.A_eq_y = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.P_jax[-1], self.Pdot_jax[-1]   ))

        # self.cov_init = jnp.identity(self.nvar*2)
        # self.cost_smoothness = jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        self.A_projection = jnp.identity(self.nvar)

        self.maxiter = 1
        self.max_grad_iter = 25
        # self.gn = GaussNewton(residual_fun= self.error_func )
        self.gn = LevenbergMarquardt(residual_fun = self.error_func, jit=True)
        self.solve_nls_batch = jit(vmap(self.solve_nls, (0, None, None, None, None, None)  ))
        self.compute_cost_stability_batch = jit(vmap(self.compute_stability_cost, in_axes = (0, 0, 0)  ))
        # self.compute_grad_batch = jit(vmap(grad(self.compute_cost, argnums = (0)), in_axes = (0, None, None, None, None, None, None, None, None,None) ))
        # self.compute_cost_batch = jit(vmap(self.compute_cost, in_axes = (0, None, None, None, None, None, None, None,None,None,None,None )  ))
        self.compute_grad = jit(grad(self.compute_cost_gd, argnums = (0)))
        self.compute_grad_unc = jit(grad(self.compute_cost_gd_unc, argnums = (0)))
        self.compute_min_angle_batch = jit(vmap(self.compute_min_angle, in_axes = (0, 0, 0)  ))
        # self.projection_polyhedron_batch = jit(vmap(projection_polyhedron, in_axes= (0, None, None) ))
        self.proj_batch = jit(vmap(self.proj_single, in_axes= (0,None) ))
        self.compute_uncertainty_batch = jit(vmap(self.compute_uncertainty, in_axes = (0, 0, None, None, None, None, None, None, None)  ))
        self.hessian = jit(jax.hessian(self.fft_loss,argnums=0))
        self.compute_z_batch = jit(vmap(self.compute_z, in_axes=(None,None,0,None,None,None,None)))
        self.gd_solver = jaxopt.GradientDescent(fun=self.fft_gd,maxiter=25000,acceleration=False)
        self.custom_gd_batch = jit(vmap(self.custom_gd,in_axes=(None,None,None,None,0,None,0,None,0,0,0,0,0,None,None,None,0,0,None,None,None,None)))
        self.custom_gd_unc_batch = jit(vmap(self.custom_gd_unc,in_axes=(None,None,None,None,0,None,0,None,0,None,None,None,None,None,None,None,0,0,0,0,0,0,0)))
        self.compute_stability_cost_unc_batch = jit(vmap(self.compute_stability_cost_unc, in_axes=(None, None, None, None, None, 0, 0, 0)))
        self.compute_stability_grad = jit(vmap(grad(self.compute_stability_cost_unc, argnums=(0)), in_axes=(None, None, None, None, None, 0, 0, 0)))
        self.compute_stability_grad_single = jit(grad(self.compute_stability_cost_unc, argnums=(0)))
        self.compute_cost_stability_vec_unc_batch = jit(vmap(self.compute_cost_stability_vec_unc, in_axes=(0, 0))) 
        self.compute_cost_stability_vec_unc_batch_2 = jit(vmap(self.compute_cost_stability_vec_unc_2, in_axes=(0, 0))) 
        self.compute_stability_cov_batch = jit(vmap(self.compute_stability_cov, in_axes=(0, None)))

        self.num_samples = 1000
        
        ###################### Terrain parameters

        self.k1 = (2.5-1)/jnp.abs(2.5-1) # for i=1
        self.k2 = (2.5-2)/jnp.abs(2.5-2) # for i=2
        self.k3 = (2.5-3)/jnp.abs(2.5-3) # for i=3
        self.k4 = (2.5-4)/jnp.abs(2.5-4) # for i=4
        self.h = 0.420/2
        self.w = 0.544/2 
        self.lamda = 2
        self.a = 0.05
        self.m = 0.1
        self.g = 9.81
        self.beta = 3
        self.eta = 0.01
        self.l1,self.l2,self.l3,self.l4 = 0.260, 0.260, 0.260, 0.260
        self.num = jnp.shape(self.P_jax)[0]

        ############################ for computing initial guess trajectory

        # A = np.diff(np.diff(np.identity(self.num), axis = 0), axis = 0)

        # temp_1 = np.zeros(self.num)
        # temp_2 = np.zeros(self.num)
        # temp_3 = np.zeros(self.num)
        # temp_4 = np.zeros(self.num)

        # temp_1[0] = 1.0
        # temp_2[0] = -2
        # temp_2[1] = 1
        # temp_3[-1] = -2
        # temp_3[-2] = 1

        # temp_4[-1] = 1.0

        # A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))
        
        # R = np.dot(A_mat.T, A_mat)
        # self.mu = jnp.zeros(self.num)
        # cov = np.linalg.pinv(np.asarray(R))
        # self.cov = jnp.asarray(cov)
        ########################### min max x bounds
        self.A_bounds_x = jnp.vstack(( self.P_jax, -self.P_jax    ))
        self.A_bounds_y = jnp.vstack(( self.P_jax, -self.P_jax    ))
        # self.rho_bounds = 1

    @partial(jit, static_argnums = (0,))
    def fft_gd(self, lambdas, params, pcd):
        lambdas = lambdas.reshape(-1, 1)
        p1 = params[0:self.num_unknowns]
        p2 = params[self.num_unknowns:2*self.num_unknowns]
        p3 = params[2*self.num_unknowns:3*self.num_unknowns]
        p4 = params[3*self.num_unknowns:4*self.num_unknowns]
        xy = pcd[:, :2]
        gt_z = pcd[:, -1]
        kernel = self.get_kernel(xy, p1, p2, p3, p4)
        pred_z = (kernel@lambdas)[:,0]
        loss = jnp.sum((pred_z - gt_z)**2)+1e-10*jnp.sum(lambdas**2)
        # loss = jnp.linalg.norm(pred_z - gt_z)
        # loss = jnp.mean(jnp.log(jnp.cosh(pred_z-gt_z)))
      
        return loss

    @partial(jit, static_argnums=(0,))
    def compute_initial_mean_cov(self, x_init, x_fin, y_init, y_fin, key):
        
        key, subkey = jax.random.split(key)
        
        eps_k = jax.random.multivariate_normal(key, self.mu, 0.001*self.cov, (self.num_batch, ))

        goal_rot = -jnp.arctan2(y_fin-y_init, x_fin-x_init)
        
        x_init_temp = x_init*jnp.cos(goal_rot)-y_init*jnp.sin(goal_rot)
        y_init_temp = x_init*jnp.sin(goal_rot)+y_init*jnp.cos(goal_rot)


        x_fin_temp = x_fin*jnp.cos(goal_rot)-y_fin*jnp.sin(goal_rot)
        y_fin_temp = x_fin*jnp.sin(goal_rot)+y_fin*jnp.cos(goal_rot)


        x_interp = jnp.linspace(x_init_temp, x_fin_temp, self.num)
        y_interp = jnp.linspace(y_init_temp, y_fin_temp, self.num)

        x_guess_temp = jnp.asarray(x_interp+0.0*eps_k) 
        y_guess_temp = jnp.asarray(y_interp+eps_k)

        x_samples_init = x_guess_temp*jnp.cos(goal_rot)+y_guess_temp*jnp.sin(goal_rot)
        y_samples_init = -x_guess_temp*jnp.sin(goal_rot)+y_guess_temp*jnp.cos(goal_rot)

        # x_samples_init_temp = jnp.linspace(x_init, x_fin, self.num) 
        # y_samples_init_temp = jnp.linspace(y_init, y_fin, self.num)

        # x_samples_init = jnp.tile(x_samples_init_temp,(self.num_batch,1))
        # y_samples_init = jnp.tile(y_samples_init_temp,(self.num_batch,1))

        # cost_regression = jnp.dot(self.P_jax.T, self.P_jax)+0.1*jnp.identity(self.nvar)
        cost_regression = jnp.dot(self.P_jax.T, self.P_jax)+0.1*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        lincost_regression_x = -jnp.dot(self.P_jax.T, x_samples_init.T).T 
        lincost_regression_y = -jnp.dot(self.P_jax.T, y_samples_init.T).T 

        cost_mat_inv = jnp.linalg.inv(cost_regression)

        primal_sol_x_init = jnp.dot(cost_mat_inv, -lincost_regression_x.T).T 
        primal_sol_y_init = jnp.dot(cost_mat_inv, -lincost_regression_y.T).T

        primal_sol_init = jnp.hstack(( primal_sol_x_init, primal_sol_y_init  ))

        primal_sol_mean = jnp.mean(primal_sol_init, axis = 0)
        
        primal_sol_cov = jnp.cov(primal_sol_init.T)+0.001*jnp.identity(2*self.nvar)

        return primal_sol_mean, primal_sol_cov, key, x_samples_init, y_samples_init
        


    # @partial(jit, static_argnums=(0,))	
    # def compute_inital_guess( self, x_samples_init, y_samples_init):

    # 	cost_regression = jnp.dot(self.P_jax.T, self.P_jax)+0.0001*jnp.identity(self.nvar)
    # 	lincost_regression_x = -jnp.dot(self.P_jax.T, x_samples_init.T).T 
    # 	lincost_regression_y = -jnp.dot(self.P_jax.T, y_samples_init.T).T 

    # 	cost_mat_inv = jnp.linalg.inv(cost_regression)

    # 	c_x_samples_init = jnp.dot(cost_mat_inv, -lincost_regression_x.T).T 
    # 	c_y_samples_init = jnp.dot(cost_mat_inv, -lincost_regression_y.T).T

    # 	x_guess = jnp.dot(self.P_jax, c_x_samples_init.T).T
    # 	y_guess = jnp.dot(self.P_jax, c_y_samples_init.T).T

    # 	return c_x_samples_init, c_y_samples_init, x_guess, y_guess

        
    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_boundary_vec(self, x_init, vx_init, y_init, vy_init, x_fin, vx_fin, y_fin, vy_fin):
        
        x_init_vec = x_init*jnp.ones((self.num_batch, 1))
        vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))

        y_init_vec = y_init*jnp.ones((self.num_batch, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))

        x_fin_vec = x_fin*jnp.ones((self.num_batch, 1))
        vx_fin_vec = vx_fin*jnp.ones((self.num_batch, 1))

        y_fin_vec = y_fin*jnp.ones((self.num_batch, 1))
        vy_fin_vec = vy_fin*jnp.ones((self.num_batch, 1))
        
        

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, x_fin_vec, vx_fin_vec  ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, y_fin_vec, vy_fin_vec  ))

        return b_eq_x, b_eq_y


    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_boundary_vec_single(self, x_init, vx_init, y_init, vy_init, x_fin, vx_fin, y_fin, vy_fin):
        
        x_init_vec = x_init*jnp.ones((1, 1))
        vx_init_vec = vx_init*jnp.ones((1, 1))

        y_init_vec = y_init*jnp.ones((1, 1))
        vy_init_vec = vy_init*jnp.ones((1, 1))

        x_fin_vec = x_fin*jnp.ones((1, 1))
        vx_fin_vec = vx_fin*jnp.ones((1, 1))

        y_fin_vec = y_fin*jnp.ones((1, 1))
        vy_fin_vec = vy_fin*jnp.ones((1, 1))
        
        

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, x_fin_vec, vx_fin_vec  ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, y_fin_vec, vy_fin_vec  ))

        return b_eq_x, b_eq_y


    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_mean(self, x_init, vx_init, y_init, vy_init, x_fin, vx_fin, y_fin, vy_fin):
        
        b_eq_x = jnp.hstack(( x_init, vx_init, x_fin, vx_fin  ))
        b_eq_y = jnp.hstack(( y_init, vy_init, y_fin, vy_fin  ))

        cost_x = self.cost_smoothness
        cost_y = self.cost_smoothness 

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))

        lincost_x = jnp.zeros(self.nvar)
        lincost_y = jnp.zeros(self.nvar)
        

        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )))
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )))

        primal_sol_x = sol_x[0: self.nvar]
        primal_sol_y = sol_y[0: self.nvar]
        

        primal_sol_mean = jnp.hstack(( primal_sol_x, primal_sol_y ) )

        return primal_sol_mean 

    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_projection(self, b_eq_x, b_eq_y, primal_sol_x_samples, primal_sol_y_samples, x_max, x_min, y_max, y_min, s_bounds_x, s_bounds_y, lamda_x, lamda_y):
        
        
        # b_bounds_x = jnp.hstack(( x_max*jnp.ones(( self.num_batch, self.num  )), -x_min*jnp.ones(( self.num_batch, self.num  ))     ))

        # b_bounds_x_aug = b_bounds_x-s_bounds_x 

        # b_bounds_y = jnp.hstack(( y_max*jnp.ones(( self.num_batch, self.num  )), -y_min*jnp.ones(( self.num_batch, self.num  ))     ))

        # b_bounds_y_aug = b_bounds_y-s_bounds_y 
        

        lincost_x = -jnp.dot(self.A_projection.T, primal_sol_x_samples.T).T#-self.rho_bounds*jnp.dot(self.A_bounds_x.T, b_bounds_x_aug.T).T 
        lincost_y = -jnp.dot(self.A_projection.T, primal_sol_y_samples.T).T#-self.rho_bounds*jnp.dot(self.A_bounds_y.T, b_bounds_y_aug.T).T

        cost_x = jnp.identity(self.nvar)#+self.rho_bounds*jnp.dot(self.A_bounds_x.T, self.A_bounds_x)
        cost_y = jnp.identity(self.nvar)#+self.rho_bounds*jnp.dot(self.A_bounds_y.T, self.A_bounds_y)

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))

        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        primal_sol_x = sol_x[:, 0: self.nvar]
        primal_sol_y = sol_y[:, 0: self.nvar]
        

        x_samples = jnp.dot(self.P_jax, primal_sol_x.T).T 
        y_samples = jnp.dot(self.P_jax, primal_sol_y.T).T 

        xdot_samples = jnp.dot(self.Pdot_jax, primal_sol_x.T).T 
        ydot_samples = jnp.dot(self.Pdot_jax, primal_sol_y.T).T 
        
        xddot_samples = jnp.dot(self.Pddot_jax, primal_sol_x.T).T 
        yddot_samples = jnp.dot(self.Pddot_jax, primal_sol_y.T).T 
        
        primal_sol_samples = jnp.hstack(( primal_sol_x, primal_sol_y ) )

        # s_bounds_x = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num )), -jnp.dot(self.A_bounds_x, primal_sol_x.T).T+b_bounds_x  )

        # res_bounds_x = jnp.dot(self.A_bounds_x, primal_sol_x.T).T-b_bounds_x+s_bounds_x 

        # s_bounds_y = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num )), -jnp.dot(self.A_bounds_y, primal_sol_y.T).T+b_bounds_y  )

        # res_bounds_y = jnp.dot(self.A_bounds_y, primal_sol_y.T).T-b_bounds_y+s_bounds_y 

        # lamda_x = lamda_x-self.rho_bounds*jnp.dot(self.A_bounds_x.T, res_bounds_x.T).T
        # lamda_y = lamda_y-self.rho_bounds*jnp.dot(self.A_bounds_y.T, res_bounds_y.T).T

        # res_norm_bounds_x = jnp.linalg.norm(res_bounds_x, axis = 1)
        # res_norm_bounds_y = jnp.linalg.norm(res_bounds_y, axis = 1)
        

        return primal_sol_samples, x_samples, xdot_samples, xddot_samples, y_samples, ydot_samples, yddot_samples, s_bounds_x, s_bounds_y, lamda_x, lamda_y 


    @partial(jit, static_argnums=(0,),backend="gpu")
    def compute_projection_single(self, b_eq_x, b_eq_y, primal_sol_x_samples, primal_sol_y_samples, x_max, x_min, y_max, y_min, s_bounds_x, s_bounds_y, lamda_x, lamda_y):
        
        
        # b_bounds_x = jnp.hstack(( x_max*jnp.ones(( 1, self.num  )), -x_min*jnp.ones(( 1, self.num  ))     ))

        # b_bounds_x_aug = b_bounds_x-s_bounds_x 

        # b_bounds_y = jnp.hstack(( y_max*jnp.ones(( 1, self.num  )), -y_min*jnp.ones(( 1, self.num  ))     ))

        # b_bounds_y_aug = b_bounds_y-s_bounds_y 
        

        # print(self.A_projection.shape)
        # print(primal_sol_x_samples.shape)
        
        # print(self.A_bounds_x.shape)
        # print(b_bounds_x_aug.shape)

        lincost_x = -jnp.dot(self.A_projection.T, primal_sol_x_samples.T).T#-self.rho_bounds*jnp.dot(self.A_bounds_x.T, b_bounds_x_aug.T).T 
        lincost_y = -jnp.dot(self.A_projection.T, primal_sol_y_samples.T).T#-self.rho_bounds*jnp.dot(self.A_bounds_y.T, b_bounds_y_aug.T).T

        cost_x = jnp.identity(self.nvar)#+self.rho_bounds*jnp.dot(self.A_bounds_x.T, self.A_bounds_x)
        cost_y = jnp.identity(self.nvar)#+self.rho_bounds*jnp.dot(self.A_bounds_y.T, self.A_bounds_y)

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))

        # print(cost_mat_x.shape)
        # print(lincost_x.shape)
        # print(b_eq_x.shape)
        # print(jnp.hstack(( -lincost_x, b_eq_x )).T.shape)
        
        lincost_x = jnp.expand_dims(lincost_x,axis=0)
        lincost_y = jnp.expand_dims(lincost_y,axis=0)

        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        # print(sol_x.shape)

        primal_sol_x = sol_x[0,0: self.nvar]
        primal_sol_y = sol_y[0,0: self.nvar]
        
        # print(self.P_jax.shape)
        # print(primal_sol_x.T.shape)
        
        x_samples = jnp.dot(self.P_jax, primal_sol_x.T).T 
        y_samples = jnp.dot(self.P_jax, primal_sol_y.T).T 

        xdot_samples = jnp.dot(self.Pdot_jax, primal_sol_x.T).T 
        ydot_samples = jnp.dot(self.Pdot_jax, primal_sol_y.T).T 
        
        xddot_samples = jnp.dot(self.Pddot_jax, primal_sol_x.T).T 
        yddot_samples = jnp.dot(self.Pddot_jax, primal_sol_y.T).T 
        
        primal_sol_samples = jnp.hstack(( primal_sol_x, primal_sol_y ) )

        # s_bounds_x = jnp.maximum( jnp.zeros(( 1, 2*self.num )), -jnp.dot(self.A_bounds_x, primal_sol_x.T).T+b_bounds_x  )

        # res_bounds_x = jnp.dot(self.A_bounds_x, primal_sol_x.T).T-b_bounds_x+s_bounds_x 

        # s_bounds_y = jnp.maximum( jnp.zeros(( 1, 2*self.num )), -jnp.dot(self.A_bounds_y, primal_sol_y.T).T+b_bounds_y  )

        # res_bounds_y = jnp.dot(self.A_bounds_y, primal_sol_y.T).T-b_bounds_y+s_bounds_y 

        # lamda_x = lamda_x-self.rho_bounds*jnp.dot(self.A_bounds_x.T, res_bounds_x.T).T
        # lamda_y = lamda_y-self.rho_bounds*jnp.dot(self.A_bounds_y.T, res_bounds_y.T).T

        # res_norm_bounds_x = jnp.linalg.norm(res_bounds_x, axis = 1)
        # res_norm_bounds_y = jnp.linalg.norm(res_bounds_y, axis = 1)
        

        return primal_sol_samples, x_samples, xdot_samples, xddot_samples, y_samples, ydot_samples, yddot_samples, s_bounds_x, s_bounds_y, lamda_x, lamda_y 



        
    @partial(jit, static_argnums=(0,))	
    def compute_ellite_samples(self, cost_batch, primal_sol_samples):

        idx_ellite = jnp.argsort(cost_batch)


        primal_sol_ellite = primal_sol_samples[idx_ellite[0:self.num_ellite]]

        return 	primal_sol_ellite, idx_ellite 

    @partial(jit, static_argnums=(0,))	
    def compute_mean_cov(self, primal_sol_ellite):
        
        primal_sol_mean = jnp.mean(primal_sol_ellite, axis = 0)
        
        primal_sol_cov = jnp.cov(primal_sol_ellite.T)+0.01*jnp.identity(2*self.nvar)

        return primal_sol_mean, primal_sol_cov
        

    @partial(jit, static_argnums=(0,))	
    def compute_samples(self, primal_sol_mean, primal_sol_cov, key):
        
        key, subkey = jax.random.split(key)
                
        primal_sol_samples = jax.random.multivariate_normal(key, primal_sol_mean, primal_sol_cov, (self.num_batch, ))

        primal_sol_x_samples = primal_sol_samples[:, 0: self.nvar]
        primal_sol_y_samples = primal_sol_samples[:, self.nvar : 2*self.nvar]
        
        return primal_sol_x_samples, primal_sol_y_samples, key #


    @partial(jit, static_argnums = (0,))
    def compute_cost_gd(self, primal_sol_samples, lam, p1, p2, p3, p4):
        
        primal_sol_x = primal_sol_samples[0: self.nvar]
        primal_sol_y = primal_sol_samples[self.nvar: 2*self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x)
        y = jnp.dot(self.P_jax, primal_sol_y)

        xdot = jnp.dot(self.Pdot_jax, primal_sol_x)
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y)

        xddot = jnp.dot(self.Pddot_jax, primal_sol_x)
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y)

        psi = jnp.arctan2(ydot, xdot) 
        args = jnp.vstack(( x, y, psi  )).T
        
        pose_sol = self.solve_nls_batch(args,lam,p1,p2,p3,p4)
            
        cost_stability_vec, constraint_stability_vec = self.compute_cost_stability_batch(pose_sol, x, y)

        cost_smoothness = jnp.sum(xddot**2)+jnp.sum(yddot**2)

        curvature = yddot*xdot-xddot*ydot/(xdot**2+ydot**2+0.0001)**1.5

        cost_stability = jnp.sum(cost_stability_vec**2)

        cost_curvature = jnp.sum(curvature**2)

        # cost = 1000*cost_stability+cost_smoothness+1e-3*cost_curvature 
        cost = cost_stability+cost_smoothness+1e-3*cost_curvature 
        # jax.debug.print("cost {cost}", cost=cost)
        return cost


    @partial(jit, static_argnums = (0,))
    def compute_stability_cost_unc(self, lam, p1, p2, p3, p4, x, y, psi):
        
        args = jnp.hstack(( x, y, psi  ))
        
        pose_sol = self.solve_nls(args,lam,p1,p2,p3,p4)

        cost_stability, constraint_stability = self.compute_stability_cost(pose_sol, x, y)

        return cost_stability
    

    @partial(jit, static_argnums = (0,))
    def compute_cost_stability_vec_unc_2(self, cost_stability, stability_cov):    
        cost_stability = cost_stability*(1+(stability_cov))
        return cost_stability

    @partial(jit, static_argnums = (0,))
    def compute_cost_stability_vec_unc(self, cost_stability, stability_cov):    
        cost_stability = cost_stability*(1+(stability_cov*1e2))
        return cost_stability
    

    @partial(jit, static_argnums = (0,))
    def compute_stability_cov(self, grad_f, cov):
        stability_cov = grad_f.T@cov@grad_f
        return stability_cov
    

    @partial(jit, static_argnums = (0,))
    def compute_cost_gd_unc(self, primal_sol_samples, lam, p1, p2, p3, p4, stability_cov):
        
        primal_sol_x = primal_sol_samples[0: self.nvar]
        primal_sol_y = primal_sol_samples[self.nvar: 2*self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x)
        y = jnp.dot(self.P_jax, primal_sol_y)

        xdot = jnp.dot(self.Pdot_jax, primal_sol_x)
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y)

        xddot = jnp.dot(self.Pddot_jax, primal_sol_x)
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y)

        psi = jnp.arctan2(ydot, xdot) 

        cost_stability_vec = self.compute_stability_cost_unc_batch(lam, p1, p2, p3, p4, x, y, psi)
       
        cost_stability_vec = self.compute_cost_stability_vec_unc_batch(cost_stability_vec, stability_cov)

        cost_smoothness = jnp.sum(xddot**2)+jnp.sum(yddot**2)

        curvature = yddot*xdot-xddot*ydot/(xdot**2+ydot**2+0.0001)**1.5

        cost_stability = jnp.sum(cost_stability_vec**2)

        cost_curvature = jnp.sum(curvature**2)

        # cost = 1000*cost_stability+cost_smoothness+1e-3*cost_curvature 
        cost = cost_stability+cost_smoothness+1e-3*cost_curvature 
        # jax.debug.print("unc_cost {cost}", cost=cost)
        return cost


    @partial(jit, static_argnums = (0,))
    def error_func(self,opt_vars, args, lam, p1, p2, p3, p4):
        
        x, y, alpha = args[0], args[1], args[2]
            
        z = opt_vars[0]
        beta = opt_vars[1]
        gamma = opt_vars[2]

        x1 = opt_vars[3]
        y1 = opt_vars[4]
        z1 = opt_vars[5]

        x2 = opt_vars[6]
        y2 = opt_vars[7]
        z2 = opt_vars[8]

        x3 = opt_vars[9]
        y3 = opt_vars[10]
        z3 = opt_vars[11]

        x4 = opt_vars[12]
        y4 = opt_vars[13]
        z4 = opt_vars[14]

        mean1 = jnp.array([[x1,y1]])
        mean2 = jnp.array([[x2,y2]])
        mean3 = jnp.array([[x3,y3]])
        mean4 = jnp.array([[x4,y4]])

        kernel1 = self.get_kernel(mean1, p1, p2, p3, p4)
        kernel2 = self.get_kernel(mean2, p1, p2, p3, p4)
        kernel3 = self.get_kernel(mean3, p1, p2, p3, p4)
        kernel4 = self.get_kernel(mean4, p1, p2, p3, p4)

        z1_pred = (kernel1@lam).squeeze(axis=(-1,-2)) 
        z2_pred = (kernel2@lam).squeeze(axis=(-1,-2)) 
        z3_pred = (kernel3@lam).squeeze(axis=(-1,-2)) 
        z4_pred = (kernel4@lam).squeeze(axis=(-1,-2)) 

        eqn1 = x + self.h*jnp.cos(alpha)*jnp.cos(beta) + self.k1*self.w*(jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma) - jnp.sin(alpha)*jnp.cos(gamma)) - self.l1*(jnp.cos(alpha)*jnp.sin(beta)*jnp.cos(gamma) - jnp.sin(alpha)*jnp.sin(gamma))-x1 
        eqn2 = y + self.h*jnp.sin(alpha)*jnp.cos(beta) + self.k1*self.w*(jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma) + jnp.cos(alpha)*jnp.cos(gamma)) - self.l1*(jnp.sin(alpha)*jnp.sin(beta)*jnp.cos(gamma) - jnp.cos(alpha)*jnp.sin(gamma))-y1 
        eqn3 = z - self.h*jnp.sin(beta) + self.k1*self.w*jnp.cos(beta)*jnp.sin(gamma) - self.l1*jnp.cos(beta)*jnp.cos(gamma) - z1 
        eqn4 = z1 - z1_pred		#3.5*(0.3*jnp.cos(0.3*x1) + 0.6*jnp.sin(0.2*y1)) 

        eqn5 = x - self.h*jnp.cos(alpha)*jnp.cos(beta) + self.k2*self.w*(jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma) - jnp.sin(alpha)*jnp.cos(gamma)) - self.l2*(jnp.cos(alpha)*jnp.sin(beta)*jnp.cos(gamma) - jnp.sin(alpha)*jnp.sin(gamma))-x2 
        eqn6 = y - self.h*jnp.sin(alpha)*jnp.cos(beta) + self.k2*self.w*(jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma) + jnp.cos(alpha)*jnp.cos(gamma)) - self.l2*(jnp.sin(alpha)*jnp.sin(beta)*jnp.cos(gamma) - jnp.cos(alpha)*jnp.sin(gamma))-y2 
        eqn7 = z + self.h*jnp.sin(beta) + self.k2*self.w*jnp.cos(beta)*jnp.sin(gamma) - self.l2*jnp.cos(beta)*jnp.cos(gamma) - z2 
        eqn8 = z2 - z2_pred		#3.5*(0.3*jnp.cos(0.3*x2) + 0.6*jnp.sin(0.2*y2)) 

        eqn9 = x - self.h*jnp.cos(alpha)*jnp.cos(beta) + self.k3*self.w*(jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma) - jnp.sin(alpha)*jnp.cos(gamma)) - self.l3*(jnp.cos(alpha)*jnp.sin(beta)*jnp.cos(gamma) - jnp.sin(alpha)*jnp.sin(gamma))-x3 
        eqn10 = y - self.h*jnp.sin(alpha)*jnp.cos(beta) + self.k3*self.w*(jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma) + jnp.cos(alpha)*jnp.cos(gamma)) - self.l3*(jnp.sin(alpha)*jnp.sin(beta)*jnp.cos(gamma) - jnp.cos(alpha)*jnp.sin(gamma))-y3 
        eqn11 = z + self.h*jnp.sin(beta) + self.k3*self.w*jnp.cos(beta)*jnp.sin(gamma) - self.l3*jnp.cos(beta)*jnp.cos(gamma) - z3
        eqn12 = z3 - z3_pred		#3.5*(0.3*jnp.cos(0.3*x3) + 0.6*jnp.sin(0.2*y3))  

        eqn13 = x + self.h*jnp.cos(alpha)*jnp.cos(beta) + self.k4*self.w*(jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma) - jnp.sin(alpha)*jnp.cos(gamma)) - self.l4*(jnp.cos(alpha)*jnp.sin(beta)*jnp.cos(gamma) - jnp.sin(alpha)*jnp.sin(gamma))-x4 
        eqn14 = y + self.h*jnp.sin(alpha)*jnp.cos(beta) + self.k4*self.w*(jnp.cos(alpha)*jnp.sin(beta)*jnp.sin(gamma) + jnp.cos(alpha)*jnp.cos(gamma)) - self.l4*(jnp.sin(alpha)*jnp.sin(beta)*jnp.cos(gamma) - jnp.cos(alpha)*jnp.sin(gamma))-y4 
        eqn15 = z - self.h*jnp.sin(beta) + self.k4*self.w*jnp.cos(beta)*jnp.sin(gamma) - self.l4*jnp.cos(beta)*jnp.cos(gamma) - z4
        eqn16 = z4 - z4_pred		#3.5*(0.3*jnp.cos(0.3*x4) + 0.6*jnp.sin(0.2*y4)) 

        return jnp.array([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9, eqn10, eqn11, eqn12, eqn13, eqn14, eqn15, eqn16])


    @partial(jit, static_argnums = (0,))
    def compute_stability_cost(self, pose_sol, X, Y):
        
        z = pose_sol[0]
        beta = pose_sol[1]
        gamma = pose_sol[2]

        x1 = pose_sol[3]
        y1 = pose_sol[4]
        z1 = pose_sol[5]

        x2 = pose_sol[6]
        y2 = pose_sol[7]
        z2 = pose_sol[8]

        x3 = pose_sol[9]
        y3 = pose_sol[10]
        z3 = pose_sol[11]

        x4 = pose_sol[12]
        y4 = pose_sol[13]
        z4 = pose_sol[14]

        Poc1 = jnp.hstack([x1,y1,z1]).reshape(3,1)    # position vectors of contact points
        Poc2 = jnp.hstack([x2,y2,z2]).reshape(3,1)
        Poc3 = jnp.hstack([x3,y3,z3]).reshape(3,1)
        Poc4 = jnp.hstack([x4,y4,z4]).reshape(3,1)
        Pog  = jnp.hstack([X,Y,z]).reshape(3,1)       # position of CM of vehicle

        e1 = Poc2 - Poc1
        e2 = Poc3 - Poc2
        e3 = Poc4 - Poc3
        e4 = Poc1 - Poc4

        e1_hat = e1/(jnp.linalg.norm(e1)+1e-6)
        e2_hat = e2/(jnp.linalg.norm(e2)+1e-6)
        e3_hat = e3/(jnp.linalg.norm(e3)+1e-6)
        e4_hat = e4/(jnp.linalg.norm(e4)+1e-6)

        pi1 = (jnp.eye(3)-e1_hat@e1_hat.T)@(Poc2-Pog)
        pi2 = (jnp.eye(3)-e2_hat@e2_hat.T)@(Poc3-Pog)
        pi3 = (jnp.eye(3)-e3_hat@e3_hat.T)@(Poc4-Pog)
        pi4 = (jnp.eye(3)-e4_hat@e4_hat.T)@(Poc1-Pog)

        pi1_hat = pi1/(jnp.linalg.norm(pi1)+1e-6)
        pi2_hat = pi2/(jnp.linalg.norm(pi2)+1e-6)
        pi3_hat = pi3/(jnp.linalg.norm(pi3)+1e-6)
        pi4_hat = pi4/(jnp.linalg.norm(pi4)+1e-6)

        F_net = jnp.array([[0.],[0.],[-self.m*self.g]]) # net force acting on vehicle

        F1 = (jnp.eye(3)-e1_hat@e1_hat.T)@F_net
        F2 = (jnp.eye(3)-e2_hat@e2_hat.T)@F_net
        F3 = (jnp.eye(3)-e3_hat@e3_hat.T)@F_net
        F4 = (jnp.eye(3)-e4_hat@e4_hat.T)@F_net

        F1_hat = F1/(jnp.linalg.norm(F1)+1e-6)
        F2_hat = F2/(jnp.linalg.norm(F2)+1e-6)
        F3_hat = F3/(jnp.linalg.norm(F3)+1e-6)
        F4_hat = F4/(jnp.linalg.norm(F4)+1e-6)

        sigma1 = jnp.tanh(jnp.dot(jnp.cross(pi1_hat.T,F1_hat.T).squeeze(),e1_hat.squeeze()))
        sigma2 = jnp.tanh(jnp.dot(jnp.cross(pi2_hat.T,F2_hat.T).squeeze(),e2_hat.squeeze()))
        sigma3 = jnp.tanh(jnp.dot(jnp.cross(pi3_hat.T,F3_hat.T).squeeze(),e3_hat.squeeze()))
        sigma4 = jnp.tanh(jnp.dot(jnp.cross(pi4_hat.T,F4_hat.T).squeeze(),e4_hat.squeeze()))   

        # sigma1 = jnp.sign(jnp.dot(jnp.cross(pi1_hat.T,F1_hat.T).squeeze(),e1_hat.squeeze()))
        # sigma2 = jnp.sign(jnp.dot(jnp.cross(pi2_hat.T,F2_hat.T).squeeze(),e2_hat.squeeze()))
        # sigma3 = jnp.sign(jnp.dot(jnp.cross(pi3_hat.T,F3_hat.T).squeeze(),e3_hat.squeeze()))
        # sigma4 = jnp.sign(jnp.dot(jnp.cross(pi4_hat.T,F4_hat.T).squeeze(),e4_hat.squeeze()))   


        tip1 = sigma1*jnp.arccos(jnp.dot(F1_hat.squeeze(),pi1_hat.squeeze()))             # tip over stability angles (theta)
        tip2 = sigma2*jnp.arccos(jnp.dot(F2_hat.squeeze(),pi2_hat.squeeze()))
        tip3 = sigma3*jnp.arccos(jnp.dot(F3_hat.squeeze(),pi3_hat.squeeze()))
        tip4 = sigma4*jnp.arccos(jnp.dot(F4_hat.squeeze(),pi4_hat.squeeze()))

        # tip1_cost = jnp.log(1 + jnp.exp((-tip1+0.1)*self.beta))/self.beta              # cost which penalizes theta<0
        # tip2_cost = jnp.log(1 + jnp.exp((-tip2+0.1)*self.beta))/self.beta
        # tip3_cost = jnp.log(1 + jnp.exp((-tip3+0.1)*self.beta))/self.beta
        # tip4_cost = jnp.log(1 + jnp.exp((-tip4+0.1)*self.beta))/self.beta

        tip1_cost = jnp.maximum(0, -tip1+0.1  )
        tip2_cost = jnp.maximum(0, -tip2+0.1  )
        tip3_cost = jnp.maximum(0, -tip3+0.1  )
        tip4_cost = jnp.maximum(0, -tip4+0.1  )
        
        
        

        tip12_cost = (tip1-tip2)**2                                                         # cost to minimize difference between four angles
        tip23_cost = (tip2-tip3)**2
        tip34_cost = (tip3-tip4)**2
        tip41_cost = (tip4-tip1)**2

        # tip12_cost = (tip1-1.57)**2                                                       # cost to minimize difference between four angles
        # tip23_cost = (tip2-1.57)**2
        # tip34_cost = (tip3-1.57)**2
        # tip41_cost = (tip4-1.57)**2


        cost_stability = tip1_cost+tip2_cost+tip3_cost+tip4_cost+0.05*(tip12_cost+tip23_cost+tip34_cost+tip41_cost)
        constraint_stability = tip1_cost+tip2_cost+tip3_cost+tip4_cost
        # cost_stability = tip12_cost+tip23_cost+tip34_cost+tip41_cost
        # cost_stability = beta**2+gamma**2

        return cost_stability, constraint_stability


    @partial(jit, static_argnums = (0,))
    def compute_min_angle(self, pose_sol, X, Y):
        
        z = pose_sol[0]
        beta = pose_sol[1]
        gamma = pose_sol[2]

        x1 = pose_sol[3]
        y1 = pose_sol[4]
        z1 = pose_sol[5]

        x2 = pose_sol[6]
        y2 = pose_sol[7]
        z2 = pose_sol[8]

        x3 = pose_sol[9]
        y3 = pose_sol[10]
        z3 = pose_sol[11]

        x4 = pose_sol[12]
        y4 = pose_sol[13]
        z4 = pose_sol[14]

        Poc1 = jnp.hstack([x1,y1,z1]).reshape(3,1)    # position vectors of contact points
        Poc2 = jnp.hstack([x2,y2,z2]).reshape(3,1)
        Poc3 = jnp.hstack([x3,y3,z3]).reshape(3,1)
        Poc4 = jnp.hstack([x4,y4,z4]).reshape(3,1)
        Pog  = jnp.hstack([X,Y,z]).reshape(3,1)       # position of CM of vehicle

        e1 = Poc2 - Poc1
        e2 = Poc3 - Poc2
        e3 = Poc4 - Poc3
        e4 = Poc1 - Poc4

        e1_hat = e1/(jnp.linalg.norm(e1)+1e-6)
        e2_hat = e2/(jnp.linalg.norm(e2)+1e-6)
        e3_hat = e3/(jnp.linalg.norm(e3)+1e-6)
        e4_hat = e4/(jnp.linalg.norm(e4)+1e-6)

        pi1 = (jnp.eye(3)-e1_hat@e1_hat.T)@(Poc2-Pog)
        pi2 = (jnp.eye(3)-e2_hat@e2_hat.T)@(Poc3-Pog)
        pi3 = (jnp.eye(3)-e3_hat@e3_hat.T)@(Poc4-Pog)
        pi4 = (jnp.eye(3)-e4_hat@e4_hat.T)@(Poc1-Pog)

        pi1_hat = pi1/(jnp.linalg.norm(pi1)+1e-6)
        pi2_hat = pi2/(jnp.linalg.norm(pi2)+1e-6)
        pi3_hat = pi3/(jnp.linalg.norm(pi3)+1e-6)
        pi4_hat = pi4/(jnp.linalg.norm(pi4)+1e-6)

        F_net = jnp.array([[0.],[0.],[-self.m*self.g]]) # net force acting on vehicle

        F1 = (jnp.eye(3)-e1_hat@e1_hat.T)@F_net
        F2 = (jnp.eye(3)-e2_hat@e2_hat.T)@F_net
        F3 = (jnp.eye(3)-e3_hat@e3_hat.T)@F_net
        F4 = (jnp.eye(3)-e4_hat@e4_hat.T)@F_net

        F1_hat = F1/(jnp.linalg.norm(F1)+1e-6)
        F2_hat = F2/(jnp.linalg.norm(F2)+1e-6)
        F3_hat = F3/(jnp.linalg.norm(F3)+1e-6)
        F4_hat = F4/(jnp.linalg.norm(F4)+1e-6)

        sigma1 = jnp.tanh(jnp.dot(jnp.cross(pi1_hat.T,F1_hat.T).squeeze(),e1_hat.squeeze()))
        sigma2 = jnp.tanh(jnp.dot(jnp.cross(pi2_hat.T,F2_hat.T).squeeze(),e2_hat.squeeze()))
        sigma3 = jnp.tanh(jnp.dot(jnp.cross(pi3_hat.T,F3_hat.T).squeeze(),e3_hat.squeeze()))
        sigma4 = jnp.tanh(jnp.dot(jnp.cross(pi4_hat.T,F4_hat.T).squeeze(),e4_hat.squeeze()))   

        # sigma1 = jnp.sign(jnp.dot(jnp.cross(pi1_hat.T,F1_hat.T).squeeze(),e1_hat.squeeze()))
        # sigma2 = jnp.sign(jnp.dot(jnp.cross(pi2_hat.T,F2_hat.T).squeeze(),e2_hat.squeeze()))
        # sigma3 = jnp.sign(jnp.dot(jnp.cross(pi3_hat.T,F3_hat.T).squeeze(),e3_hat.squeeze()))
        # sigma4 = jnp.sign(jnp.dot(jnp.cross(pi4_hat.T,F4_hat.T).squeeze(),e4_hat.squeeze()))   


        tip1 = sigma1*jnp.arccos(jnp.dot(F1_hat.squeeze(),pi1_hat.squeeze()))             # tip over stability angles (theta)
        tip2 = sigma2*jnp.arccos(jnp.dot(F2_hat.squeeze(),pi2_hat.squeeze()))
        tip3 = sigma3*jnp.arccos(jnp.dot(F3_hat.squeeze(),pi3_hat.squeeze()))
        tip4 = sigma4*jnp.arccos(jnp.dot(F4_hat.squeeze(),pi4_hat.squeeze()))

        min_tip_angle = jnp.min(jnp.array([tip1,tip2,tip3,tip4]))

        return min_tip_angle

    
    @partial(jit, static_argnums = (0,))
    def compute_uncertainty(self, X, Y, key, lam, cov, p1, p2, p3, p4):
        key, subkey = jax.random.split(key)
        lam = lam.squeeze(axis=-1)
        sampled_lam = jax.random.multivariate_normal(subkey,lam,cov,shape=(self.num_samples,))  
       
        z_values = self.compute_z_batch(X,Y,sampled_lam,p1,p2,p3,p4)    
        std_z = jnp.std(z_values)

        return std_z

    @partial(jit, static_argnums = (0,))
    def compute_z(self,X,Y,sampled_params,p1,p2,p3,p4):
        lam = sampled_params.reshape(-1,1)
        points = jnp.vstack((X,Y)).T
        kernel = self.get_kernel(points, p1, p2, p3, p4)
        z = jnp.clip((kernel@lam)[:,0],-6.0,6.0)
        return z

    @partial(jit, static_argnums = (0,))
    def compute_cost(self, primal_sol_samples, x_max, x_min, y_max, y_min, lam, p1, p2, p3, p4, key, cov):
        
        primal_sol_x = primal_sol_samples[0: self.nvar]
        primal_sol_y = primal_sol_samples[self.nvar: 2*self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x)
        y = jnp.dot(self.P_jax, primal_sol_y)

        xdot = jnp.dot(self.Pdot_jax, primal_sol_x)
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y)

        xddot = jnp.dot(self.Pddot_jax, primal_sol_x)
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y)

        psi = jnp.arctan2(ydot, xdot) 
        args = jnp.vstack(( x, y, psi  )).T
        
        pose_sol = self.solve_nls_batch(args,lam,p1,p2,p3,p4)
        
        # stability = []
        # # constraint = []

        # for i in range(0, self.num):
        
        # 	cost_stability_vec, constraint_stability_vec = self.compute_stability_cost(pose_sol[i], x[i], y[i])
        # 	stability.append(cost_stability_vec)
            
        cost_stability_vec, constraint_stability_vec = self.compute_cost_stability_batch(pose_sol, x, y)

        # cost_smoothness = jnp.linalg.norm(xddot)+jnp.linalg.norm(yddot)
        cost_smoothness = jnp.sum(xddot**2)+jnp.sum(yddot**2)

        curvature = yddot*xdot-xddot*ydot/(xdot**2+ydot**2+0.0001)**1.5

        # cost_stability = jnp.linalg.norm(cost_stability_vec)
        cost_stability = jnp.sum(cost_stability_vec**2)

        # constraint_stability_batch = jnp.linalg.norm(constraint_stability_matrix, axis = 1)

        # cost_curvature = jnp.linalg.norm(curvature)
        cost_curvature = jnp.sum(curvature**2)

        # cost_bounds_x = jnp.linalg.norm(jnp.log(jnp.maximum( jnp.zeros(self.num),   x-x_max   )+1.0))+jnp.linalg.norm(jnp.log(jnp.maximum( jnp.zeros(self.num),   -x+x_min   )+1.0))
        # cost_bounds_y = jnp.linalg.norm(jnp.log(jnp.maximum( jnp.zeros(self.num),   y-y_max   )+1.0))+jnp.linalg.norm(jnp.log(jnp.maximum( jnp.zeros(self.num),   -y+y_min   )+1.0))

        # cost_bounds_x = jnp.linalg.norm(jnp.maximum( jnp.zeros(self.num),   x-x_max   ))+jnp.linalg.norm(jnp.maximum( jnp.zeros(self.num),   -x+x_min   ))
        # cost_bounds_y = jnp.linalg.norm(jnp.maximum( jnp.zeros(self.num),   y-y_max   ))+jnp.linalg.norm(jnp.maximum( jnp.zeros(self.num),   -y+y_min   ))

        # x_max_cost = jnp.log(1 + jnp.exp(-(x_max-x)*2))/2
        # x_min_cost = jnp.log(1 + jnp.exp(-(x-x_min)*2))/2
        
        # y_max_cost = jnp.log(1 + jnp.exp(-(y_max-y)*2))/2
        # y_min_cost = jnp.log(1 + jnp.exp(-(y-y_min)*2))/2

        # cost_bounds_tot = x_max_cost+x_min_cost+y_max_cost+y_min_cost

        # cost_bounds = jnp.linalg.norm(cost_bounds_tot)
        
        uncertain_cost_vec = self.compute_uncertainty_batch(x, y, key, lam, cov, p1, p2, p3, p4)

        uncertain_cost = jnp.sum(jnp.exp(10*uncertain_cost_vec))

        cost = 1000*cost_stability+cost_smoothness+1e-3*cost_curvature #0.001*cost_smoothness+0.006*cost_curvature#cost_stability++0.001*cost_bounds  #+cost_bounds_x+cost_bounds_y 

        final_cost = uncertain_cost*cost

        return final_cost


    @partial(jit, static_argnums = (0,))
    def custom_cost_fn(self,current_x, current_y, current_theta, 
                            next_x, next_y, next_theta, 
                            base_cost, lam, p1, p2, p3, p4, key, cov):
        """
        JIT-compiled custom cost function.
        """
        # Calculate turning penalty
        theta_diff = jnp.abs(jnp.mod(next_theta - current_theta + jnp.pi, 2 * jnp.pi) - jnp.pi)
        turning_penalty = 0.2 * theta_diff
        
        args = jnp.hstack(( next_x, next_y, next_theta  )).T
        
        # nls_start_time = time.time()
        # pose_sol = jax.block_until_ready(self.solve_nls(args,lam,p1,p2,p3,p4))
        # print("nls_time =", time.time()-nls_start_time)
        pose_sol = self.solve_nls(args,lam,p1,p2,p3,p4)

        cost_stability, constraint_stability = self.compute_stability_cost(pose_sol, next_x, next_y)

        stability_cost = cost_stability**2

        # unc_start_time = time.time()
        uncertain_cost = self.compute_uncertainty(next_x, next_y, key, lam, cov, p1, p2, p3, p4)
        # uncertain_cost = jax.block_until_ready(self.compute_uncertainty(next_x, next_y, key, lam, cov, p1, p2, p3, p4))
        # print("unc time =", time.time()-unc_start_time)

        uncertain_cost = jnp.exp(uncertain_cost)

        cost = base_cost + turning_penalty + stability_cost #+ 0.5*uncertain_cost

        final_cost = uncertain_cost*cost
            
        return final_cost
		
	
    @partial(jit, static_argnums = (0,))
    def custom_cost_fn_unc(self,current_x, current_y, current_theta, 
                            next_x, next_y, next_theta, 
                            base_cost, lam, p1, p2, p3, p4, key, cov):
        """
        JIT-compiled custom cost function.
        """
        # Calculate turning penalty
        theta_diff = jnp.abs(jnp.mod(next_theta - current_theta + jnp.pi, 2 * jnp.pi) - jnp.pi)
        turning_penalty = 0.2 * theta_diff
        
        args = jnp.hstack(( next_x, next_y, next_theta  )).T
        
        # nls_start_time = time.time()
        # pose_sol = jax.block_until_ready(self.solve_nls(args,lam,p1,p2,p3,p4))
        # print("nls_time =", time.time()-nls_start_time)
        pose_sol = self.solve_nls(args,lam,p1,p2,p3,p4)

        cost_stability, constraint_stability = self.compute_stability_cost(pose_sol, next_x, next_y)
      
        grad_f = self.compute_stability_grad_single(lam, p1, p2, p3, p4, next_x, next_y, next_theta)

        stability_cov = grad_f.T@cov@grad_f
        
        cost_stability_unc = cost_stability*(1+(stability_cov*1e2))

        cost = base_cost + turning_penalty + cost_stability_unc**2
         
        return jnp.squeeze(cost) 


    @partial(jit, static_argnums = (0,))
    def solve_nls(self, args, lam, p1, p2, p3, p4):
        
        # initial_guess = jnp.zeros(15)
        initial_guess = jnp.array([0.0,   0.0,  0.0, args[0],    args[1],    0.0,\
                            args[0],    args[1],    0.0,   args[0],     args[1],   0.0,\
                                    args[0],    args[1],    0.0])

        gn_sol = self.gn.run(initial_guess, args, lam, p1, p2, p3, p4).params

        return gn_sol


    @partial(jit, static_argnums = (0,))
    def fft_kernel(self, points, p1, p2, p3, p4):
        """
        Function for creating fft kernel per point
        point = list or array of size 2 containing x and y
        indexes = indexes obtained from get_indexes()
        """
        x = points[0]
        y = points[1]
        
        return jnp.concatenate((jnp.cos(p1*x + p2*y), jnp.sin(p3*x + p4*y)))


    @partial(jit, static_argnums = (0,))
    def costFunction(self, weights, kernel, z):
        loss = jnp.sum((kernel@weights - z)**2)
        return loss
		
    
    @partial(jit, static_argnums = (0,))
    def get_indexes(self):
        """
        frequency = frequency used for modelling fft for surface fitting
        """
        u = jnp.arange(self.frequency)
        v = jnp.arange(self.frequency)
        indexes = jnp.array(jnp.meshgrid(u,v)).T.reshape(-1,2)
        return indexes

    @partial(jit, static_argnums = (0,))
    def fft_gradient_descent(self,lambdas, kernel, gt_z):
        pred_z = (kernel@lambdas)[:,0]
        loss = jnp.sum((pred_z - gt_z)**2)
        
        return loss

    @partial(jit, static_argnums = (0,))
    def get_kernel(self, xy, p1, p2, p3, p4):
        """
        Creates a fft kernel from the local point cloud patch
        fft_batch = vmap function for creating the fft kernel
        local_xyz = local patch obtained from get_local_patch(0)
        indexes = frequencies for fft obtained from get_indexes()
        """
        c_term = jnp.vstack((p1, p2))
        s_term = jnp.vstack((p3, p4))
        kernel = jnp.hstack((jnp.cos(xy[:, :2]@c_term), jnp.sin(xy[:, :2]@s_term)))
        
        return kernel

    @partial(jit, static_argnums = (0,))
    def fft_gradient_descent(self, lambdas, p1, p2, p3, p4, pcd):
        xy = pcd[:, :2]
        gt_z = pcd[:, -1]
        kernel = self.get_kernel(xy, p1, p2, p3, p4)
        pred_z = (kernel@lambdas)[:,0]
        loss = jnp.linalg.norm(pred_z - gt_z)
        return loss

    @partial(jit, static_argnums = (0,))
    def fft_loss(self, lambdas, params, pcd):
        lambdas = lambdas.reshape(-1, 1)
        p1 = params[0:self.num_unknowns]
        p2 = params[self.num_unknowns:2*self.num_unknowns]
        p3 = params[2*self.num_unknowns:3*self.num_unknowns]
        p4 = params[3*self.num_unknowns:4*self.num_unknowns]
        xy = pcd[:, :2]
        gt_z = pcd[:, -1]
        kernel = self.get_kernel(xy, p1, p2, p3, p4)
        pred_z = (kernel@lambdas)[:,0]
        loss = jnp.sum((pred_z - gt_z)**2)
        # loss = jnp.mean(jnp.log(jnp.cosh(pred_z-gt_z)))
        return loss


    @partial(jit, static_argnums = (0,))
    def fft_nls(self,lambdas, kernel, gt_z):
        pred_z = (kernel@lambdas.reshape(-1, 1))[:,0]
        loss = pred_z - gt_z
        return loss


    @partial(jit, static_argnums = (0,))
    def lossFunction(self,x, y, z, weights, p1, p2, p3, p4):
        xy = jnp.vstack((x, y)).T
        # kernel = jax_fft(xy, indexes)
        kernel = self.get_kernel(xy, p1, p2, p3, p4)
        # loss = kernel@weights - z
        pred_z = kernel@weights
        loss = jnp.sum((pred_z - z)**2)
        return loss

    @partial(jit, static_argnums=(0,))
    def proj_single(self,p,C):
        return projection_polyhedron(p,C,check_feasible = False)





    @partial(jit, static_argnums=(0,))	
    def custom_gd(self, x_init, vx_init, y_init, vy_init, x_fin, vx_fin, y_fin, vy_fin, lam, p1, p2, p3, p4, eta, beta_1, beta_2, x, y,
                        x_max,y_max,x_min,y_min):

        x_samples_init = jnp.linalg.inv(self.P_jax.T @ self.P_jax+0.001*jnp.identity(self.nvar)) @ self.P_jax.T @ x
        y_samples_init = jnp.linalg.inv(self.P_jax.T @ self.P_jax+0.001*jnp.identity(self.nvar)) @ self.P_jax.T @ y

        primal_sol_samples_init = jnp.concatenate([x_samples_init,y_samples_init])

        res_grad_init = jnp.zeros(self.max_grad_iter)
        
        mt_init = jnp.zeros(2*self.nvar)
        vt_init = jnp.zeros(2*self.nvar)
        t_init = 0   
        eps = 1e-8

        b_eq_x, b_eq_y = self.compute_boundary_vec_single(x_init, vx_init, y_init, vy_init, x_fin, vx_fin, y_fin, vy_fin)

        s_bounds_x_init = jnp.zeros(( 1, 2*self.num  ))
        s_bounds_y_init = jnp.zeros(( 1, 2*self.num  ))
        lamda_x_init =  jnp.zeros(( 1, self.nvar  ))
        lamda_y_init =  jnp.zeros(( 1, self.nvar  ))

        def lax_grad(carry,idx):
            primal_sol_samples,res_grad,\
                 s_bounds_x,s_bounds_y,lamda_x,lamda_y,mt,vt,t = carry

            t += 1

            grad_vec = self.compute_grad(primal_sol_samples, lam, p1, p2, p3, p4)

            mt = beta_1*mt + (1-beta_1)*grad_vec
            vt = beta_2*vt + (1-beta_2)*grad_vec**2
            mt_hat = mt/(1-beta_1**t)
            vt_hat = vt/(1-beta_2**t)

            primal_sol_samples = primal_sol_samples - eta*mt_hat/(jnp.sqrt(vt_hat)+eps)
            
            primal_sol_x_samples = primal_sol_samples[0: self.nvar] 
            primal_sol_y_samples = primal_sol_samples[self.nvar: 2*self.nvar] 
            
            # primal_sol_samples, x_samples, xdot_samples, xddot_samples, y_samples, \
            # ydot_samples, yddot_samples, s_bounds_x, s_bounds_y, lamda_x, lamda_y  \
            #     = self.compute_projection_single(b_eq_x, b_eq_y, primal_sol_x_samples, primal_sol_y_samples,
            #                                       x_max, x_min, y_max, y_min, s_bounds_x, s_bounds_y, lamda_x, lamda_y)

            A = jax.scipy.linalg.block_diag(self.A_eq_x, self.A_eq_y)
            b = jnp.hstack(( b_eq_x, b_eq_y  )).squeeze(axis=0)
            G = jax.scipy.linalg.block_diag(self.A_bounds_x, self.A_bounds_y)
            b_bounds_x = jnp.hstack(( x_max*jnp.ones(self.num  ), -x_min*jnp.ones(self.num  )     ))
            b_bounds_y = jnp.hstack(( y_max*jnp.ones(self.num  ), -y_min*jnp.ones(self.num  )     ))
            h = jnp.hstack(( b_bounds_x, b_bounds_y  ))

            primal_sol_samples = projection_polyhedron(primal_sol_samples,(A,b,G,h),check_feasible = False)

            cost_single = self.compute_cost_gd(primal_sol_samples, lam, p1, p2, p3, p4)
            
            res_grad = res_grad.at[idx].set(cost_single)

            return (primal_sol_samples,res_grad,\
                 s_bounds_x,s_bounds_y,lamda_x,lamda_y,mt,vt,t), (grad_vec)
        
        carry_init = (primal_sol_samples_init,res_grad_init,
                      s_bounds_x_init,s_bounds_y_init,lamda_x_init,lamda_y_init,mt_init,vt_init,t_init)
        carry_final,result = jax.lax.scan(lax_grad,carry_init,jnp.arange(self.max_grad_iter))

        primal_sol_samples,res_grad,\
                 s_bounds_x,s_bounds_y,lamda_x,lamda_y,mt,vt,t = carry_final      
          
        grad_vec = result[0]

        primal_sol_x_samples = primal_sol_samples[0: self.nvar] 
        primal_sol_y_samples = primal_sol_samples[self.nvar: 2*self.nvar] 

        return primal_sol_x_samples,primal_sol_y_samples,res_grad


    @partial(jit, static_argnums=(0,))	
    def custom_gd_unc(self, x_init, vx_init, y_init, vy_init, x_fin, vx_fin, y_fin, vy_fin, lam, p1, p2, p3, p4, eta, beta_1, beta_2, x, y,
                        x_max,y_max,x_min,y_min, cov):

        x_samples_init = jnp.linalg.inv(self.P_jax.T @ self.P_jax+0.001*jnp.identity(self.nvar)) @ self.P_jax.T @ x
        y_samples_init = jnp.linalg.inv(self.P_jax.T @ self.P_jax+0.001*jnp.identity(self.nvar)) @ self.P_jax.T @ y

        primal_sol_samples_init = jnp.concatenate([x_samples_init,y_samples_init])

        res_grad_init = jnp.zeros(self.max_grad_iter)
        
        mt_init = jnp.zeros(2*self.nvar)
        vt_init = jnp.zeros(2*self.nvar)
        t_init = 0   
        eps = 1e-8

        b_eq_x, b_eq_y = self.compute_boundary_vec_single(x_init, vx_init, y_init, vy_init, x_fin, vx_fin, y_fin, vy_fin)

        s_bounds_x_init = jnp.zeros(( 1, 2*self.num  ))
        s_bounds_y_init = jnp.zeros(( 1, 2*self.num  ))
        lamda_x_init =  jnp.zeros(( 1, self.nvar  ))
        lamda_y_init =  jnp.zeros(( 1, self.nvar  ))

        def lax_grad(carry,idx):
            primal_sol_samples,res_grad,\
                 s_bounds_x,s_bounds_y,lamda_x,lamda_y,mt,vt,t = carry

            t += 1

            primal_sol_x = primal_sol_samples[0: self.nvar]
            primal_sol_y = primal_sol_samples[self.nvar: 2*self.nvar]

            x = jnp.dot(self.P_jax, primal_sol_x)
            y = jnp.dot(self.P_jax, primal_sol_y)

            xdot = jnp.dot(self.Pdot_jax, primal_sol_x)
            ydot = jnp.dot(self.Pdot_jax, primal_sol_y)

            psi = jnp.arctan2(ydot, xdot) 

            grad_f = self.compute_stability_grad(lam, p1, p2, p3, p4, x, y, psi).squeeze(axis=-1)

            stability_cov = self.compute_stability_cov_batch(grad_f, cov)

            grad_vec = self.compute_grad_unc(primal_sol_samples, lam, p1, p2, p3, p4, stability_cov)

            mt = beta_1*mt + (1-beta_1)*grad_vec
            vt = beta_2*vt + (1-beta_2)*grad_vec**2
            mt_hat = mt/(1-beta_1**t)
            vt_hat = vt/(1-beta_2**t)

            primal_sol_samples = primal_sol_samples - eta*mt_hat/(jnp.sqrt(vt_hat)+eps)
            
            # primal_sol_x_samples = primal_sol_samples[0: self.nvar] 
            # primal_sol_y_samples = primal_sol_samples[self.nvar: 2*self.nvar] 
            
            # primal_sol_samples, x_samples, xdot_samples, xddot_samples, y_samples, \
            # ydot_samples, yddot_samples, s_bounds_x, s_bounds_y, lamda_x, lamda_y  \
            #     = self.compute_projection_single(b_eq_x, b_eq_y, primal_sol_x_samples, primal_sol_y_samples,
            #                                       x_max, x_min, y_max, y_min, s_bounds_x, s_bounds_y, lamda_x, lamda_y)

            A = jax.scipy.linalg.block_diag(self.A_eq_x, self.A_eq_y)
            b = jnp.hstack(( b_eq_x, b_eq_y  )).squeeze(axis=0)
            G = jax.scipy.linalg.block_diag(self.A_bounds_x, self.A_bounds_y)
            b_bounds_x = jnp.hstack(( x_max*jnp.ones(self.num  ), -x_min*jnp.ones(self.num  )     ))
            b_bounds_y = jnp.hstack(( y_max*jnp.ones(self.num  ), -y_min*jnp.ones(self.num  )     ))
            h = jnp.hstack(( b_bounds_x, b_bounds_y  ))

            primal_sol_samples = projection_polyhedron(primal_sol_samples,(A,b,G,h),check_feasible = False)

            cost_single = self.compute_cost_gd_unc(primal_sol_samples, lam, p1, p2, p3, p4, stability_cov)
            
            res_grad = res_grad.at[idx].set(cost_single)

            return (primal_sol_samples,res_grad,\
                 s_bounds_x,s_bounds_y,lamda_x,lamda_y,mt,vt,t), (grad_vec)
        
        carry_init = (primal_sol_samples_init,res_grad_init,
                      s_bounds_x_init,s_bounds_y_init,lamda_x_init,lamda_y_init,mt_init,vt_init,t_init)
        carry_final,result = jax.lax.scan(lax_grad,carry_init,jnp.arange(self.max_grad_iter))

        primal_sol_samples,res_grad,\
                 s_bounds_x,s_bounds_y,lamda_x,lamda_y,mt,vt,t = carry_final      
          
        grad_vec = result[0]

        primal_sol_x_samples = primal_sol_samples[0: self.nvar] 
        primal_sol_y_samples = primal_sol_samples[self.nvar: 2*self.nvar] 


        # x_samples = jnp.dot(self.P_jax, primal_sol_x_samples.T).T 
        # y_samples = jnp.dot(self.P_jax, primal_sol_y_samples.T).T

        # xdot_samples = jnp.dot(self.Pdot_jax, primal_sol_x_samples.T).T
        # ydot_samples = jnp.dot(self.Pdot_jax, primal_sol_y_samples.T).T

        # psi_samples = jnp.arctan2(ydot_samples, xdot_samples) 
        # args = jnp.vstack(( x_samples, y_samples, psi_samples  )).T
    
        # min_sol = self.solve_nls_batch(args, lam, p1, p2, p3, p4)
        # min_tip_angle = self.compute_min_angle_batch(min_sol,x_samples,y_samples)

        # min_tip = jnp.sum(min_tip_angle)/len(min_tip_angle)

        return primal_sol_x_samples,primal_sol_y_samples,res_grad
   
   
   
		


	  

  

  


	
	 
		

	 
	 
	

  
  

		
	



		
	
	
  

  
  
  
	


	






