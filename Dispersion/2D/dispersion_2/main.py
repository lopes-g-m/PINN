'''
User: lopes-g-m 
Author: Guilherme Lopes 
Year: 2024

Description: 
    This code trains a Physics-Informed Data-Driven model to 
    reproduce a 2D steady-state dispersion simulations of methane in the air using DeepXDE. 

Auxiliary code:
    functions.py, plot_functions.py

Input files:     
    dispersion_2.xlsx,domain_bump.xlsx
    
Output files:
    Images,results.pkl

Libraries:
    !pip install deepxde 
    !pip install tensorflow 
    !pip install matplotlib 
'''

import deepxde as dde 
import tensorflow as tf  
import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat  
import re 
import pandas as pd 
import matplotlib.tri as tri 
import matplotlib.pyplot as plt 
import time 
import sys 
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,current_directory)

from functions import load_excel_data, load_geometry , normalize_array_0_1,\
     denormalize_array_0_1,calculate_time,split_train_test, normalize_dataset_0_1
     
from plot_functions import av_rel_error,relative_error, plot_loss_history,plot_train_history,plot_3_2d_domains,\
     plot_3_2d_domains_bump,plot_2d_domains,plot_2d_domains_bump, sinal_relative_error,plot_percentage_histogram,\
     percentage_between,calculate_percentage_in_ranges
        
###############################################################################

dataFilePath = current_directory + "/dataset/"
savePath = current_directory + "/results/" 
dataFileName = 'dispersion_2' 
Re = 5.434
Pe =  3.922
results_list = []  

############################################################################### 
    
# Load dispersion data
initial_time = time.time()
str_ident = dataFileName
dataFileNameExt = dataFileName +  '.xlsx'
dataFileFullPath =  dataFilePath + dataFileNameExt
[load_x, load_y , load_u, load_v, load_p, load_c ] = load_excel_data(dataFileFullPath) 
x_lim_inf, y_lim_inf , x_lim_sup, y_lim_sup = load_geometry(dataFilePath + 'domain_bump.xlsx')  

# Test split fraction
test_frac = 0.2
# ANN hyperparameters 
layer_size = [2] + [50] * 10 + [4]              
activation = "swish" 
initializer = "Glorot uniform"  
 
# Training/Optimizer setup 
optimizer = "adam" 
Niter = 10000
Ndisplay = 1
learn_rate = 1e-3
steps_frequency = 1 

###############################################################################

data_sets = [(load_x, load_y, load_u),(load_x, load_y, load_v),(load_x, load_y, load_p),(load_x, load_y, load_c)]
titles = ['$U_{CFD}$', '$V_{CFD}$', '$P_{CFD}$', '$C_{CFD}$']    
plot_2d_domains_bump(data_sets, titles, 'x', 'y', str_ident, savePath, x_lim_inf, y_lim_inf)
    
# Normalize data
norm_load_x,norm_load_y,norm_load_u,norm_load_v,norm_load_p,norm_load_c = normalize_dataset_0_1(load_x, load_y , load_u, load_v, load_p, load_c)

# Split
train_x, train_y, norm_train_u, norm_train_v, norm_train_p, norm_train_c, test_x, test_y, norm_test_u,norm_test_v,norm_test_p,norm_test_c= split_train_test(load_x, load_y , norm_load_u, norm_load_v, norm_load_p, norm_load_c, test_frac)

Nscattered = len(norm_train_c)
Ntest = len(norm_test_c)
Nboundary = 0 

minX,minY,minU,minV,minP,minW = load_x.min(0), load_y.min(0), load_u.min(0), load_v.min(0), load_p.min(0), load_c.min(0)
maxX,maxY,maxU,maxV,maxP,maxW = load_x.max(0), load_y.max(0), load_u.max(0), load_v.max(0), load_p.max(0), load_c.max(0)

# Denormalize test
test_u = denormalize_array_0_1(norm_test_u, minU, maxU).reshape(-1, 1)
test_v = denormalize_array_0_1(norm_test_v, minV, maxV).reshape(-1, 1)
test_p = denormalize_array_0_1(norm_test_p, minP, maxP).reshape(-1, 1)
test_c = denormalize_array_0_1(norm_test_c, minW, maxW).reshape(-1, 1)
  
###############################################################################

# Parameters to be identified
C1 = dde.Variable(0.0)
C2 = dde.Variable(0.0)
Lx_min, Lx_max =  0, 300  
Ly_min, Ly_max =   0, 100 

# Define Spatial domain plot_data
# Rectangular
space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
train_xy = np.hstack((train_x, train_y))
test_xy = np.hstack((test_x, test_y))

# Supervised learning
observe_u = dde.icbc.PointSetBC(train_xy, norm_train_u, component=0)
observe_v = dde.icbc.PointSetBC(train_xy, norm_train_v, component=1)
observe_p = dde.icbc.PointSetBC(train_xy, norm_train_p, component=2)
observe_c = dde.icbc.PointSetBC(train_xy, norm_train_c, component=3)

# Define system of PDEs 

def PDE(x, y):
    
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    c = y[:, 3:4]
    du_x = dde.grad.jacobian(y, x, i=0, j=0) # du/dx
    du_y = dde.grad.jacobian(y, x, i=0, j=1) # du/dy    
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)    
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)    
    dc_x = dde.grad.jacobian(y, x, i=3, j=0) # dc/dx
    dc_y = dde.grad.jacobian(y, x, i=3, j=1)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0) # d2u/dx2
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)    
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)    
    dc_xx = dde.grad.hessian(y, x,  component=3, i=0, j=0)
    dc_yy = dde.grad.hessian(y, x,  component=3, i=1, j=1)
    # Continuity equation
    continuity = du_x + dv_y

    # Navier Stokes Equation
    # Steady state, propierties contants, newton fluid, laminar
    x_momentum =  (u * du_x + v * du_y) + dp_x - C1 * (du_xx + du_yy)
    y_momentum =  (u * dv_x + v * dv_y) + dp_y - C1 * (dv_xx + dv_yy) 

    # Differential Mass Transfer Equation
    # incompressible and properties constant
    mass_transf =  (u * dc_x + v * dc_y) - C2 * (dc_xx + dc_yy)

    return [ continuity,  x_momentum, y_momentum, mass_transf ]

results = [] # Store results

# Training datasets and Loss
data = dde.data.pde.PDE(space_domain,
                          PDE,
                          [ observe_u, observe_v, observe_p, observe_c],
                          num_domain=Nscattered, 
                          num_boundary=Nboundary,
                          anchors=train_xy,  
                          solution=None,
                          num_test=Ntest, 
                          auxiliary_var_function=None)


# Callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([C1, C2], period=100, filename=fnamevar)
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
train_initial_time = time.time()
model.compile(optimizer, lr=learn_rate, external_trainable_variables=[C1, C2])

# Save model
modelSavePath = savePath + "model.ckpt"
checker = dde.callbacks.ModelCheckpoint(modelSavePath  , save_better_only=True, period=Niter)

loss_history, train_state = model.train(iterations=Niter, callbacks=[checker,variable], display_every=Ndisplay, disregard_previous_best=True)
train_final_time = time.time()
test_initial_time = time.time() 

# Model predictions
pred_xy = np.hstack((test_x, test_y)) 
norm_pred_uvpw = model.predict(pred_xy)
test_final_time = time.time()
pred_x, pred_y = pred_xy[:, 0], pred_xy[:, 1]
norm_pred_u, norm_pred_v, norm_pred_p , norm_pred_c = norm_pred_uvpw[:, 0], norm_pred_uvpw[:, 1], norm_pred_uvpw[:, 2],norm_pred_uvpw[:,3] 

# Denormalize prediction
pred_u = denormalize_array_0_1(norm_pred_u, minU, maxU).reshape(-1, 1)
pred_v = denormalize_array_0_1(norm_pred_v, minV, maxV).reshape(-1, 1)
pred_p = denormalize_array_0_1(norm_pred_p, minP, maxP).reshape(-1, 1)
pred_c = denormalize_array_0_1(norm_pred_c, minW, maxW).reshape(-1, 1)

# Calculate NMAPE
av_rel_error_u,sre_u = av_rel_error(pred_u,test_u),sinal_relative_error(pred_u,test_u)
av_rel_error_v,sre_v = av_rel_error(pred_v,test_v),sinal_relative_error(pred_v,test_v)
av_rel_error_p,sre_p = av_rel_error(pred_p,test_p),sinal_relative_error(pred_p,test_p)
av_rel_error_c,sre_c = av_rel_error(pred_c,test_c),sinal_relative_error(pred_c,test_c)

train_delta_t_ms,train_delta_t_s,train_delta_t_m, train_delta_t_h =  calculate_time (train_initial_time,train_final_time) # calculate train time
test_delta_t_ms,test_delta_t_s,test_delta_t_m, test_delta_t_h =  calculate_time (test_initial_time,test_final_time) # calculate prediction time

# Plot train 
plot_train_history(loss_history, steps_frequency , savePath)
   
# Plot comparison
ob_x_list = [test_x, test_x, test_x, test_x] 
ob_y_list = [test_y, test_y, test_y, test_y]
ob_u_list = [test_u, test_v, test_p, test_c] 
ob_v_list = [pred_u, pred_v, pred_p, pred_c]
sre_list = [sre_u,  sre_v,  sre_p,  sre_c]

titles = ['$U$', '$V$', '$P$', '$C$']    
plot_3_2d_domains_bump(ob_x_list, ob_y_list, ob_u_list, ob_v_list, titles, "x", "y", " " , str_ident, x_lim_inf, y_lim_inf, savePath)

       
# Results dictionary
results = {
    "layer_size": layer_size,
    "loss_history": loss_history,
    "train_state": train_state,
    "test_x": test_x,
    "test_y": test_y,
    "pred_u": pred_u,
    "pred_v": pred_v,
    "pred_p": pred_p,
    "pred_c": pred_c,
    "av_rel_error_u": av_rel_error_u,
    "av_rel_error_v": av_rel_error_v,
    "av_rel_error_p": av_rel_error_p,
    "av_rel_error_c": av_rel_error_c,
    "train_delta_t_ms": train_delta_t_ms,
    "train_delta_t_s": train_delta_t_s,
    "train_delta_t_m": train_delta_t_m,
    "train_delta_t_h": train_delta_t_h,
    "test_delta_t_ms": test_delta_t_ms,
    "test_delta_t_s": test_delta_t_s,
    "test_delta_t_m": test_delta_t_m,
    "test_delta_t_h": test_delta_t_h,
    "Re": Re,
    "Pe": Pe,
    "dataFileName": dataFileName,
    "savePath": savePath,
    "load_x": load_x, 
    "load_y": load_y,
    "load_u": load_u,
    "load_v": load_v,
    "load_p": load_p,
    "load_c": load_c,
    "test_frac": test_frac,
    "activation": activation,
    "initializer": initializer,
    "layer_size": layer_size,
    "Nboundary": Nboundary,
    "optimizer": optimizer,
    "Niter": Niter,
    "Ndisplay": Ndisplay,
    "learn_rate": learn_rate,
    "steps_frequency": steps_frequency,
    "test_u": test_u,
    "test_v": test_v,
    "test_p": test_p,
    "test_c": test_c,
    "train_x": train_x,
    "train_y": train_y,
    "norm_train_u": norm_train_u,
    "norm_train_v": norm_train_v,
    "norm_train_p": norm_train_p,
    "norm_train_c": norm_train_c }     

results_list.append(results)

    # Save results_list as a pickle file
with open( savePath +  'results.pkl', 'wb') as file: 
  pickle.dump(results_list, file)  
