'''
User: lopes-g-m 
Author: Guilherme Lopes 
Year: 2024

Description: 
    This code trains a Physics-Informed Data-Driven model to 
    reproduce a 3D steady-state dispersion simulations of methane in the air using DeepXDE. 

Auxiliary code:
    functions.py, plot_functions.py

Input files:     
    dispersion3D.xlsx
    
Output files:
    Images,results.pkl,load_c.vtk,pred_c.vtk,test_c.vtk 

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
import io 

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,current_directory)

from functions import load_excel_data, load_geometry , normalize_array_0_1,\
     denormalize_array_0_1,calculate_time,split_train_test,   normalize_dataset_0_1      
     
from plot_functions import plot_data_distribution,av_rel_error,relative_error,plot_train_history, plot_loss_history,\
     plot_3_2d_domains,  plot_3_2d_domains_bump,plot_2d_domains, plot_2d_domains_bump,generate_3D_vtk
        
     
dataFilePath = current_directory + "/dataset/"
savePath = current_directory + "/results/"   

############################################################################### 
    
results_list = []         
    
# Load dispersion data
initial_time = time.time() 
dataFileName = "dispersion3D"
str_ident = dataFileName
dataFileNameExt = dataFileName +  '.xlsx'
dataFileFullPath =  dataFilePath + dataFileNameExt
[load_x, load_y , load_z , load_c ] = load_excel_data(dataFileFullPath)

# Test split fraction
test_frac = 0.2

# ANN 
activation = "swish" 
initializer = "Glorot uniform"  
layer_size= [3] + [20] * 10 + [5] 
Nboundary = 0 

# Training/Optimizer setup 
optimizer = "adam" 
Niter = 10000
Ndisplay = 10
learn_rate = 1e-3
steps_frequency = 1 
num_equations = 4 
Batch = 10000

###############################################################################

# Normalize data        
norm_load_x,norm_load_y,norm_load_z, norm_load_c= normalize_dataset_0_1(load_x,load_y, load_z , load_c)

# Split
train_x , train_y, train_z , norm_train_c, test_x , test_y , test_z,norm_test_c = split_train_test(load_x, load_y , load_z, norm_load_c , test_frac)

Nscattered = len(norm_train_c) 
Ntest = int(test_frac * len(norm_test_c))    
minX,minY,minZ,minC = load_x.min(0), load_y.min(0), load_z.min(0) , load_c.min(0)
maxX,maxY,maxZ,maxC = load_x.max(0), load_y.max(0), load_z.max(0) , load_c.max(0)

# Denormalize test
test_c = denormalize_array_0_1(norm_test_c, minC, maxC).reshape(-1, 1)

###############################################################################

# Parameters to be identified
C1 = dde.Variable(0.0)
C2 = dde.Variable(0.0)

Lx_min, Lx_max =  int(minX), int(maxX) 
Ly_min, Ly_max =   int(minY), int(maxY) 
Lz_min, Lz_max =   int(minZ), int(maxZ)   
 
# Define Spatial domain plot_data
# Cuboid
space_domain = dde.geometry.geometry_3d.Cuboid([Lx_min, Ly_min, Lz_min], [Lx_max, Ly_max,Lz_max])

train_xyz = np.hstack((train_x, train_y, train_z))
test_xyz = np.hstack((test_x, test_y,test_z))     
observe_c = dde.icbc.PointSetBC(train_xyz, norm_train_c, component=4)      

# Define system of PDEs     
def PDE(x, y):
    
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    p = y[:, 3:4]
    c = y[:, 4:5]

    du_x = dde.grad.jacobian(y, x, i=0, j=0) # du/dx
    du_y = dde.grad.jacobian(y, x, i=0, j=1) # du/dy
    du_z = dde.grad.jacobian(y, x, i=0, j=2) # du/dy        
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dv_z = dde.grad.jacobian(y, x, i=1, j=2)        
    dw_x = dde.grad.jacobian(y, x, i=2, j=0)
    dw_y = dde.grad.jacobian(y, x, i=2, j=1)
    dw_z = dde.grad.jacobian(y, x, i=2, j=2)        
    dp_x = dde.grad.jacobian(y, x, i=3, j=0)
    dp_y = dde.grad.jacobian(y, x, i=3, j=1)
    dp_z = dde.grad.jacobian(y, x, i=3, j=2)        
    dc_x = dde.grad.jacobian(y, x, i=4, j=0) # dc/dx
    dc_y = dde.grad.jacobian(y, x, i=4, j=1)
    dc_z = dde.grad.jacobian(y, x, i=4, j=2)

    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    du_zz = dde.grad.hessian(y, x, component=0, i=2, j=2)        
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    dv_zz = dde.grad.hessian(y, x, component=1, i=2, j=2)        
    dw_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)
    dw_yy = dde.grad.hessian(y, x, component=2, i=1, j=1)
    dw_zz = dde.grad.hessian(y, x, component=2, i=2, j=2)        
    dc_xx = dde.grad.hessian(y, x,  component=4, i=0, j=0)
    dc_yy = dde.grad.hessian(y, x,  component=4, i=1, j=1)
    dc_zz = dde.grad.hessian(y, x,  component=4, i=2, j=2)

    # Continuity equation
    continuity = du_x + dv_y + dw_z

    # Navier Stokes Equation
    # Steady state, propierties contants, newton fluid, laminar
    x_momentum =  (u * du_x + v * du_y + w * du_z) + dp_x - C1 * (du_xx + du_yy + du_zz)        
    y_momentum =  (u * dv_x + v * dv_y + w * dv_z) + dp_y - C1 * (dv_xx + dv_yy + dv_zz)         
    z_momentum =  (u * dw_x + v * dw_y + w * dw_z) + dp_z - C1 * (dw_xx + dw_yy + dw_zz) 

    # Differential Mass Transfer Equation
    # Incompressible and properties constant
    mass_transf =  (u * dc_x + v * dc_y + w * dc_z) - C2 * (dc_xx + dc_yy + dc_zz)

    return [ continuity,  x_momentum, y_momentum, z_momentum , mass_transf ]

results = [] # Store results

# Training datasets and Loss
data = dde.data.pde.PDE(space_domain,
                          PDE,
                          [observe_c],
                          num_domain=Nscattered, # Training points
                          num_boundary=Nboundary,
                          anchors=train_xyz,  # Training points 
                          solution=None,
                          num_test=Ntest, # Testing points
                          auxiliary_var_function=None)


# Callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([C1, C2], period=100, filename=fnamevar)
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

# callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([C1, C2], period=100, filename=fnamevar)

train_initial_time = time.time()
model.compile(optimizer, lr=learn_rate, external_trainable_variables=[C1, C2])

# save model
modelSavePath = savePath + "model.ckpt"
checker = dde.callbacks.ModelCheckpoint(modelSavePath  , save_better_only=True, period=Niter)

loss_history, train_state = model.train(iterations=Niter, batch_size=Batch, callbacks=[checker,variable], display_every=Ndisplay, disregard_previous_best=True)
train_final_time = time.time()
test_initial_time = time.time()

# Visualize data in 3D  
test_x, test_y , test_z = load_x, load_y , load_z       

# Model predictions
pred_xyz = np.hstack((test_x, test_y , test_z)) 
norm_pred_uvwpc = model.predict(pred_xyz)
test_final_time = time.time()
pred_x, pred_y ,pred_z = pred_xyz[:, 0], pred_xyz[:, 1] ,pred_xyz[:, 2] 
norm_pred_u, norm_pred_v,norm_pred_w, norm_pred_p , norm_pred_c = norm_pred_uvwpc[:, 0], norm_pred_uvwpc[:, 1], norm_pred_uvwpc[:, 2],norm_pred_uvwpc[:,3] ,norm_pred_uvwpc[:,4] 

norm_pred_u, norm_pred_v,norm_pred_w, norm_pred_p , norm_pred_c = norm_pred_u.reshape(-1, 1), norm_pred_v.reshape(-1, 1), norm_pred_w.reshape(-1, 1), norm_pred_p.reshape(-1, 1) , norm_pred_c.reshape(-1, 1)
# Denormalize prediction

pred_c = denormalize_array_0_1(norm_pred_c, minC, maxC).reshape(-1, 1)
    
train_delta_t_ms,train_delta_t_s,train_delta_t_m, train_delta_t_h =  calculate_time (train_initial_time,train_final_time) # calculate train time
test_delta_t_ms,test_delta_t_s,test_delta_t_m, test_delta_t_h =  calculate_time (test_initial_time,test_final_time) # calculate prediction time
    
# Plot train 
plot_train_history(loss_history, steps_frequency , savePath)

generate_3D_vtk(test_x,test_y,test_z,load_c,'load_c',savePath) 
generate_3D_vtk(test_x,test_y,test_z,load_c,'test_c',savePath)    
generate_3D_vtk(test_x,test_y,test_z,pred_c,'pred_c',savePath)  
  
# Save results in a dictionary
results = {
    "layer_size": layer_size,
    "loss_history": loss_history,
    "train_state": train_state,
    "test_x": test_x,
    "test_y": test_y,
    "test_z": test_z,
    "test_c": test_c,
    "pred_c": pred_c,
    "train_delta_t_ms": train_delta_t_ms,
    "train_delta_t_s": train_delta_t_s,
    "train_delta_t_m": train_delta_t_m,
    "train_delta_t_h": train_delta_t_h,
    "test_delta_t_ms": test_delta_t_ms,
    "test_delta_t_s": test_delta_t_s,
    "test_delta_t_m": test_delta_t_m,
    "test_delta_t_h": test_delta_t_h,
    "dataFileName": dataFileName,
    "savePath": savePath,
    "load_x": load_x,
    "load_y": load_y,
    "load_c": load_c,
    "test_frac": test_frac,
    "activation": activation,
    "initializer": initializer,
    "Nboundary": Nboundary,
    "norm_pred_c": norm_pred_c,
    "norm_pred_u": norm_pred_u,
    "norm_pred_v": norm_pred_v,
    "norm_pred_w": norm_pred_w,
    "norm_pred_p": norm_pred_p,
    "optimizer": optimizer,
    "Niter": Niter,
    "Ndisplay": Ndisplay,
    "learn_rate": learn_rate,
    "steps_frequency": steps_frequency,
    "num_equations": num_equations,
    "Batch": Batch   }

results_list.append(results)

    # Save results_list as a pickle file
with open( savePath +  'results.pkl', 'wb') as file: 
  pickle.dump(results_list, file) 
