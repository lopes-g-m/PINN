'''
User: lopes-g-m 
Author: Guilherme Lopes 
Year: 2024

Description: 
    This code trains a Physics-Informed Data-Driven model to 
    reproduce a Lid Driven Cavity problem using DeepXDE. 

Auxiliary code:
    functions.py, plot_functions.py

Input files:     
    cavity_Re_100.xlsx
    
Output files:
    Images,results.pkl

Libraries:
    !pip install deepxde 
    !pip install tensorflow 
    !pip install matplotlib 
'''

import deepxde as dde  
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
import pickle 

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,current_directory)

from functions import load_excel_data, percentage_greater_than,load_geometry , normalize_array_0_1,\
     denormalize_array_0_1,calculate_time,split_train_test, normalize_dataset_0_1
     
from plot_functions import plot_data_distribution,av_rel_error,relative_error,plot_2d_domains,\
     plot_loss_history,plot_train_history,plot_3_2d_domains, plot_3_2d_domains_bump, plot_2d_domains_bump, sinal_relative_error
          
dataFilePath = current_directory + "/dataset/"
savePath = current_directory + "/results/" 

results_list = []  

dataFileName = 'cavity_Re_100'
Re = 100

str_ident = dataFileName
dataFileNameExt = dataFileName +  '.xlsx'
dataFileFullPath =  dataFilePath + dataFileNameExt 
load_x, load_y, load_u, load_v, load_p = load_excel_data(dataFileFullPath) 

data_sets = [(load_x, load_y, load_u),(load_x, load_y, load_v),(load_x, load_y, load_p)]
titles = ['$U_{CFD}$', '$V_{CFD}$', '$P_{CFD}$']    
    
plot_2d_domains(data_sets, titles, 'x', 'y', str_ident, savePath)    

# Test split fraction 
test_frac = 0.4 

minX, minY,minU,minV,minP = load_x.min(0), load_y.min(0), load_u.min(0), load_v.min(0), load_p.min(0) 
maxX,maxY,maxU,maxV,maxP  = load_x.max(0), load_y.max(0), load_u.max(0), load_v.max(0), load_p.max(0)

# Normalize data set 
norm_load_u = normalize_array_0_1(load_u) 
norm_load_v = normalize_array_0_1(load_v) 
norm_load_p = normalize_array_0_1(load_p) 

# Split 
train_x, train_y, norm_train_u, norm_train_v, norm_train_p, test_x,test_y, norm_test_u,norm_test_v,norm_test_p= split_train_test(load_x, load_y , norm_load_u, norm_load_v, norm_load_p, test_frac)

# Denormalize test
test_u = denormalize_array_0_1(norm_test_u, minU, maxU).reshape(-1, 1)
test_v = denormalize_array_0_1(norm_test_v, minV, maxV).reshape(-1, 1)
test_p = denormalize_array_0_1(norm_test_p, minP, maxP).reshape(-1, 1)

###############################################################################    

Niter = 10000
Ndisplay = 10
steps_frequency = 10
activation = "swish"
layer_sizes = [2] + [100] * 10 + [3] 
initializer = "Glorot uniform" 
    
learn_rate = 1e-3
Optimizer = "adam"

# Parameters to be identified 
C1 = dde.Variable(0.0)
C2 = dde.Variable(0.0)

Lx_min, Lx_max = 0, 10
Ly_min, Ly_max =  0, 10

# Define Spatial domain plot_data
# Rectangular
space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
train_xy = np.hstack((train_x, train_y))

test_xy = np.hstack((test_x, test_y))

# Dirichlet boundary condition for a set of points 
observe_u = dde.icbc.PointSetBC(train_xy, norm_train_u, component=0)
observe_v = dde.icbc.PointSetBC(train_xy, norm_train_v, component=1)
observe_p = dde.icbc.PointSetBC(train_xy, norm_train_p, component=2)


# Define Navier Stokes Equations 
def Navier_Stokes_Equation(x, y):
    u = y[:, 0:1] 
    v = y[:, 1:2]
    p = y[:, 2:3]
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    continuity = du_x + dv_y
    x_momentum = C1 * (u * du_x + v * du_y) + dp_x - C2 * (du_xx + du_yy)
    y_momentum = C1 * (u * dv_x + v * dv_y) + dp_y - C2 * (dv_xx + dv_yy)
    return [continuity, x_momentum, y_momentum]

################################################################################   

# Store results
results = []
 
layer_size = layer_sizes
domain = len(norm_train_u) 
boundary = 0 
    
# Training datasets and Loss
data = dde.data.pde.PDE(space_domain, 
                          Navier_Stokes_Equation, 
                          [observe_u, observe_v, observe_p],
                          num_domain=domain, # Training points
                          num_boundary=boundary, # Training points
                          anchors=train_xy, 
                          solution=None, 
                          num_test=None, # Testing points
                          auxiliary_var_function=None)      

net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

# Callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([C1, C2], period=100, filename=fnamevar)

train_initial_time = time.time()

# Compile, train and save model 
model.compile(Optimizer, lr=learn_rate, external_trainable_variables=[C1, C2])
loss_history, train_state = model.train(
    iterations=Niter, callbacks=[variable], display_every=Ndisplay, disregard_previous_best=True)

train_final_time = time.time()
test_initial_time = time.time()    

str_ident = dataFileName + " " + str(domain) + str(boundary) 

model.save(save_path = savePath) 
f = model.predict(test_xy, operator=Navier_Stokes_Equation)
print("Mean residual:", np.mean(np.absolute(f))) 

# Model predictions
pred_xy = test_xy
norm_pred_uvp = model.predict(test_xy) 
pred_x, pred_y = pred_xy[:, 0], pred_xy[:, 1]
norm_pred_u, norm_pred_v, norm_pred_p = norm_pred_uvp[:, 0], norm_pred_uvp[:, 1], norm_pred_uvp[:, 2]

# Denormalize prediction 
pred_u = denormalize_array_0_1(norm_pred_u, minU, maxU).reshape(-1, 1)
pred_v = denormalize_array_0_1(norm_pred_v, minV, maxV).reshape(-1, 1)
pred_p = denormalize_array_0_1(norm_pred_p, minP, maxP).reshape(-1, 1)

# Calculate NMRAE
av_rel_error_u = av_rel_error(pred_u,test_u) 
av_rel_error_v = av_rel_error(pred_v,test_v) 
av_rel_error_p = av_rel_error(pred_p,test_p)    

sre_u, sre_v, sre_p = sinal_relative_error(test_u, pred_u),sinal_relative_error(test_v, pred_v),sinal_relative_error(test_p, pred_p)
 
# Plot train 
plot_train_history(loss_history, steps_frequency , savePath)
   
# Plot comparison
ob_x_list = [test_x, test_x, test_x] 
ob_y_list = [test_y, test_y, test_y]
ob_u_list = [test_u, test_v, test_p] 
ob_v_list = [pred_u, pred_v, pred_p]
sre_list = [sre_u, sre_v, sre_p]    
titles = ['$U$', '$V$', '$P$']

plot_3_2d_domains(ob_x_list, ob_y_list, ob_u_list, ob_v_list, titles, 'x', 'y', ' ', str_ident, savePath)    
  
results.append({
    "layer_size": layer_size,
    "loss_history": loss_history,
    "train_state": train_state,
    "av_rel_error_u": av_rel_error_u,
    "av_rel_error_v": av_rel_error_v,
    "av_rel_error_p": av_rel_error_p,
    "ob_x_list": ob_x_list,
    "ob_y_list": ob_y_list,
    "ob_u_list": ob_u_list, 
    "ob_v_list": ob_v_list,
    "sre_list": sre_list,
    "train_x":train_x,
    "train_y":train_y,
    "pred_x":pred_x,
    "pred_y":pred_y,
    "norm_train_u":norm_train_u,
    "norm_train_v": norm_train_v,
    "norm_train_p": norm_train_p,
    "test_x":test_x,
    "test_y":test_y,
    "test_u":test_u,
    "test_v":test_v,
    "test_p":test_p,
    "pred_u": pred_u,
    "pred_v": pred_v,
    "pred_p": pred_p,
    "train_initial_time":  train_initial_time,
    "train_final_time": train_final_time,
    "Niter": Niter,
    "Ndisplay": Ndisplay,
    "steps_frequency": steps_frequency,
    "activation": activation,
    "initializer": initializer,
    "learn_rate": learn_rate,
    "Optimizer": Optimizer,
    "test_frac":test_frac
}) 

results_list.append(results)

with open( savePath +  'results.pkl', 'wb') as file: 
  pickle.dump(results_list, file) 
