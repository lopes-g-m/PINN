import tensorflow.compat.v1 as tf 
import numpy as np
import scipy.io
import time
import sys 
import os
import pickle
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import vtk 

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,current_directory)
tf.disable_v2_behavior() 

from utilities import neural_net, Navier_Stokes_3D, steady_state_Navier_Stokes_3D, \
                      tf_session, mean_squared_error, model                   
                      
from functions import load_excel_data, percentage_greater_than,load_geometry , normalize_array_0_1,\
     denormalize_array_0_1,calculate_time,split_train_test, calc_conc, get_sample,\
     normalize_dataset_0_1 ,fraction_in_range
     
from plot_functions import plot_data_distribution,av_rel_error,plot_gov_eq_loss,\
     plot_loss_history,plot_sup_loss, plot_gov_4_eq_loss,plot_train_history,plot_3_2d_domains,\
     plot_3_2d_domains_bump,plot_2d_domains, plot_2d_domains_bump,plot_2_loss_history,\
     plot_error_dist,relative_error,generate_3D_vtk,plot_loss,generate_unstructured_3D_vtk,\
     interpolate_to_structured_grid
     
from pre_process import import_multiple_flacs_steady_simulations,pre_process_data,organizeSimulationData,\
     prepare_data,remove_element_mesh,flatten_arrays_dic,flatten_arrays,\
     get_simulation_data,import_flacs_steady_simulation,import_cgns_flacs

if __name__ == "__main__":         
    savePath = current_directory + "/results/"     
    dataFilePath = "/home/l4r1s4/Guilherme/dataset/steady_dispersion/" 
    batch_size = 10000    
    layers = [3] + 10 * [50] + [5]   
    test_frac = 0.2
    Niter = 10000  
    Pe = 1
    Re = 1     
    # Adam optimizer 
    lr = 1e-3  
    steps_frequency=1    
    lim_inf_ch4 = 0.05
    lim_sup_ch4 = 0.15     
    Nx = 20
    Ny = 60
    Nz = 100   
    ############################################################################### 
    # Import dataset 
    # Dispersion 4 
    path4 = current_directory + "/dataset/"
    file4 = "010300"    
    simulationDataCGNS_4 = import_cgns_flacs(path4, file4)
    paths = [path4]       
    flowSolutions = ["FlowSolutionCC#0000003949"]          
    filenames = [file4]     
    arrLeakRate = [1]       
    MultipleSimulationsData = import_multiple_flacs_steady_simulations(paths, filenames,flowSolutions)
    load_data_x, load_data_y, load_data_z, load_data_c, load_data_u, load_data_v, load_data_w, load_data_p = organizeSimulationData(MultipleSimulationsData)
    data_x, data_y, data_z, data_m, data_c, data_u, data_v, data_w, data_p = pre_process_data(MultipleSimulationsData, arrLeakRate)
    load_x, load_y , load_z , load_c = data_x, data_y, data_z, data_c
    ###############################################################################   

    minX,maxX = min(load_x),max(load_x)
    minY,maxY = min(load_y),max(load_y)
    minZ,maxZ = min(load_z),max(load_z)
    
    deltaX = (maxX-minX)/Nx
    deltaY = (maxY-minY)/Ny
    deltaZ = (maxZ-minZ)/Nz
    
    Vcell = deltaX*deltaY*deltaZ

    minC,maxC = min(load_c),max(load_c)
    
    norm_load_c = normalize_array_0_1(load_c)
    
    data_x_int, data_y_int, data_z_int, data_c_int = interpolate_to_structured_grid(load_x, load_y , load_z , load_c , Nx,Ny,Nz, method='linear')
    val_x,val_y,val_z,val_c = data_x_int, data_y_int, data_z_int, data_c_int  

    initial_time = time.time()       
    train_x, test_x, train_y, test_y, train_z, test_z,  train_c, test_c = train_test_split(data_x, data_y, data_z, norm_load_c, test_size=test_frac, random_state=42)
    
    x_eqns, y_eqns, z_eqns = train_x, train_y, train_z    
    
    model = model(train_x, train_y, train_z, train_c,
                           x_eqns, y_eqns, z_eqns,
                           layers, batch_size,
                           Pec = Pe, Rey = Re)   
       
    
    train_initial_time = time.time()    
    loss_history = model.train(Niter , learning_rate=lr) 
    plot_loss(loss_history, savePath, title='loss_history.png')   
    train_final_time = time.time()
    test_initial_time = time.time()   
    norm_pred_c, pred_u, pred_v, pred_w, pred_p = model.predict( val_x,val_y,val_z)     
    pred_c = denormalize_array_0_1(norm_pred_c, minC, maxC)        
    test_final_time = time.time()      
    re_c = relative_error(pred_c,val_c)
    av_re_c = av_rel_error(pred_c,val_c)
    
    # Cloud Size        
    frac_val_c = fraction_in_range(val_c,lim_inf_ch4,lim_sup_ch4)    
    frac_load_c = fraction_in_range(load_c,lim_inf_ch4,lim_sup_ch4)
    frac_pred_c = fraction_in_range(pred_c,lim_inf_ch4,lim_sup_ch4)     
    V_val_c = frac_val_c * Vcell * len(val_c) 
    V_pred_c = frac_pred_c * Vcell * len(pred_c) 
    
    generate_3D_vtk(val_x,val_y,val_z, pred_c, 'title', savePath+'model_prediction/pred.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk(val_x,val_y,val_z, val_c, 'title', savePath+'simulation/dispersion.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk(val_x,val_y,val_z, re_c, 'title', savePath+'error/error.vtk', Nx, Ny, Nz, num_columns=6) 
          
    # Calculate time     
    train_delta_t_ms,train_delta_t_s,train_delta_t_m, train_delta_t_h =  calculate_time (train_initial_time,train_final_time) # calculate train time
    test_delta_t_ms,test_delta_t_s,test_delta_t_m, test_delta_t_h =  calculate_time (test_initial_time,test_final_time) # calculate prediction time
           