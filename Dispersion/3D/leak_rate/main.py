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
     normalize_dataset_0_1,fraction_in_range
     
from plot_functions import plot_data_distribution,av_rel_error,plot_gov_eq_loss,\
     plot_loss_history,plot_sup_loss, plot_gov_4_eq_loss,plot_train_history,plot_3_2d_domains,\
     plot_3_2d_domains_bump,plot_2d_domains, plot_2d_domains_bump,plot_2_loss_history,\
     plot_error_dist,relative_error,generate_3D_vtk,plot_loss,generate_unstructured_3D_vtk,\
     interpolate_to_structured_grid
     
from pre_process import import_multiple_flacs_steady_simulations,pre_process_data,organizeSimulationData,\
     prepare_data,remove_element_mesh,flatten_arrays_dic,flatten_arrays,\
     get_simulation_data,import_flacs_steady_simulation

if __name__ == "__main__":     
    
    savePath = current_directory + "/results/"     
    dataFilePath = "/home/l4r1s4/Guilherme/dataset/steady_dispersion/" 
    batch_size = 10000    
    
    layers = [4] + 10* [50] + [5]   
    test_frac = 0.2
    Niter = 1
    
    Pe = 1
    Re = 1     
    # Adam optimizer 
    lr = 1e-3  
    steps_frequency=1
    
    lim_inf_ch4 = 0.05
    lim_sup_ch4 = 0.15    
    
    Nx = 40
    Ny = 80
    Nz = 50
    
    paths = [current_directory + "/dataset/dispersion_35_steady/",
             current_directory + "/dataset/dispersion_36_steady/",
             current_directory + "/dataset/dispersion_37_steady/",
             current_directory + "/dataset/dispersion_38_steady/"]        
    
    flowSolutions = ["FlowSolutionCC#0000000251", 
                     "FlowSolutionCC#0000000257",
                     "FlowSolutionCC#0000000263",
                     "FlowSolutionCC#0000000259"]      
    
    filenames = ["010100", "010101", "010102", "010103"]     
    titles = [f"Simulation {i}" for i in range(1, 5)]  
    arrLeakRate = [1,2,3,4]   
    
    MultipleSimulationsData = import_multiple_flacs_steady_simulations(paths, filenames,flowSolutions)
    load_data_x, load_data_y, load_data_z, load_data_c, load_data_u, load_data_v, load_data_w, load_data_p = organizeSimulationData(MultipleSimulationsData)
    
    simuation = 0
    data_x_simuation,data_y_simuation,data_z_simuation,data_c_simuation,data_u_simuation,data_v_simuation,data_w_simuation,data_p_simuation = get_simulation_data(load_data_x,load_data_y,load_data_z,
                            load_data_c,load_data_u,load_data_v,load_data_w,load_data_p,simuation)
    
    
    #data_x , data_y , data_z , data_c , data_u , data_v ,data_w , data_p  = prepare_data(load_data_x, load_data_y , load_data_z,  load_data_c , load_data_u, load_data_v, load_data_w, load_data_p)

    data_x, data_y, data_z, data_m, data_c, data_u, data_v, data_w, data_p = pre_process_data(MultipleSimulationsData, arrLeakRate)
    
    # Validation files    
    path42 = current_directory + "/dataset/dispersion_42/"
    path43 = current_directory + "/dataset/dispersion_43/"
    path44 = current_directory + "/dataset/dispersion_44/"
    path45 = current_directory + "/dataset/dispersion_45/"
    path46 = current_directory + "/dataset/dispersion_46/"

    file42 = "010107"
    file43 = "010108"
    file44 = "010109"
    file45 = "010110"
    file46 = "010111"

    flowSolution42 = "FlowSolutionCC#0000000246"
    flowSolution43 = "FlowSolutionCC#0000000246"
    flowSolution44 = "FlowSolutionCC#0000000243"
    flowSolution45 = "FlowSolutionCC#0000000350" 

    data_points_43, data_c_43, data_u_43, data_v_43, data_w_43, data_p_43 = import_flacs_steady_simulation(path43, file43,flowSolution43)
    data_points_42, data_c_42, data_u_42, data_v_42, data_w_42, data_p_42 = import_flacs_steady_simulation(path42, file42,flowSolution42)
    data_points_45, data_c_45, data_u_45, data_v_45, data_w_45, data_p_45 = import_flacs_steady_simulation(path45, file45,flowSolution45)
    
    data_x_int_42, data_y_int_42, data_z_int_42, data_c_int_42 = interpolate_to_structured_grid(data_points_42, data_c_42 , Nx,Ny,Nz, method='linear')
    data_x_int_43, data_y_int_43, data_z_int_43, data_c_int_43 = interpolate_to_structured_grid(data_points_43, data_c_43 , Nx,Ny,Nz, method='linear')
    data_x_int_45, data_y_int_45, data_z_int_45, data_c_int_45 = interpolate_to_structured_grid(data_points_45, data_c_45 , Nx,Ny,Nz, method='linear')
    
    val_x_1_5,val_y_1_5,val_z_1_5,val_c_1_5 = data_x_int_43, data_y_int_43, data_z_int_43, data_c_int_43 
    val_x_2_5,val_y_2_5,val_z_2_5,val_c_2_5 = data_x_int_42, data_y_int_42, data_z_int_42, data_c_int_42 
    val_x_3_5,val_y_3_5,val_z_3_5,val_c_3_5 = data_x_int_45, data_y_int_45, data_z_int_45, data_c_int_45    
    
    ###########################################################################      
    minX,maxX = min(data_x),max(data_x)
    minY,maxY = min(data_y),max(data_y)
    minZ,maxZ = min(data_z),max(data_z)    
    deltaX = (maxX-minX)/Nx
    deltaY = (maxY-minY)/Ny
    deltaZ = (maxZ-minZ)/Nz    
    Vcell = deltaX*deltaY*deltaZ    
    ###########################################################################

    initial_time = time.time()
    
    minC,maxC = min(data_c),max(data_c)    
    norm_data_c = normalize_array_0_1(data_c)
    
    train_x, test_x, train_y, test_y, train_z, test_z, train_m,test_m, train_c, test_c = train_test_split(data_x, data_y, data_z, data_m, norm_data_c, test_size=test_frac, random_state=42)

    x_eqns, y_eqns, z_eqns = train_x, train_y, train_z    
    
    model = model(train_x, train_y, train_z, train_c, train_m,
                           x_eqns, y_eqns, z_eqns,
                           layers, batch_size,
                           Pec = Pe, Rey = Re) 
    
    
    # Calculate Re and Pe       
    train_initial_time = time.time()    
    loss_history = model.train(Niter , learning_rate=lr) 
    plot_loss(loss_history, savePath, title='loss_history.png')      
  
    train_final_time = time.time()
    test_initial_time = time.time()
    
    val_m_value = 1.5 
    val_m_1_5 = np.full((val_x_1_5.shape[0], 1), val_m_value) 
    norm_pred_c_1_5, pred_u_1_5, pred_v_1_5, pred_w_1_5, pred_p_1_5 = model.predict( val_x_1_5,val_y_1_5,val_z_1_5,val_m_1_5)   
    pred_c_1_5 = denormalize_array_0_1(norm_pred_c_1_5, minC, maxC)
    
    val_m_value = 2.5 
    val_m_2_5 = np.full((val_x_2_5.shape[0], 1), val_m_value) 
    norm_pred_c_2_5, pred_u_2_5, pred_v_2_5, pred_w_2_5, pred_p_2_5 = model.predict( val_x_2_5,val_y_2_5,val_z_2_5,val_m_2_5)    
    pred_c_2_5 = denormalize_array_0_1(norm_pred_c_2_5, minC, maxC)
    
    val_m_value = 3.5 
    val_m_3_5 = np.full((val_x_3_5.shape[0], 1), val_m_value) 
    norm_pred_c_3_5, pred_u_3_5, pred_v_3_5, pred_w_3_5, pred_p_3_5 = model.predict( val_x_3_5,val_y_3_5,val_z_3_5,val_m_3_5)     
    pred_c_3_5 = denormalize_array_0_1(norm_pred_c_3_5, minC, maxC)
    
    test_final_time = time.time()
    
    re_c_1_5 = relative_error(pred_c_1_5,val_c_1_5)
    re_c_2_5 = relative_error(pred_c_2_5,val_c_2_5)
    re_c_3_5 = relative_error(pred_c_3_5,val_c_3_5) 
    
    av_re_1_5 = av_rel_error(pred_c_1_5,val_c_1_5)
    av_re_2_5 = av_rel_error(pred_c_2_5,val_c_2_5)
    av_re_3_5 = av_rel_error(pred_c_3_5,val_c_3_5)  
    
    # Cloud Size        
    frac_val_c_1_5 = fraction_in_range(val_c_1_5,lim_inf_ch4,lim_sup_ch4)    
    frac_pred_c_1_5 = fraction_in_range(pred_c_1_5,lim_inf_ch4,lim_sup_ch4)     
    V_val_c_1_5 = frac_val_c_1_5 * Vcell * len(val_c_1_5) 
    V_pred_c_1_5 = frac_pred_c_1_5 * Vcell * len(pred_c_1_5) 
    
    frac_val_c_2_5 = fraction_in_range(val_c_2_5,lim_inf_ch4,lim_sup_ch4)    
    frac_pred_c_2_5 = fraction_in_range(pred_c_2_5,lim_inf_ch4,lim_sup_ch4)     
    V_val_c_2_5 = frac_val_c_2_5 * Vcell * len(val_c_2_5) 
    V_pred_c_2_5 = frac_pred_c_2_5 * Vcell * len(pred_c_2_5)
    
    frac_val_c_3_5 = fraction_in_range(val_c_3_5,lim_inf_ch4,lim_sup_ch4)    
    frac_pred_c_3_5 = fraction_in_range(pred_c_3_5,lim_inf_ch4,lim_sup_ch4)     
    V_val_c_3_5 = frac_val_c_3_5 * Vcell * len(val_c_3_5) 
    V_pred_c_3_5 = frac_pred_c_3_5 * Vcell * len(pred_c_3_5)
    
    generate_3D_vtk( val_x_1_5,val_y_1_5,val_z_1_5, pred_c_1_5, 'title', savePath+'model_prediction/pred_1_5.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_2_5,val_y_2_5,val_z_2_5, pred_c_2_5, 'title', savePath+'model_prediction/pred_2_5.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_3_5,val_y_3_5,val_z_3_5, pred_c_3_5, 'title', savePath+'model_prediction/pred_3_5.vtk', Nx, Ny, Nz, num_columns=6) 
    
    generate_3D_vtk( val_x_1_5,val_y_1_5,val_z_1_5, val_c_1_5, 'title', savePath+'simulation/dispersion_1_5.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_2_5,val_y_2_5,val_z_2_5, val_c_2_5, 'title', savePath+'simulation/dispersion_2_5.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_3_5,val_y_3_5,val_z_3_5, val_c_3_5, 'title', savePath+'simulation/dispersion_3_5.vtk', Nx, Ny, Nz, num_columns=6) 
    
    generate_3D_vtk( val_x_1_5,val_y_1_5,val_z_1_5, re_c_1_5, 'title', savePath+'error/error_1_5.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_2_5,val_y_2_5,val_z_2_5, re_c_2_5, 'title', savePath+'error/error_2_5.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_3_5,val_y_3_5,val_z_3_5, re_c_3_5, 'title', savePath+'error/error_3_5.vtk', Nx, Ny, Nz, num_columns=6)    
         
    # Calculate time     
    train_delta_t_ms,train_delta_t_s,train_delta_t_m, train_delta_t_h =  calculate_time (train_initial_time,train_final_time) # calculate train time
    test_delta_t_ms,test_delta_t_s,test_delta_t_m, test_delta_t_h =  calculate_time (test_initial_time,test_final_time) # calculate prediction time
       
        
    ###############################################################################
    
    path1 = current_directory + "/dataset/dispersion_35_steady/"
    path2 = current_directory + "/dataset/dispersion_36_steady/"
    path3 = current_directory + "/dataset/dispersion_37_steady/"
    path4 = current_directory + "/dataset/dispersion_38_steady/"
    
    file1 = "010100"
    file2 = "010101"
    file3 = "010102"
    file4 = "010103"    
    
    flowSolution1 = "FlowSolutionCC#0000000251"
    flowSolution2 = "FlowSolutionCC#0000000257"
    flowSolution3 = "FlowSolutionCC#0000000263"
    flowSolution4 =  "FlowSolutionCC#0000000259"
    
    
    data_points_1, data_c_1, data_u_1, data_v_1, data_w_1, data_p_1 = import_flacs_steady_simulation(path1, file1,flowSolution1)
    data_points_2, data_c_2, data_u_2, data_v_2, data_w_2, data_p_2 = import_flacs_steady_simulation(path2, file2,flowSolution2)
    data_points_3, data_c_3, data_u_3, data_v_3, data_w_3, data_p_3 = import_flacs_steady_simulation(path3, file3,flowSolution3)
    data_points_4, data_c_4, data_u_4, data_v_4, data_w_4, data_p_4 = import_flacs_steady_simulation(path4, file4,flowSolution4)    
    
    data_x_int_1, data_y_int_1, data_z_int_1, data_c_int_1 = interpolate_to_structured_grid(data_points_1, data_c_1 , Nx,Ny,Nz, method='linear')
    data_x_int_2, data_y_int_2, data_z_int_2, data_c_int_2 = interpolate_to_structured_grid(data_points_2, data_c_2 , Nx,Ny,Nz, method='linear')
    data_x_int_3, data_y_int_3, data_z_int_3, data_c_int_3 = interpolate_to_structured_grid(data_points_3, data_c_3 , Nx,Ny,Nz, method='linear')
    data_x_int_4, data_y_int_4, data_z_int_4, data_c_int_4 = interpolate_to_structured_grid(data_points_4, data_c_4 , Nx,Ny,Nz, method='linear')    
    
    val_x_1,val_y_1,val_z_1,val_c_1 = data_x_int_1, data_y_int_1, data_z_int_1, data_c_int_1 
    val_x_2,val_y_2,val_z_2,val_c_2 = data_x_int_2, data_y_int_2, data_z_int_2, data_c_int_2 
    val_x_3,val_y_3,val_z_3,val_c_3 = data_x_int_3, data_y_int_3, data_z_int_3, data_c_int_3    
    val_x_4,val_y_4,val_z_4,val_c_4 = data_x_int_4, data_y_int_4, data_z_int_4, data_c_int_4        
    
    ###############################################################################      
    
    val_m_value = 1
    val_m_1 = np.full((val_x_1.shape[0], 1), val_m_value) 
    norm_pred_c_1, pred_u_1, pred_v_1, pred_w_1, pred_p_1 = model.predict( val_x_1,val_y_1,val_z_1,val_m_1) 
    pred_c_1 = denormalize_array_0_1(norm_pred_c_1, minC, maxC)      
    
    val_m_value = 2
    val_m_2 = np.full((val_x_2.shape[0], 1), val_m_value) 
    norm_pred_c_2, pred_u_2, pred_v_2, pred_w_2, pred_p_2 = model.predict( val_x_2,val_y_2,val_z_2,val_m_2)  
    pred_c_2 = denormalize_array_0_1(norm_pred_c_2, minC, maxC)       
    
    val_m_value = 3
    val_m_3 = np.full((val_x_3.shape[0], 1), val_m_value) 
    norm_pred_c_3, pred_u_3, pred_v_3, pred_w_3, pred_p_3 = model.predict( val_x_3,val_y_3,val_z_3,val_m_3)       
    pred_c_3 = denormalize_array_0_1(norm_pred_c_3, minC, maxC)   
    
    val_m_value = 4
    val_m_4 = np.full((val_x_4.shape[0], 1), val_m_value) 
    norm_pred_c_4, pred_u_4, pred_v_4, pred_w_4, pred_p_4 = model.predict( val_x_4,val_y_4,val_z_4,val_m_4) 
    pred_c_4 = denormalize_array_0_1(norm_pred_c_4, minC, maxC)        
    
    re_c_1 = relative_error(pred_c_1,val_c_1)
    re_c_2 = relative_error(pred_c_2,val_c_2)
    re_c_3 = relative_error(pred_c_3,val_c_3) 
    re_c_4 = relative_error(pred_c_4,val_c_4) 
    
    av_re_1 = av_rel_error(pred_c_1,val_c_1)
    av_re_2 = av_rel_error(pred_c_2,val_c_2)
    av_re_3 = av_rel_error(pred_c_3,val_c_3)
    av_re_4 = av_rel_error(pred_c_4,val_c_4)    
    
    frac_val_c_1 = fraction_in_range(val_c_1,lim_inf_ch4,lim_sup_ch4)    
    frac_pred_c_1 = fraction_in_range(pred_c_1,lim_inf_ch4,lim_sup_ch4)     
    V_val_c_1 = frac_val_c_1 * Vcell * len(val_c_1) 
    V_pred_c_1 = frac_pred_c_1 * Vcell * len(pred_c_1) 
    
    frac_val_c_2 = fraction_in_range(val_c_2,lim_inf_ch4,lim_sup_ch4)    
    frac_pred_c_2 = fraction_in_range(pred_c_2,lim_inf_ch4,lim_sup_ch4)     
    V_val_c_2 = frac_val_c_2 * Vcell * len(val_c_2) 
    V_pred_c_2 = frac_pred_c_2 * Vcell * len(pred_c_2)
    
    frac_val_c_3 = fraction_in_range(val_c_3,lim_inf_ch4,lim_sup_ch4)    
    frac_pred_c_3 = fraction_in_range(pred_c_3,lim_inf_ch4,lim_sup_ch4)     
    V_val_c_3 = frac_val_c_3 * Vcell * len(val_c_3) 
    V_pred_c_3 = frac_pred_c_3 * Vcell * len(pred_c_3)
    
    frac_val_c_4 = fraction_in_range(val_c_4,lim_inf_ch4,lim_sup_ch4)    
    frac_pred_c_4 = fraction_in_range(pred_c_4,lim_inf_ch4,lim_sup_ch4)     
    V_val_c_4 = frac_val_c_4 * Vcell * len(val_c_4) 
    V_pred_c_4 = frac_pred_c_4 * Vcell * len(pred_c_4)     
    
    generate_3D_vtk( val_x_1,val_y_1,val_z_1, pred_c_1, 'title', savePath+'model_prediction/pred_1.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_2,val_y_2,val_z_2, pred_c_2, 'title', savePath+'model_prediction/pred_2.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_3,val_y_3,val_z_3, pred_c_3, 'title', savePath+'model_prediction/pred_3.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_4,val_y_4,val_z_4, pred_c_4, 'title', savePath+'model_prediction/pred_4.vtk', Nx, Ny, Nz, num_columns=6) 
    
    generate_3D_vtk( val_x_1,val_y_1,val_z_1, val_c_1, 'title', savePath+'simulation/dispersion_1.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_2,val_y_2,val_z_2, val_c_2, 'title', savePath+'simulation/dispersion_2.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_3,val_y_3,val_z_3, val_c_3, 'title', savePath+'simulation/dispersion_3.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_4,val_y_4,val_z_4, val_c_4, 'title', savePath+'simulation/dispersion_4.vtk', Nx, Ny, Nz, num_columns=6) 
    
    generate_3D_vtk( val_x_1,val_y_1,val_z_1, re_c_1, 'title', savePath+'error/error_1.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_2,val_y_2,val_z_2, re_c_2, 'title', savePath+'error/error_2.vtk', Nx, Ny, Nz, num_columns=6) 
    generate_3D_vtk( val_x_3,val_y_3,val_z_3, re_c_3, 'title', savePath+'error/error_3.vtk', Nx, Ny, Nz, num_columns=6)  
    generate_3D_vtk( val_x_4,val_y_4,val_z_4, re_c_4, 'title', savePath+'error/error_4.vtk', Nx, Ny, Nz, num_columns=6)    
    
    ###############################################################################
         
    
    important_variables = []
    
    important_variables = {
        "savePath": savePath,
        "dataFilePath": dataFilePath,
        "batch_size": batch_size,
        "loss_history": loss_history,
        "layers": layers,
        "test_frac": test_frac,
        "Pe": Pe,
        "Re": Re,
        "lr": lr,
        "steps_frequency": steps_frequency,
        "paths": paths,
        "flowSolutions": flowSolutions,
        "filenames": filenames,
        "titles": titles,
        "arrLeakRate": arrLeakRate,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "initial_time": initial_time,
        "train_delta_t_ms": train_delta_t_ms,
        "train_delta_t_s": train_delta_t_s,
        "train_delta_t_m": train_delta_t_m,
        "train_delta_t_h": train_delta_t_h,
        "test_delta_t_ms": test_delta_t_ms,
        "test_delta_t_s": test_delta_t_s,
        "test_delta_t_m": test_delta_t_m,
        "test_delta_t_h": test_delta_t_h
    }

    # Determine file path for pickle file
    pickle_file_path = os.path.join(savePath, "important_variables.pickle")
    
    # Store variables in a pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(important_variables, f)
