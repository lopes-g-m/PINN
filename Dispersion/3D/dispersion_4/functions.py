import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import h5py



def load_excel_data(excel_path):  
  # read excel 
  excel_data_frame = pd.read_excel(excel_path) 
  # read columns from cavity file 
  # convert data to array 
  data_mass_frac = np.array(excel_data_frame['Scalars'])
  data_size = len(data_mass_frac)
  data_points_0 = np.array(excel_data_frame['Points:0']).reshape(data_size,1)
  data_points_1 = np.array(excel_data_frame['Points:1']).reshape(data_size,1)
  data_points_2 = np.array(excel_data_frame['Points:2']).reshape(data_size,1)
  data_mass_frac = np.array(excel_data_frame['Scalars']).reshape(data_size,1) 
  data_time = np.zeros(data_size).reshape(data_size,1) 
  # train data 
  train_x = data_points_0
  train_y = data_points_1
  train_z = data_points_2
  train_c =  data_mass_frac
  return [train_x, train_y , train_z,  train_c] # all function variables


def load_geometry(excel_path):  
  # read excel 
  excel_data_frame = pd.read_excel(excel_path) 

  x_lim_inf = np.array(excel_data_frame['x_lim_inf'])
  
  data_size = len(x_lim_inf)
  
  x_lim_inf = np.array(excel_data_frame['x_lim_inf'])
  y_lim_inf = np.array(excel_data_frame['y_lim_inf'])
  x_lim_sup = np.array(excel_data_frame['x_lim_sup'])
  y_lim_sup = np.array(excel_data_frame['y_lim_sup'])
  
  return [x_lim_inf, y_lim_inf , x_lim_sup, y_lim_sup] 

def normalize_array_0_1(arr):
    """
    Normalizes an array to the range of [0,1].
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr

def denormalize_array_0_1(normalized_arr, original_min, original_max):
    """
    Denormalizes an array from the range of [0,1] back to its original scale.
    """
    denormalized_arr = normalized_arr * (original_max - original_min) + original_min
    return denormalized_arr

def calculate_time (initial_time,final_time):
  delta_t = final_time - initial_time
  delta_t_ms = delta_t * 1000 # mili seconds 
  delta_t_s = delta_t  # seconds 
  delta_t_m = delta_t/60  # minutes 
  delta_t_h = delta_t/3600 # hours 
  return delta_t_ms,delta_t_s,delta_t_m,delta_t_h

def split_train_test(load_x, load_y , load_z, load_c, test_frac):
  # input and output variables 
  var_input = np.concatenate((load_x, load_y, load_z), axis=1) 
  #var_output = np.concatenate((load_c), axis=1) 
  var_output = load_c
  # split train test 
  var_in_train, var_in_test, var_out_train, var_out_test = train_test_split( var_input, var_output, test_size=test_frac, random_state=42) 
  train_x , train_y , train_z = np.split(var_in_train, 3, axis=1)  
  test_x , test_y, test_z = np.split(var_in_test, 3, axis=1) 
  train_c = var_out_train
  test_c = var_out_test
  #train_c = np.split(var_out_train, 1, axis=1)  
  #test_c = np.split(var_out_test, 1, axis=1)  
  return  train_x , train_y, train_z ,train_c, test_x , test_y , test_z,test_c 

def calc_conc(arr_w): # calculate concentration 
  # properties for 1 atm and T=27oC
  rho_air = 1.1769 # kg/m³ (welty)
  rho_ch4 = 0.65281 # kg/m³ (NIST)
  rho_mix = arr_w * rho_ch4 + (1-arr_w) * rho_air
  arr_mass_conc_ch4 = arr_w * rho_mix 
  return arr_mass_conc_ch4

def get_sample(NdataPoints, norm_load_x, norm_load_y, norm_load_u, norm_load_v,norm_load_p, norm_load_c): 
    sample_norm_load_x = np.random.choice(norm_load_x.reshape(-1), NdataPoints, replace=False)
    sample_norm_load_y = np.random.choice(norm_load_y.reshape(-1), NdataPoints, replace=False)
    sample_norm_load_u = np.random.choice(norm_load_u.reshape(-1), NdataPoints, replace=False)
    sample_norm_load_v = np.random.choice(norm_load_v.reshape(-1), NdataPoints, replace=False)
    sample_norm_load_p = np.random.choice(norm_load_p.reshape(-1), NdataPoints, replace=False)
    sample_norm_load_c = np.random.choice(norm_load_c.reshape(-1), NdataPoints, replace=False)
    return sample_norm_load_x,sample_norm_load_y,sample_norm_load_u,sample_norm_load_v,sample_norm_load_p,sample_norm_load_c 

def normalize_dataset_0_1(load_x,load_y, load_z , load_c): 
    norm_load_x = normalize_array_0_1(load_x)
    norm_load_y = normalize_array_0_1(load_y)
    norm_load_z = normalize_array_0_1(load_z)    
    norm_load_c = normalize_array_0_1(load_c)     
    return norm_load_x,norm_load_y,norm_load_z, norm_load_c

def percentage_greater_than(array, limit):
    count_greater = sum(1 for element in array if element > limit)
    percentage = (count_greater / len(array)) * 100.0    
    return percentage


def fraction_in_range(data, lim_inf_ch4=0.05, lim_sup_ch4=0.15):
    """
    Calculate the fraction of data points that fall within the specified range.

    Parameters:
    data (array-like): The data to be analyzed.
    lim_inf_ch4 (float): The lower limit of the range (default is 0.05).
    lim_sup_ch4 (float): The upper limit of the range (default is 0.15).

    Returns:
    float: The fraction of data points within the specified range.
    """
    data = np.array(data)
    within_range = (data >= lim_inf_ch4) & (data <= lim_sup_ch4)
    fraction = np.sum(within_range) / len(data)
    
    return fraction








  
