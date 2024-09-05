import h5py
import numpy as np 
from collections import defaultdict
from scipy.interpolate import griddata

def process_simulation_data(SimulationDataCGNS_1_grid, SimulationDataCGNS_1,flowSolution):
    # Check and move "GridCoordinates" if present in the grid data
    if "GridCoordinates" in SimulationDataCGNS_1_grid:
        SimulationDataCGNS_1["GridCoordinates"] = SimulationDataCGNS_1_grid.pop("GridCoordinates")

    
    # Extract grid coordinates
    data_points = SimulationDataCGNS_1["GridCoordinates"]
    
    # Extract flow solution data
    flowSolution = SimulationDataCGNS_1[flowSolution]
    
    # Extract specific data fields
    data_c = flowSolution["MoleFractionFuel"][" data"]
    data_u = flowSolution["VelocityX"][" data"]
    data_v = flowSolution["VelocityY"][" data"]
    data_w = flowSolution["VelocityZ"][" data"]
    data_p = flowSolution["Pressure"][" data"]
    
    # Adjust dimensions
    data_c = data_c[:-1, :-1, :-1]
    data_u = data_u[:-1, :-1, :-1]
    data_v = data_v[:-1, :-1, :-1]
    data_w = data_w[:-1, :-1, :-1]
    data_p = data_p[:-1, :-1, :-1]
    
    # Flatten and reshape the data
    data_c = data_c.flatten().reshape(-1, 1)
    data_u = data_u.flatten().reshape(-1, 1)
    data_v = data_v.flatten().reshape(-1, 1)
    data_w = data_w.flatten().reshape(-1, 1)
    data_p = data_p.flatten().reshape(-1, 1)
    
    # Extract coordinate data
    simulation_points_x = data_points["CoordinateX"][" data"]
    simulation_points_y = data_points["CoordinateY"][" data"]
    simulation_points_z = data_points["CoordinateZ"][" data"]
    
    # Flatten and reshape the coordinate data
    data_x = simulation_points_x.flatten().reshape(-1, 1)
    data_y = simulation_points_y.flatten().reshape(-1, 1)
    data_z = simulation_points_z.flatten().reshape(-1, 1)
    
    # Return the loaded data
    return data_x, data_y, data_z, data_c



def import_cgns_flacs(path, fileName):
    filePath = path + fileName + ".cgns"
    with h5py.File(filePath, 'r') as f:
        group_dict = {}
        
        def traverse_hdf5_group(group, grandparent_key='', parent_key='', grandgrandparent_key=''):
            data_dict = {}
            for key in group.keys():
                item = group[key]
                key_path = f"{grandgrandparent_key}/{grandparent_key}/{parent_key}/{key}" if grandgrandparent_key and grandparent_key and parent_key else \
                           f"{grandparent_key}/{parent_key}/{key}" if grandparent_key and parent_key else \
                           f"{parent_key}/{key}" if parent_key else key
                if isinstance(item, h5py.Group):
                    traverse_hdf5_group(item, grandparent_key=parent_key, parent_key=key, grandgrandparent_key=grandparent_key)
                elif isinstance(item, h5py.Dataset):
                    data_array = np.array(item)
                    data_dict[key] = data_array 
            group_key = f"{grandparent_key}/{parent_key}" if grandparent_key and parent_key else parent_key
            group_dict[group_key] = data_dict
        
        traverse_hdf5_group(f)
    
    # Flatten the dictionary keys
    simulationDataCGNS = {}
    for key, value in group_dict.items():
        parts = key.split('/')
        main_key = parts[0]  # Use the first part of the key as the main key
        rest_of_key = '/'.join(parts[1:])  # Store the rest of the key as nested dictionaries
        if main_key not in simulationDataCGNS:
            simulationDataCGNS[main_key] = {}

        # Perform one more split on the rest_of_key
        parts_rest = rest_of_key.split('/')
        if len(parts_rest) > 1:
            second_level_key = parts_rest[0]
            rest_of_rest_key = '/'.join(parts_rest[1:])
            if second_level_key not in simulationDataCGNS[main_key]:
                simulationDataCGNS[main_key][second_level_key] = {}
            simulationDataCGNS[main_key][second_level_key][rest_of_rest_key] = value
        else:
            simulationDataCGNS[main_key][rest_of_key] = value

    return simulationDataCGNS

def import_flacs_steady_simulation(path,fileName,flowSolution):
    
    simulationDataCGNS = import_cgns_flacs(path,fileName)   
    
    data_points = simulationDataCGNS["GridCoordinates"]
    
    flowSolution = simulationDataCGNS[flowSolution]
        
    data_c = flowSolution["MoleFractionFuel"] 
    data_u = flowSolution["VelocityX"]
    data_v = flowSolution["VelocityY"]
    data_w = flowSolution["VelocityZ"]
    data_p = flowSolution["Pressure"]
    
    
    
    data_c = data_c[" data"]
    data_u = data_u[" data"]
    data_v = data_v[" data"]
    data_w = data_w[" data"]
    data_p = data_p[" data"]        
    
    return  data_points, data_c ,data_u , data_v , data_w, data_p


def import_multiple_flacs_steady_simulations(paths, filenames,flowSolutions):
    all_simulation_data = {}
    
    for path, file_name,flowSolution in zip(paths, filenames,flowSolutions):
        data_points, data_c, data_u, data_v, data_w, data_p = import_flacs_steady_simulation(path, file_name,flowSolution)
        
        simulation_result = {
            "points": data_points,
            "c": data_c,
            "u": data_u,
            "v": data_v,
            "w": data_w,
            "p": data_p
        }
        
        all_simulation_data[path + file_name] = simulation_result
        
        
    all_simulation_data = convert_dic_array(all_simulation_data)
    
        
    
    return all_simulation_data
    
def convert_dic_array(data):    
    arrays_with_keys = []
    for outer_key in data:
        inner_dict = data[outer_key]
        for inner_key, inner_array in inner_dict.items():
            arrays_with_keys.append((outer_key, inner_key, inner_array))     
    return arrays_with_keys


def organizeSimulationData(MultipleSimulationsData):
    
    data_points = [] 
    data_c = []
    data_u = []
    data_v = []
    data_w = []
    data_p = []

    for i, (outer_key, inner_key, inner_array) in enumerate(MultipleSimulationsData):
        if inner_key == 'points':
            data_points.append(inner_array)        
        if inner_key == 'c':
            data_c.append(inner_array)
        elif inner_key == 'u':
            data_u.append(inner_array)
        elif inner_key == 'v':
            data_v.append(inner_array)
        elif inner_key == 'w':
            data_w.append(inner_array)
        elif inner_key == 'p':
            data_p.append(inner_array) 
            
    simulation_points = defaultdict(list)    
    for point in data_points:
        for key, value in point.items():
            simulation_points[key].append(value)            
            
            
    simulation_points_x = {}
    simulation_points_y = {}
    simulation_points_z = {} 
    
    simulation_points_x = simulation_points["CoordinateX"]
    simulation_points_y = simulation_points["CoordinateY"]
    simulation_points_z = simulation_points["CoordinateZ"]

    return     simulation_points_x, simulation_points_y , simulation_points_z,  data_c , data_u, data_v, data_w, data_p
    

def flatten_arrays(data_c):
    if isinstance(data_c, list):
        flattened_arrays = [arr.flatten() for arr in data_c]
    elif isinstance(data_c, dict):
        flattened_arrays = [arr.flatten() for arr in data_c.values()]
    else:
        raise ValueError("Input must be a list of NumPy arrays or a dictionary with arrays as values.")    
    return flattened_arrays


def flatten_arrays_dic(data_x):
    flattened_arrays = []
    for d in data_x:
        for arr in d.values():
            flattened_arrays.append(arr.flatten())
    return flattened_arrays
  
def prepare_data(load_data_x, load_data_y , load_data_z,  load_data_c , load_data_u, load_data_v, load_data_w, load_data_p):
        load_data_c = remove_element_mesh(load_data_c)    
        load_data_u = remove_element_mesh(load_data_u)    
        load_data_v = remove_element_mesh(load_data_v)    
        load_data_w = remove_element_mesh(load_data_w)    
        load_data_p = remove_element_mesh(load_data_p)   
           
        data_x = flatten_arrays_dic(load_data_x) 
        data_y = flatten_arrays_dic(load_data_y)
        data_z = flatten_arrays_dic(load_data_z)
        data_c = flatten_arrays(load_data_c)
        data_u = flatten_arrays(load_data_u)
        data_v = flatten_arrays(load_data_v)
        data_w = flatten_arrays(load_data_w) 
        data_p = flatten_arrays(load_data_p)        
        
        data_x = np.concatenate(data_x).reshape(-1, 1)
        data_y = np.concatenate(data_y).reshape(-1, 1)
        data_z = np.concatenate(data_z).reshape(-1, 1)
        data_c = np.concatenate(data_c).reshape(-1, 1)
        data_u = np.concatenate(data_u).reshape(-1, 1)
        data_v = np.concatenate(data_v).reshape(-1, 1)
        data_w = np.concatenate(data_w).reshape(-1, 1)
        data_p = np.concatenate(data_p).reshape(-1, 1)       
        
        return data_x , data_y , data_z , data_c , data_u , data_v ,data_w , data_p 



def get_simulation_data(load_data_x,load_data_y,load_data_z,
                        load_data_c,load_data_u,load_data_v,load_data_w,load_data_p,position):       
    load_data_c = remove_element_mesh(load_data_c)    
    load_data_u = remove_element_mesh(load_data_u)    
    load_data_v = remove_element_mesh(load_data_v)    
    load_data_w = remove_element_mesh(load_data_w)    
    load_data_p = remove_element_mesh(load_data_p)          
    data_x = flatten_arrays_dic(load_data_x) 
    data_y = flatten_arrays_dic(load_data_y)
    data_z = flatten_arrays_dic(load_data_z)
    data_c = flatten_arrays(load_data_c)
    data_u = flatten_arrays(load_data_u)
    data_v = flatten_arrays(load_data_v)
    data_w = flatten_arrays(load_data_w) 
    data_p = flatten_arrays(load_data_p)      
    data_x_position = data_x[position].reshape(-1,1)
    data_y_position = data_y[position].reshape(-1,1)
    data_z_position = data_z[position].reshape(-1,1)
    data_c_position = data_c[position].reshape(-1,1)
    data_u_position = data_u[position].reshape(-1,1)
    data_v_position = data_v[position].reshape(-1,1)
    data_w_position = data_w[position].reshape(-1,1)
    data_p_position = data_p[position].reshape(-1,1)
    
    return data_x_position,data_y_position,data_z_position,data_c_position,data_u_position,data_v_position,data_w_position,data_p_position

def remove_point_mesh(array):
    shape = array.shape
    sliced_array = array[:-1, :-1, :-1]  # Assuming you want to remove the last element in each dimension
    return sliced_array



def remove_element_mesh(list_of_arrays):    
    for idx, arr in enumerate(list_of_arrays):
        shape = arr.shape
        sliced_arr = arr[:-1, :-1, :-1]  
        list_of_arrays[idx] = sliced_arr        
    return list_of_arrays


def reduce_dimensions(arr):
    if arr.shape[0] > 1 and arr.shape[1] > 1 and arr.shape[2] > 1:
        return arr[:-1, :-1, :-1]
    else:
        raise ValueError("Array dimensions must be greater than 1 to reduce each dimension by 1")

# Function to apply reduction to a list of arrays
def reduce_dimensions_list(arr_list):
    return [reduce_dimensions(arr) for arr in arr_list]



def pre_process_data(MultipleSimulationsData, arrLeakRate):
    
    load_data_x, load_data_y, load_data_z, load_data_c, load_data_u, load_data_v, load_data_w, load_data_p = organizeSimulationData(MultipleSimulationsData)
    
    reduced_data_c = reduce_dimensions_list(load_data_c)
    reduced_data_u = reduce_dimensions_list(load_data_u)
    reduced_data_v = reduce_dimensions_list(load_data_v)
    reduced_data_w = reduce_dimensions_list(load_data_w)
    reduced_data_p = reduce_dimensions_list(load_data_p)         
    
    dimensionX = [len(dic[' data']) for dic in load_data_x]
    dimensionY = [len(dic[' data'][0]) for dic in load_data_y]
    dimensionZ = [len(dic[' data'][0][0]) for dic in load_data_z]    

    flat_load_data_c = flatten_arrays(reduced_data_c)
    flat_load_data_u = flatten_arrays(reduced_data_u)
    flat_load_data_v = flatten_arrays(reduced_data_v)
    flat_load_data_w = flatten_arrays(reduced_data_w)
    flat_load_data_p = flatten_arrays(reduced_data_p)
    flat_load_data_x = flatten_arrays_dic(load_data_x)
    flat_load_data_y = flatten_arrays_dic(load_data_y)
    flat_load_data_z = flatten_arrays_dic(load_data_z)
    
    element_sizes = [len(arr) for arr in flat_load_data_c]
    flat_load_data_R = [np.array([arrLeakRate[i % len(arrLeakRate)] for _ in range(size)]).reshape(-1, 1) for i, size in enumerate(element_sizes)]

    data_c = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_c])
    data_u = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_u])
    data_v = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_v])
    data_w = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_w])
    data_p = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_p])
    data_x = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_x])
    data_y = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_y])
    data_z = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_z])
    data_R = np.concatenate([arr.reshape(-1, 1) for arr in flat_load_data_R])
    
    return data_x, data_y, data_z, data_R, data_c, data_u, data_v, data_w, data_p


    
    