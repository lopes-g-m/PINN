import numpy as np
from matplotlib import artist
import matplotlib.axes as axes
from matplotlib import axes
import matplotlib.pyplot as plt 
import matplotlib.tri as tri 
import vtk 
from vtk import vtkPoints
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from scipy.interpolate import griddata
import os 
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,current_directory)

from pre_process import remove_point_mesh

def plot_2d_domains(data_sets, titles, x_label, y_label, str_ident, savePath):
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))
    fig.subplots_adjust(hspace=0.5)

    for i, (ob_x, ob_y, ob_u) in enumerate(data_sets):
        max_value = max(ob_u)
        min_value = min(ob_u)
        plot_x = ob_x.flatten()
        plot_y = ob_y.flatten()
        plot_u = ob_u.flatten()

        triang = tri.Triangulation(plot_x, plot_y)
        ax = axs[i // 2, i % 2]

        ax.set_aspect("equal")
        tcf = ax.tricontourf(
            triang,
            plot_u,
            levels=100,
            cmap="jet",
            vmin=min_value,
            vmax=max_value
        )

        ticks_arr = np.linspace(min_value, max_value, 8).flatten()      
        
        fig.colorbar(tcf, ax=ax, location="right", format='%.1e', ticks=ticks_arr, shrink=0.8)  # Adjust the shrink parameter

        ax.tricontour(triang, plot_u, linestyles="solid", colors="k", linewidths=0)
        ax.set_title(titles[i])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    figName = savePath + str_ident + '_combined.png'
    fig.savefig(figName)
    plt.show()
    return 

def plot_data_distribution(train_x, train_y, train_u, train_v, train_p, train_c, title, str_ident, savePath):
    var_input_train = np.concatenate((train_x, train_y), axis=1)
    var_output_train = np.concatenate((train_u, train_v, train_p, train_c), axis=1)
    x_labels = ['X', 'Y']
    # plot input variable
    fig, ax = plt.subplots()
    plt.plot(var_input_train.T, 'ob', fillstyle='full', markersize=6, alpha=1)
    plt.grid(':', linewidth=1)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation='vertical')
    plt.title(title + ' - Input Distribution')  # Added title
    plt.tight_layout()
    plt.show()
    figName = savePath + 'Input_dist_' + str_ident + '.png'
    fig.savefig(figName)

    x_labels = ['U', 'V', 'P', 'C']
    # plot output variable
    fig, ax = plt.subplots()
    plt.plot(var_output_train.T, 'ob', fillstyle='full', markersize=6, alpha=1)
    plt.grid(':', linewidth=1)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation='vertical')
    plt.title(title + ' - Output Distribution')  # Added title
    plt.tight_layout()
    plt.show()
    figName = savePath + 'Output_dist_' + str_ident + '.png'
    fig.savefig(figName)
    return

def relative_error(pred, ref):
      abs_diff = np.abs(pred - ref)
      rel_diff = abs_diff / (max(ref) - min(ref))
      rel_err = rel_diff * 100
      return rel_err

def av_rel_error(pred, ref):
      rel_error = relative_error(pred, ref)
      av_rel_error = np.mean(rel_error)
      return av_rel_error 
  
def plot_2d_domains_bump(data_sets, titles, x_label, y_label, str_ident, savePath, x_lim_inf, y_lim_inf):
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))
    fig.subplots_adjust(hspace=0.5)

    for i, (ob_x, ob_y, ob_u) in enumerate(data_sets):
        max_value = max(ob_u)
        min_value = min(ob_u)
        plot_x = ob_x.flatten()
        plot_y = ob_y.flatten()
        plot_u = ob_u.flatten()

        triang = tri.Triangulation(plot_x, plot_y)
        ax = axs[i // 2, i % 2]

        ax.set_aspect("equal")
        tcf = ax.tricontourf(
            triang,
            plot_u,
            levels=100,
            cmap="jet",
            vmin=min_value,
            vmax=max_value
        )

        ax.plot(x_lim_inf, y_lim_inf, color='black', label=None)
        ax.fill_between(x_lim_inf, y_lim_inf, alpha=1.0, color='black', label=None)


        ticks_arr = np.linspace(min_value, max_value, 8).flatten()      
        
        fig.colorbar(tcf, ax=ax, location="right", format='%.1e', ticks=ticks_arr, shrink=0.8)
        ax.tricontour(triang, plot_u, linestyles="solid", colors="k", linewidths=0)

        ax.set_title(titles[i])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    figName = savePath + str_ident + '_combined.png'
    fig.savefig(figName)
    plt.show()

def plot_3_2d_domains(ob_x_list, ob_y_list, ob_u_list, ob_v_list, titles, x_label, y_label, unit, str_ident, savePath):
    num_plots = len(ob_x_list)

    fig, axs = plt.subplots(num_plots, 3, figsize=(8 * 3, 3 * num_plots))

    for i in range(num_plots):
        ob_x, ob_y, ob_u, ob_v = ob_x_list[i], ob_y_list[i], ob_u_list[i], ob_v_list[i]

        ob_uv = np.concatenate((ob_u, ob_v))
        max_value = max([ob_uv.max(), ob_v.max()])
        min_value = min([ob_uv.min(), ob_v.min()])

        ob_re = relative_error(ob_v, ob_u)
        max_re = ob_re.max()
        min_re = ob_re.min()

        plot_x = ob_x.flatten()
        plot_y = ob_y.flatten()

        axs[i, 0].set_aspect("equal")
        axs[i, 1].set_aspect("equal")
        axs[i, 2].set_aspect("equal")

        ticks_arr = np.linspace(min_value, max_value, 10).flatten()
        ticks_re = np.linspace(min_re, max_re, 10).flatten()

        tcf_u = axs[i, 0].tricontourf(
            tri.Triangulation(plot_x, plot_y),
            ob_u.flatten(),
            levels=100,
            cmap="jet")

        tcf_v = axs[i, 1].tricontourf(
            tri.Triangulation(plot_x, plot_y),
            ob_v.flatten(),
            levels=100,
            cmap="jet")

        tcf_re = axs[i, 2].tricontourf(
            tri.Triangulation(plot_x, plot_y),
            ob_re.flatten(),
            levels=100,
            cmap="jet")

        cbar_u = fig.colorbar(tcf_u, ax=axs[i, 0], shrink=0.7, format='%.1e', orientation='vertical')
        cbar_u.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_u.ax.xaxis.set_ticks_position('bottom')
        cbar_u.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_u.ax.xaxis.set_label_text(unit)
        label = cbar_u.ax.xaxis.get_label()
        label.set_rotation(0)

        cbar_v = fig.colorbar(tcf_v, ax=axs[i, 1], shrink=0.7, format='%.1e', orientation='vertical')
        cbar_v.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_v.ax.xaxis.set_ticks_position('bottom')
        cbar_v.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_v.ax.xaxis.set_label_text(unit)
        label = cbar_v.ax.xaxis.get_label()
        label.set_rotation(0)

        cbar_re = fig.colorbar(tcf_re, ax=axs[i, 2], shrink=0.7, format='%02d', orientation='vertical')
        cbar_re.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_re.ax.xaxis.set_ticks_position('bottom')
        cbar_re.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_re.ax.xaxis.set_label_text("")
        label = cbar_re.ax.xaxis.get_label()
        label.set_rotation(0)

        axs[i, 0].tricontour(tri.Triangulation(plot_x, plot_y), ob_u.flatten(), linestyles="solid", colors="k",
                             linewidths=0)
        axs[i, 1].tricontour(tri.Triangulation(plot_x, plot_y), ob_v.flatten(), linestyles="solid", colors="k",
                             linewidths=0)
        axs[i, 2].tricontour(tri.Triangulation(plot_x, plot_y), ob_re.flatten(), linestyles="solid", colors="k",
                             linewidths=0)

        title_font = {'size': '12'}

        axs[i, 0].set_xlabel(x_label)
        axs[i, 0].set_ylabel(y_label)
        axs[i, 0].set_title(titles[i] + ' ' + '$_{CFD}$', fontdict=title_font)

        axs[i, 1].set_xlabel(x_label)
        axs[i, 1].set_ylabel(y_label)
        axs[i, 1].set_title(titles[i] + ' ' + '$_{ANN}$', fontdict=title_font)

        axs[i, 2].set_xlabel(x_label)
        axs[i, 2].set_ylabel(y_label)
        axs[i, 2].set_title(titles[i] + ' ' + '$_{NMAPE}$', fontdict=title_font)

    figName = savePath + titles[0] + str_ident + '_all_variables.png'
    fig.savefig(figName)
    return

def plot_3_2d_domains_bump(ob_x_list, ob_y_list, ob_u_list, ob_v_list, titles, x_label, y_label, unit, str_ident, x_lim_inf, y_lim_inf, savePath):
    num_plots = len(ob_x_list)

    fig, axs = plt.subplots(num_plots, 3, figsize=(8 * 3, 3 * num_plots))

    for i in range(num_plots):
        ob_x, ob_y, ob_u, ob_v = ob_x_list[i], ob_y_list[i], ob_u_list[i], ob_v_list[i]

        ob_uv = np.concatenate((ob_u, ob_v))
        max_value = max([ob_uv.max(), ob_v.max()])
        min_value = min([ob_uv.min(), ob_v.min()])

        ob_re = relative_error(ob_v, ob_u)
        max_re = ob_re.max()
        min_re = ob_re.min()

        plot_x = ob_x.flatten()
        plot_y = ob_y.flatten()

        axs[i, 0].set_aspect("equal")
        axs[i, 1].set_aspect("equal")
        axs[i, 2].set_aspect("equal")

        ticks_arr = np.linspace(min_value, max_value, 10).flatten()
        ticks_re = np.linspace(min_re, max_re, 10).flatten()

        tcf_u = axs[i, 0].tricontourf(
            tri.Triangulation(plot_x, plot_y),
            ob_u.flatten(),
            levels=100,
            cmap="jet")

        tcf_v = axs[i, 1].tricontourf(
            tri.Triangulation(plot_x, plot_y),
            ob_v.flatten(),
            levels=100,
            cmap="jet")

        tcf_re = axs[i, 2].tricontourf(
            tri.Triangulation(plot_x, plot_y),
            ob_re.flatten(),
            levels=100,
            cmap="jet")

        axs[i, 0].plot(x_lim_inf, y_lim_inf, color='black', label=None)
        axs[i, 0].fill_between(x_lim_inf, y_lim_inf, alpha=0.9, color='black', label=None)

        axs[i, 1].plot(x_lim_inf, y_lim_inf, color='black', label=None)
        axs[i, 1].fill_between(x_lim_inf, y_lim_inf, alpha=0.9, color='black', label=None)

        axs[i, 2].plot(x_lim_inf, y_lim_inf, color='black', label=None)
        axs[i, 2].fill_between(x_lim_inf, y_lim_inf, alpha=0.9, color='black', label=None)

        cbar_u = fig.colorbar(tcf_u, ax=axs[i, 0], shrink=0.7, format='%.1e', orientation='vertical')
        cbar_u.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_u.ax.xaxis.set_ticks_position('bottom')
        cbar_u.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_u.ax.xaxis.set_label_text(unit)
        label = cbar_u.ax.xaxis.get_label()
        label.set_rotation(0)

        cbar_v = fig.colorbar(tcf_v, ax=axs[i, 1], shrink=0.7, format='%.1e', orientation='vertical')
        cbar_v.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_v.ax.xaxis.set_ticks_position('bottom')
        cbar_v.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_v.ax.xaxis.set_label_text(unit)
        label = cbar_v.ax.xaxis.get_label()
        label.set_rotation(0)

        cbar_re = fig.colorbar(tcf_re, ax=axs[i, 2], shrink=0.7, format='%02d', orientation='vertical')
        cbar_re.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_re.ax.xaxis.set_ticks_position('bottom')
        cbar_re.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_re.ax.xaxis.set_label_text("")
        label = cbar_re.ax.xaxis.get_label()
        label.set_rotation(0)

        for ax in axs[i]:
            ax.tricontour(tri.Triangulation(plot_x, plot_y), ob_u.flatten(), linestyles="solid", colors="k", linewidths=0)
            ax.tricontour(tri.Triangulation(plot_x, plot_y), ob_v.flatten(), linestyles="solid", colors="k", linewidths=0)
            ax.tricontour(tri.Triangulation(plot_x, plot_y), ob_re.flatten(), linestyles="solid", colors="k", linewidths=0)

        title_font = {'size': '12'}

        axs[i, 0].set_xlabel(x_label)
        axs[i, 0].set_ylabel(y_label)
        axs[i, 0].set_title(titles[i] + ' ' + '$_{CFD}$', fontdict=title_font)

        axs[i, 1].set_xlabel(x_label)
        axs[i, 1].set_ylabel(y_label)
        axs[i, 1].set_title(titles[i] + ' ' + '$_{ANN}$', fontdict=title_font)

        axs[i, 2].set_xlabel(x_label)
        axs[i, 2].set_ylabel(y_label)
        axs[i, 2].set_title(titles[i] + ' ' + '$_{NMAPE}$', fontdict=title_font)

    figName = savePath + titles[0] + str_ident + '_all_variables.png'
    fig.savefig(figName)
    return


###############################################################################
# Training graphics.

def plot_loss_history(loss_history, steps_frequency , savePath):
    loss_train = np.sum(loss_history.loss_train, axis=1)
    loss_test = np.sum(loss_history.loss_test, axis=1)    
    plt.figure()
    plt.semilogy(loss_history.steps, loss_train, 'b', label="$L_{T}$", linewidth=0.8)
    plt.semilogy(loss_history.steps, loss_test, 'r', label="$L_{V}$", linewidth=0.5)
    plt.ylabel("Loss")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.yscale('log')  # Set y-axis to log scale
    figName = savePath + 'Loss_train_test' + '.png'
    plt.savefig(figName) 
    plt.show()
    return 

def plot_2_loss_history(loss_history0,loss_history1, steps_frequency , savePath):
    loss_train0 = np.sum(loss_history0.loss_train, axis=1)
    loss_train1 = np.sum(loss_history1.loss_train, axis=1)    
    plt.figure()
    plt.semilogy(loss_history0.steps, loss_train0, 'b', label="$L_{T1}$", linewidth=0.5)
    plt.semilogy(loss_history1.steps, loss_train1, 'k', label="$L_{T2}$", linewidth=0.5)
    plt.ylabel("Loss")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.yscale('log')  # Set y-axis to log scale
    figName = savePath + 'Loss_2_train' + '.png'
    plt.savefig(figName) 
    plt.show()
    return 

def plot_train_history(loss_history, steps_frequency , savePath):
    loss_train = np.sum(loss_history.loss_train, axis=1)
    loss_test = np.sum(loss_history.loss_test, axis=1)    
    plt.figure()
    plt.semilogy(loss_history.steps, loss_train, 'b', label="$L_{T}$", linewidth=0.8)
    #plt.semilogy(loss_history.steps, loss_test, 'r', label="$L_{V}$", linewidth=0.5)
    plt.ylabel("Loss")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.yscale('log')  # Set y-axis to log scale
    figName = savePath + 'Loss_train' + '.png'
    plt.savefig(figName) 
    plt.show()
    return 
       
def plot_gov_eq_loss(loss_history, title, steps_frequency , str_ident,  savePath):      
    loss_train = loss_history.loss_train
    loss_test = loss_history.loss_test       
    loss_train = np.array(loss_train)
    loss_test = np.array(loss_test)    
    #loss_train_cont = loss_train[:,0]
    #loss_test_cont = loss_test[:, 0]       
    loss_train_ns_x = loss_train[:,0]
    loss_test_ns_x = loss_test[:,0]     
    loss_train_ns_y = loss_train[:,1] 
    loss_test_ns_y = loss_test[:,1]  
    loss_train_mass_transf = loss_train[:,2] 
    loss_test_mass_transf = loss_test[:,2]     
 
        
    plt.figure()    
    # Plotting       
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)      
    #plt.semilogy(loss_history.steps[::steps_frequency], loss_train_cont[::steps_frequency], 'black', label="Continuity", linewidth=0.8)  
    # plt.semilogy(loss_history.steps, loss_test_cont, 'gray', label="Continuity test loss ", linewidth=0.5)      
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_ns_x[::steps_frequency], 'blue', label="Navier Stokes (x) ", linewidth=0.8)  
    # plt.semilogy(loss_history.steps, loss_test_ns_x, 'red', label="NS x test loss ", linewidth=0.5)      
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_ns_y[::steps_frequency], 'green', label="Navier Stokes (y) ", linewidth=0.8)  
    # plt.semilogy(loss_history.steps, loss_test_ns_y, 'purple', label="NS y test loss ", linewidth=0.5)        
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_mass_transf[::steps_frequency], 'red', label="Transport ", linewidth=0.8)  
    # plt.semilogy(loss_history.steps, loss_test_mass_transf, 'purple', label="NS y test loss ", linewidth=0.5)        
    
   
    
    plt.xlabel('Iterations')
    plt.ylabel('Governing Equations Loss') 
    #plt.title()
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 8})  
    plt.yscale('log')  # Set y-axis to log scale
    figName = savePath + str_ident + 'Loss_train_test' + '.png'
    plt.savefig(figName)
    plt.show()
    return  

def plot_gov_4_eq_loss(loss_history, title, steps_frequency , str_ident, savePath):      
    loss_train = loss_history.loss_train
    loss_test = loss_history.loss_test       
    loss_train = np.array(loss_train)
    loss_test = np.array(loss_test) 
    idx = 0 
    loss_train_cont = loss_train[:,idx]
    loss_test_cont = loss_test[:, idx]  
    idx = idx + 1      
    loss_train_ns_x = loss_train[:,idx]
    loss_test_ns_x = loss_test[:,idx]   
    idx = idx + 1      
    loss_train_ns_y = loss_train[:,idx] 
    loss_test_ns_y = loss_test[:,idx]  
    idx = idx + 1      
    loss_train_mass_transf = loss_train[:,idx] 
    loss_test_mass_transf = loss_test[:,idx]           
    plt.figure()    
    # Plotting       
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)      
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_cont[::steps_frequency], 'black', label="Continuity", linewidth=0.8)  
    # plt.semilogy(loss_history.steps, loss_test_cont, 'gray', label="Continuity test loss ", linewidth=0.5)      
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_ns_x[::steps_frequency], 'blue', label="Navier Stokes (x) ", linewidth=0.8)  
    # plt.semilogy(loss_history.steps, loss_test_ns_x, 'red', label="NS x test loss ", linewidth=0.5)      
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_ns_y[::steps_frequency], 'green', label="Navier Stokes (y) ", linewidth=0.8)  
    # plt.semilogy(loss_history.steps, loss_test_ns_y, 'purple', label="NS y test loss ", linewidth=0.5)        
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_mass_transf[::steps_frequency], 'red', label="Transpor ", linewidth=0.8)  
    # plt.semilogy(loss_history.steps, loss_test_mass_transf, 'purple', label="NS y test loss ", linewidth=0.5)          
    plt.xlabel('Epochs')
    plt.ylabel('Governing Equations Loss') 
    #plt.title()
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 8})  
    plt.yscale('log')  # Set y-axis to log scale
    figName = savePath + str_ident + 'Loss_train_test' + '.png'
    plt.savefig(figName)
    plt.show()
    return  

def plot_sup_loss(loss_history, title, steps_frequency ,num_equations , str_ident, savePath):   
    loss_train = np.array(loss_history.loss_train)
    loss_test = np.array(loss_history.loss_test)   
    idx = num_equations 
    loss_train_sup_u = loss_train[:, idx]
    loss_test_sup_u = loss_test[:, idx]  
    idx = idx + 1 
    loss_train_sup_v = loss_train[:, idx]
    loss_test_sup_v = loss_test[:, idx]     
    idx = idx + 1 
    loss_train_sup_p = loss_train[:, idx]
    loss_test_sup_p = loss_test[:, idx]   
    idx = idx + 1 
    loss_train_sup_w = loss_train[:, idx]
    loss_test_sup_w = loss_test[:, idx]    
    plt.figure()    
    # Plotting       
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)   
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_sup_u[::steps_frequency], 'blue', label="Velocity (x) ", linewidth=0.8)  
    # plt.semilogy(loss_history.steps * 1000, loss_test_sup_u, 'gray', label="Continuity test loss ", linewidth=0.5)  
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_sup_v[::steps_frequency], 'green', label="Velocity (y) ", linewidth=0.8)  
    # plt.semilogy(loss_history.steps * 1000, loss_test_sup_v, 'red', label="NS x test loss ", linewidth=0.5)  
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_sup_p[::steps_frequency], 'black', label="Pressure  ", linewidth=0.8)
    # plt.semilogy(loss_history.steps * 1000, loss_test_sup_p, 'purple', label="NS y test loss ", linewidth=0.5)    
    plt.semilogy(loss_history.steps[::steps_frequency], loss_train_sup_w[::steps_frequency], 'red', label="Concentration  ", linewidth=0.8)
    plt.xlabel('Iterations')
    plt.ylabel('Supervised Loss') 
    # plt.title(title)  # You can add a title here if needed
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 8})  
    plt.yscale('log')  # Set y-axis to log scale
    figName = savePath + str_ident + 'Loss_train_test' + '.png'
    plt.savefig(figName)
    plt.show()
    return


def interpolate_to_structured_grid(plot_x,plot_y,plot_z, plot_c, Nx=50, Ny=50, Nz=50, method='linear'):
    """
    Interpolates unstructured grid data to a structured grid.

    Parameters:
    - plot_x: 1D array of x-coordinates of unstructured grid points
    - plot_y: 1D array of y-coordinates of unstructured grid points
    - plot_z: 1D array of z-coordinates of unstructured grid points
    - plot_c: 1D array of values at the unstructured grid points
    - Nx, Ny, Nz: Number of points in each dimension for the structured grid
    - method: Interpolation method, 'linear', 'nearest', or 'cubic'

    Returns:
    - XI, YI, ZI: Structured grid coordinates
    - stru_plot_c: Interpolated values on the structured grid
        
        
    # Prepare Dataset 
    arr_data_points = [array for sub_dict in data_points.values() for array in sub_dict.values()]
    transformed_arrays = [arr.flatten().reshape(-1, 1) for arr in arr_data_points]
    data_x_reshape = transformed_arrays[0]
    data_y_reshape = transformed_arrays[1]
    data_z_reshape = transformed_arrays[2]
    
    data_c =  remove_point_mesh(data_c)  
    data_c_reshape = np.reshape(data_c, (-1, 1))  

    # Ensure the input arrays are 1D and have the same length
    plot_x = np.ravel(data_x_reshape)
    plot_y = np.ravel(data_y_reshape)
    plot_z = np.ravel(data_z_reshape)
    plot_c = np.ravel(data_c_reshape)
    """

    assert plot_x.shape == plot_y.shape == plot_z.shape == plot_c.shape, "Input arrays must have the same shape"

    # Print debug information
    #print(f"Shapes: plot_x={plot_x.shape}, plot_y={plot_y.shape}, plot_z={plot_z.shape}, plot_c={plot_c.shape}")

    # Define the structured grid
    xi = np.linspace(np.min(plot_x), np.max(plot_x), num=Nx)
    yi = np.linspace(np.min(plot_y), np.max(plot_y), num=Ny)
    zi = np.linspace(np.min(plot_z), np.max(plot_z), num=Nz)

    # Create a meshgrid for the structured grid
    XI, YI, ZI = np.meshgrid(xi, yi, zi)

    # Interpolate the data
    points = np.column_stack((plot_x, plot_y, plot_z))
    values = plot_c

    stru_plot_c = griddata(points, values, (XI, YI, ZI), method=method)
    
    plot_x =  np.reshape(XI, (-1, 1))  
    plot_y =  np.reshape(YI, (-1, 1))  
    plot_z =  np.reshape(ZI, (-1, 1))  
    plot_c =  np.reshape(stru_plot_c, (-1, 1))  

    return plot_x, plot_y, plot_z, plot_c


def generate_3D_vtk(plot_x, plot_y, plot_z, plot_c ,title, savePath, dimX,dimY,dimZ,num_columns=6):
    plot_xyz = np.hstack((plot_x, plot_y, plot_z)) 
    sorted_indices = np.lexsort((plot_xyz[:, 1], plot_xyz[:, 0], plot_xyz[:, 2]))    
    test_xyz = plot_xyz[sorted_indices]
    test_c = plot_c[sorted_indices]      
    os.makedirs(os.path.dirname(savePath), exist_ok=True)    
    min_x,min_y,min_z = min(plot_x), min(plot_y), min(plot_z)
    max_x,max_y,max_z = max(plot_x), max(plot_y), max(plot_z)    
    test_xyz = test_xyz.astype(int)

    max_scalar_length = max(len(str(scalar)) for scalar in test_c)

    with open(savePath, 'w') as vtk_file:
        vtk_file.write("# vtk DataFile Version 2.0\n")
        vtk_file.write(title + "\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET STRUCTURED_GRID\n")
        vtk_file.write("DIMENSIONS {} {} {}\n".format(dimY,dimX,dimZ ))
        vtk_file.write("POINTS {} float\n".format(len(test_xyz)))
        for point in test_xyz:
            vtk_file.write("{:<10} {:<10} {:<10}\n".format(*point))
        vtk_file.write("\n")
        vtk_file.write("POINT_DATA {}\n".format(len(test_xyz)))
        vtk_file.write("SCALARS Scalars float\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        
        num_values = len(test_c)
        num_rows = -(-num_values // num_columns)
        for i in range(num_rows):
            for j in range(num_columns):
                idx = i * num_columns + j
                if idx < num_values:
                    vtk_file.write("{:<{width}}".format(str(test_c[idx]).strip("[]"), width=max_scalar_length + 1))
            vtk_file.write("\n")           
    return

def generate_unstructured_3D_vtk(plot_x, plot_y, plot_z, plot_c, title, savePath, dimX, dimY, dimZ, num_columns=6):
    plot_xyz = np.column_stack((plot_x, plot_y, plot_z)) 
    sorted_indices = np.lexsort((plot_xyz[:, 1], plot_xyz[:, 0], plot_xyz[:, 2]))    
    sorted_xyz = plot_xyz[sorted_indices]
    sorted_c = plot_c[sorted_indices]      
    os.makedirs(os.path.dirname(savePath), exist_ok=True)    
    
    max_scalar_length = max(len(str(scalar)) for scalar in sorted_c)

    with open(savePath, 'w') as vtk_file:
        vtk_file.write("# vtk DataFile Version 2.0\n")
        vtk_file.write(title + "\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")
        vtk_file.write("POINTS {} float\n".format(len(sorted_xyz)))
        for point in sorted_xyz:
            vtk_file.write("{:<10} {:<10} {:<10}\n".format(*point))
        vtk_file.write("\n")
        vtk_file.write("CELLS {} {}\n".format(len(sorted_xyz), 2*len(sorted_xyz)))
        for i in range(len(sorted_xyz)):
            vtk_file.write("1 {:<10}\n".format(i))
        vtk_file.write("\n")
        vtk_file.write("CELL_TYPES {}\n".format(len(sorted_xyz)))
        for _ in range(len(sorted_xyz)):
            vtk_file.write("1\n")
        vtk_file.write("\n")
        vtk_file.write("POINT_DATA {}\n".format(len(sorted_xyz)))
        vtk_file.write("SCALARS Scalars float\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        
        num_values = len(sorted_c)
        num_rows = -(-num_values // num_columns)
        for i in range(num_rows):
            for j in range(num_columns):
                idx = i * num_columns + j
                if idx < num_values:
                    vtk_file.write("{:<{width}}".format(str(sorted_c[idx]).strip("[]"), width=max_scalar_length + 1))
            vtk_file.write("\n")  


def percentage_between(array, value_min, valeu_max): 
    count_between = sum(1 for element in array if ((element > value_min) and (element < valeu_max)))
    percentage = (count_between / len(array)) * 100.0    
    return percentage

def calculate_percentage_in_ranges(array, min_value, max_value, increment=0.25):
    percentage_ranges = {}
    for value in np.arange(min_value, max_value + increment, increment):
        value_min = value
        value_max = value + increment
        percentage = percentage_between(array, value_min, value_max)
        percentage_ranges[(value_min, value_max)] = percentage
    return [(value[0], value[1], percentage) for value, percentage in percentage_ranges.items()]

def plot_error_dist(sre_list, titles,  str_ident, savePath):
    fig, ax = plt.subplots(figsize=(7, 4))
    sre = sre_list[0]                  
    perc_ranges = calculate_percentage_in_ranges(sre, -5, 5, 0.25) 
    bar_width = 0.20
    bar_positions = [(value_min + value_max) / 2 for value_min, value_max, _ in perc_ranges]
    percentages = [perc for _, _, perc in perc_ranges]
    ax.bar(bar_positions, percentages, width=bar_width, color='black')  
    ax.set_xlabel('NMAPE', fontsize=18) 
    ax.set_ylabel('$y_{PINN}$ (%)', fontsize=18) 
    ax.set_xticks(np.arange(-5, 5, 1.0))
    ax.tick_params(axis='x', labelsize=14)  
    ax.tick_params(axis='y', labelsize=14)  
    figName = savePath + titles[0] + str_ident + 'ed.png'
    fig.savefig(figName)     
    return

def plot_loss(loss_history, savePath, title='loss_history.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='black', label='$L_{T}$')
    plt.ylabel('Total Loss', fontsize=18)  
    plt.xlabel('Number of Iterations', fontsize=18)  
    plt.legend(fontsize=18) 
    plt.grid(False)
    plt.yscale('log')
    figName = savePath + title
    plt.savefig(figName)
    plt.show()
    return

