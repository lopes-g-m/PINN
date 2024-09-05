import numpy as np
from matplotlib import artist
import matplotlib.axes as axes
from matplotlib import axes
import matplotlib.pyplot as plt 
import matplotlib.tri as tri 
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker


def plot_2d_domains(data_sets, titles, x_label, y_label, str_ident, savePath):
    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
    fig.subplots_adjust(hspace=0.5)
    for i, (ob_x, ob_y, ob_u) in enumerate(data_sets):       
        max_value = max(ob_u)
        min_value = min(ob_u)
        plot_x = ob_x.flatten()
        plot_y = ob_y.flatten()
        plot_u = ob_u.flatten()
        plot_x, plot_y = plot_x/10, plot_y/10   # mm 
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

        ticks_arr = np.linspace(min_value, max_value, 5).flatten()         
        cbar = fig.colorbar(tcf, ax=ax, location="right", format='%.1e', ticks=ticks_arr, shrink=0.8)  # Adjust the shrink parameter
        cbar.ax.tick_params(labelsize=18)          
        ax.tricontour(triang, plot_u, linestyles="solid", colors="k", linewidths=0)
        ax.set_title(titles[i], fontsize=25)  
        ax.set_xlabel(x_label, fontsize=18)  
        ax.set_ylabel(y_label, fontsize=18)  
        ax.tick_params(axis='both', which='major', labelsize=18) 
    figName = savePath + str_ident + '_combined.png'
    fig.savefig(figName)
    plt.show()
    return


def plot_2d_domains_bump(data_sets, titles, x_label, y_label, str_ident, savePath, x_lim_inf, y_lim_inf):
    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
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
        ticks_arr = np.linspace(min_value, max_value, 5).flatten()          
        cbar = fig.colorbar(tcf, ax=ax, location="right", format='%.1e', ticks=ticks_arr, shrink=0.8)
        cbar.ax.tick_params(labelsize=18)
        ax.tricontour(triang, plot_u, linestyles="solid", colors="k", linewidths=0)
        ax.set_title(titles[i], fontsize=25)  
        ax.set_xlabel(x_label, fontsize=18) 
        ax.set_ylabel(y_label, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18) 
    figName = savePath + str_ident + '_combined.png'
    fig.savefig(figName)
    plt.show()
    return

def plot_data_distribution(train_x, train_y, train_u, train_v, train_p, train_c, title, str_ident, savePath):
    var_input_train = np.concatenate((train_x, train_y), axis=1)
    var_output_train = np.concatenate((train_u, train_v, train_p, train_c), axis=1)
    x_labels = ['X', 'Y']
    fig, ax = plt.subplots()
    plt.plot(var_input_train.T, 'ob', fillstyle='full', markersize=6, alpha=1)
    plt.grid(':', linewidth=1)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation='vertical')
    plt.title(title + ' - Input Distribution')
    plt.tight_layout()
    plt.show()
    figName = savePath + 'Input_dist_' + str_ident + '.png'
    fig.savefig(figName)
    x_labels = ['U', 'V', 'P', 'C']
    fig, ax = plt.subplots()
    plt.plot(var_output_train.T, 'ob', fillstyle='full', markersize=6, alpha=1)
    plt.grid(':', linewidth=1)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation='vertical')
    plt.title(title + ' - Output Distribution')
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
    
def sinal_relative_error(pred, ref):
      diff = pred - ref
      rel_diff = diff / (max(ref) - min(ref))
      rel_err = rel_diff * 100
      return rel_err  
    
def percentage_between(array, value_min, valeu_max): 
    count_between = sum(1 for element in array if ((element > value_min) and (element < valeu_max)))
    percentage = (count_between / len(array)) * 100.0    
    return percentage

def calculate_percentage_in_ranges(array, min_value, max_value):
    percentage_ranges = {}
    for value in range(min_value-1, max_value):
        value_min = value
        value_max = value + 1
        percentage = percentage_between(array, value_min, value_max)
        percentage_ranges[(value_min, value_max)] = percentage
    return [(value[0], value[1], percentage) for value, percentage in percentage_ranges.items()]

def av_rel_error(pred, ref):
      rel_error = relative_error(pred, ref)
      av_rel_error = np.mean(rel_error)
      return av_rel_error 
  
def plot_3_2d_domains(ob_x_list, ob_y_list, ob_u_list, ob_v_list, titles, x_label, y_label, unit, str_ident, savePath):
    num_plots = len(ob_x_list)

    fig, axs = plt.subplots(num_plots, 3, figsize=(16 * 3, 5 * num_plots)) 
    for i in range(num_plots):
        ob_x, ob_y, ob_u, ob_v = ob_x_list[i], ob_y_list[i], ob_u_list[i], ob_v_list[i]
        ob_x, ob_y = ob_x/10, ob_y/10 # mm       
        ob_uv = np.concatenate((ob_u, ob_v))
        max_value = max([ob_uv.max(), ob_v.max()])
        min_value = min([ob_uv.min(), ob_v.min()])
        ob_re = relative_error(ob_v, ob_u)
        sre = sinal_relative_error(ob_v, ob_u) 
        max_re = ob_re.max()
        min_re = ob_re.min()        
        plot_x = ob_x.flatten()
        plot_y = ob_y.flatten()
        plot_x, plot_y = plot_x/10, plot_y/10        
        axs[i, 0].set_aspect("equal") 
        axs[i, 1].set_aspect("equal") 
        axs[i, 2].set_aspect("equal")   

        ticks_arr = np.linspace(min_value, max_value).flatten()    
        ticks_re = np.linspace(min_re, max_re).flatten()

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

        
        max_ticks = 5        
        cbar_u = fig.colorbar(tcf_u, ax=axs[i, 0], shrink=0.7, format='%.1e', orientation='vertical')        
        cbar_u.ax.yaxis.set_major_locator(ticker.MaxNLocator(max_ticks))
        cbar_u.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_u.ax.xaxis.set_ticks_position('bottom')
        cbar_u.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_u.ax.xaxis.set_label_text(unit)
        label = cbar_u.ax.xaxis.get_label()
        label.set_rotation(0)
        cbar_u.ax.tick_params(labelsize=22)

        cbar_v = fig.colorbar(tcf_v, ax=axs[i, 1], shrink=0.7, format='%.1e', orientation='vertical')
        cbar_v.ax.yaxis.set_major_locator(ticker.MaxNLocator(max_ticks))
        cbar_v.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_v.ax.xaxis.set_ticks_position('bottom')
        cbar_v.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_v.ax.xaxis.set_label_text(unit)
        label = cbar_v.ax.xaxis.get_label()
        label.set_rotation(0)
        cbar_v.ax.tick_params(labelsize=22)
        
        cbar_re = fig.colorbar(tcf_re, ax=axs[i, 2], shrink=0.7, format='%02d', orientation='vertical')
        cbar_re.ax.yaxis.set_major_locator(ticker.MaxNLocator(max_ticks))
        cbar_re.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_re.ax.xaxis.set_ticks_position('bottom')
        cbar_re.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_re.ax.xaxis.set_label_text("")
        label = cbar_re.ax.xaxis.get_label()
        label.set_rotation(0)
        cbar_re.ax.tick_params(labelsize=22)

        axs[i, 0].tricontour(tri.Triangulation(plot_x, plot_y), ob_u.flatten(), linestyles="solid", colors="k",
                             linewidths=0)
        axs[i, 1].tricontour(tri.Triangulation(plot_x, plot_y), ob_v.flatten(), linestyles="solid", colors="k",
                             linewidths=0)
        axs[i, 2].tricontour(tri.Triangulation(plot_x, plot_y), ob_re.flatten(), linestyles="solid", colors="k",
                             linewidths=0)

        title_font = {'size': '30'}  

        axs[i, 0].set_xlabel(x_label, fontsize=22)  
        axs[i, 0].set_ylabel(y_label, fontsize=22)  
        axs[i, 0].set_title(titles[i] + ' ' + '$_{CFD}$', fontdict=title_font)
        axs[i, 0].tick_params(axis='both', which='major', labelsize=22)  

        axs[i, 1].set_xlabel(x_label, fontsize=22) 
        axs[i, 1].set_ylabel(y_label, fontsize=22)  
        axs[i, 1].set_title(titles[i] + ' ' + '$_{PINN}$', fontdict=title_font)
        axs[i, 1].tick_params(axis='both', which='major', labelsize=22)  

        axs[i, 2].set_xlabel(x_label, fontsize=22)  
        axs[i, 2].set_ylabel(y_label, fontsize=22)  
        axs[i, 2].set_title(titles[i] + ' ' + '$_{NMAPE}$', fontdict=title_font)
        axs[i, 2].tick_params(axis='both', which='major', labelsize=22) 

    figName = savePath + titles[0] + str_ident + '_all_variables.png'
    fig.savefig(figName)
    return

def plot_3_2d_domains_bump(ob_x_list, ob_y_list, ob_u_list, ob_v_list, titles, x_label, y_label, unit, str_ident, x_lim_inf, y_lim_inf, savePath):
    num_plots = len(ob_x_list)
    fig, axs = plt.subplots(num_plots, 3, figsize=(14 * 3, 4 * num_plots))
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

        for ax in axs[i]:
            ax.plot(x_lim_inf, y_lim_inf, color='black', label=None)
            ax.fill_between(x_lim_inf, y_lim_inf, alpha=0.9, color='black', label=None)
            ax.tricontour(tri.Triangulation(plot_x, plot_y), ob_u.flatten(), linestyles="solid", colors="k", linewidths=0)
            ax.tricontour(tri.Triangulation(plot_x, plot_y), ob_v.flatten(), linestyles="solid", colors="k", linewidths=0)
            ax.tricontour(tri.Triangulation(plot_x, plot_y), ob_re.flatten(), linestyles="solid", colors="k", linewidths=0)

        max_ticks = 5        
        cbar_u = fig.colorbar(tcf_u, ax=axs[i, 0], shrink=0.7, format='%.1e', orientation='vertical')        
        cbar_u.ax.yaxis.set_major_locator(ticker.MaxNLocator(max_ticks))
        cbar_u.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_u.ax.xaxis.set_ticks_position('bottom')
        cbar_u.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_u.ax.xaxis.set_label_text(unit)
        label = cbar_u.ax.xaxis.get_label()
        label.set_rotation(0)
        cbar_u.ax.tick_params(labelsize=22)

        cbar_v = fig.colorbar(tcf_v, ax=axs[i, 1], shrink=0.7, format='%.1e', orientation='vertical')
        cbar_v.ax.yaxis.set_major_locator(ticker.MaxNLocator(max_ticks))
        cbar_v.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_v.ax.xaxis.set_ticks_position('bottom')
        cbar_v.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_v.ax.xaxis.set_label_text(unit)
        label = cbar_v.ax.xaxis.get_label()
        label.set_rotation(0)
        cbar_v.ax.tick_params(labelsize=22)
        
        cbar_re = fig.colorbar(tcf_re, ax=axs[i, 2], shrink=0.7, format='%02d', orientation='vertical')
        cbar_re.ax.yaxis.set_major_locator(ticker.MaxNLocator(max_ticks))
        cbar_re.ax.get_yaxis().set_label_coords(0, 0.5)
        cbar_re.ax.xaxis.set_ticks_position('bottom')
        cbar_re.ax.xaxis.set_label_coords(1.5, -0.15)
        cbar_re.ax.xaxis.set_label_text("")
        label = cbar_re.ax.xaxis.get_label()
        label.set_rotation(0)
        cbar_re.ax.tick_params(labelsize=22)

        title_font = {'size': '25'}

        axs[i, 0].tricontour(tri.Triangulation(plot_x, plot_y), ob_u.flatten(), linestyles="solid", colors="k",
                             linewidths=0)
        axs[i, 1].tricontour(tri.Triangulation(plot_x, plot_y), ob_v.flatten(), linestyles="solid", colors="k",
                             linewidths=0)
        axs[i, 2].tricontour(tri.Triangulation(plot_x, plot_y), ob_re.flatten(), linestyles="solid", colors="k",
                             linewidths=0)

        title_font = {'size': '30'}  

        axs[i, 0].set_xlabel(x_label, fontsize=22)  
        axs[i, 0].set_ylabel(y_label, fontsize=22)  
        axs[i, 0].set_title(titles[i] + ' ' + '$_{CFD}$', fontdict=title_font)
        axs[i, 0].tick_params(axis='both', which='major', labelsize=22)  

        axs[i, 1].set_xlabel(x_label, fontsize=22)  
        axs[i, 1].set_ylabel(y_label, fontsize=22)  
        axs[i, 1].set_title(titles[i] + ' ' + '$_{PINN}$', fontdict=title_font)
        axs[i, 1].tick_params(axis='both', which='major', labelsize=22) 

        axs[i, 2].set_xlabel(x_label, fontsize=22)  
        axs[i, 2].set_ylabel(y_label, fontsize=22)  
        axs[i, 2].set_title(titles[i] + ' ' + '$_{NMAPE}$', fontdict=title_font)
        axs[i, 2].tick_params(axis='both', which='major', labelsize=22) 

    figName = savePath + titles[0] + str_ident + '_all_variables.png'
    fig.savefig(figName)
    plt.show()
    return

def plot_percentage_histogram(data, ax):
    perc_ranges = calculate_percentage_in_ranges(data, -10, 10)
    bins = np.arange(-11, 11, 1)
    percentages = [perc for _, _, perc in perc_ranges]
    ax.bar(bins[:-1], percentages, width=1, color='blue')        
    ax.set_xlabel('NMAPE')     
    ax.set_xticks(np.linspace(-10 , 10, 11))
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Histogram')
    ax.grid(False)
    return 

def plot_error_dist(ob_x_list, ob_y_list,sre_list, titles,  str_ident, savePath):
    num_plots = len(ob_x_list)
    fig, axs = plt.subplots(1, num_plots, figsize=( 7 * num_plots ,  4))
    for i in range(num_plots):   
        sre = sre_list[i]                  
        perc_ranges = calculate_percentage_in_ranges(sre, -5, 5)
        bins = np.arange(-6, 6, 1)
        percentages = [perc for _, _, perc in perc_ranges]
        axs[i].bar(bins[:-1], percentages, width=1, color='blue')        
        axs[i].set_xlabel('NMAPE')     
        axs[i].set_xticks(np.linspace(-5 , 5, 11)) 
        axs[i].set_xlim([-5.5, 5.5]) 
        axs[i].set_ylabel('$y_{PINN}$ (%) ')
        axs[i].set_title(titles[i] + '$_{Dist}$')
        axs[i].grid(False) 
    figName = savePath + titles[0] + str_ident + '4_all_variables.png'
    fig.savefig(figName)       
    return

def plot_points(load_x, load_y, title1, train_x, train_y, title2, test_x, test_y, title3, savePath):
    # Load the data and set up the plot
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    # Plot the data for each subplot
    axs[0].scatter(load_x, load_y, color='black', marker='o', s=1)
    axs[0].set_title(title1, fontsize=18)  
    axs[0].set_xlabel('x', fontsize=18)  
    axs[0].set_ylabel('y', fontsize=18)  
    axs[0].tick_params(axis='both', which='major', labelsize=18)  
    axs[0].set_xlim(min(load_x), max(load_x))
    axs[0].set_ylim(min(load_y), max(load_y))
    axs[1].scatter(train_x, train_y, color='black', marker='o', s=1)
    axs[1].set_title(title2, fontsize=18)
    axs[1].set_xlabel('x', fontsize=18)
    axs[1].set_ylabel('y', fontsize=18)
    axs[1].tick_params(axis='both', which='major', labelsize=18)
    axs[1].set_xlim(min(train_x), max(train_x))
    axs[1].set_ylim(min(train_y), max(train_y))
    axs[2].scatter(test_x, test_y, color='black', marker='o', s=1)
    axs[2].set_title(title3, fontsize=18)
    axs[2].set_xlabel('x', fontsize=18)
    axs[2].set_ylabel('y', fontsize=18)
    axs[2].tick_params(axis='both', which='major', labelsize=18)
    axs[2].set_xlim(min(test_x), max(test_x))
    axs[2].set_ylim(min(test_y), max(test_y))

    plt.tight_layout()
    plt.savefig(savePath)
    plt.show()
    
    figName = savePath + 'points.png'
    fig.savefig(figName)
    
    return


def plot_points_bump(load_x, load_y, title1, train_x, train_y, title2, test_x, test_y, title3, x_lim_inf, y_lim_inf, savePath):
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
    axs[0].scatter(load_x, load_y, color='black', marker='o', s=1)
    axs[0].set_title(title1, fontsize=25) 
    axs[0].set_xlabel('x', fontsize=18)  
    axs[0].set_ylabel('y', fontsize=18)  
    axs[0].tick_params(axis='both', which='major', labelsize=18) 
    axs[0].set_xlim(min(load_x), max(load_x))
    axs[0].set_ylim(min(load_y), max(load_y))
    axs[0].plot(x_lim_inf, y_lim_inf, color='black', label=None)
    axs[0].fill_between(x_lim_inf, y_lim_inf, alpha=1.0, color='black', label=None)
    
    axs[1].scatter(train_x, train_y, color='black', marker='o', s=1)
    axs[1].set_title(title2, fontsize=25)
    axs[1].set_xlabel('x', fontsize=18)
    axs[1].set_ylabel('y', fontsize=18)
    axs[1].tick_params(axis='both', which='major', labelsize=18)  
    axs[1].set_xlim(min(train_x), max(train_x))
    axs[1].set_ylim(min(train_y), max(train_y))
    axs[1].plot(x_lim_inf, y_lim_inf, color='black', label=None)
    axs[1].fill_between(x_lim_inf, y_lim_inf, alpha=1.0, color='black', label=None)
    
    axs[2].scatter(test_x, test_y, color='black', marker='o', s=1)
    axs[2].set_title(title3, fontsize=25)
    axs[2].set_xlabel('x', fontsize=18)
    axs[2].set_ylabel('y', fontsize=18)
    axs[2].tick_params(axis='both', which='major', labelsize=18)  
    axs[2].set_xlim(min(test_x), max(test_x))
    axs[2].set_ylim(min(test_y), max(test_y))
    axs[2].plot(x_lim_inf, y_lim_inf, color='black', label=None)
    axs[2].fill_between(x_lim_inf, y_lim_inf, alpha=1.0, color='black', label=None)    
    plt.tight_layout()
    plt.savefig(savePath)
    plt.show()
    figName = savePath + 'points'  + '.png'
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
    plt.yscale('log')  
    figName = savePath + 'Loss_train_test' + '.png'
    plt.savefig(figName) 
    plt.show()
    return 

def plot_train_history(loss_history, steps_frequency , savePath):
    loss_train = np.sum(loss_history.loss_train, axis=1)
    loss_test = np.sum(loss_history.loss_test, axis=1)    
    plt.figure()
    plt.semilogy(loss_history.steps, loss_train, 'b', label="$L_{T}$", linewidth=0.8)
    plt.ylabel("Loss")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.yscale('log')  
    figName = savePath + 'Loss_train' + '.png'
    plt.savefig(figName) 
    plt.show()
    return 

def plot_comparison_3_sup_loss(loss_history_1, title_1, loss_history_2, title_2, loss_history_3, title_3, title, steps_frequency, str_ident, savePath):
    loss_train_1 = np.sum(loss_history_1.loss_train, axis=1)
    loss_train_2 = np.sum(loss_history_2.loss_train, axis=1)
    loss_train_3 = np.sum(loss_history_3.loss_train, axis=1)       
    plt.semilogy(loss_history_1.steps[::steps_frequency], loss_train_1[::steps_frequency], 'red', label=title_1, linewidth=0.8)  
    plt.semilogy(loss_history_2.steps[::steps_frequency], loss_train_2[::steps_frequency], 'blue', label=title_2, linewidth=0.8)  
    plt.semilogy(loss_history_3.steps[::steps_frequency], loss_train_3[::steps_frequency], 'black', label=title_3, linewidth=0.8)  
    plt.xlabel('Number of Iterations', fontsize=18)  
    plt.ylabel('Total Loss', fontsize=18)  
    plt.title(title, fontsize=25)  
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 18})  
    plt.tick_params(axis='both', which='major', labelsize=18)  
    plt.yscale('log')  
    plt.tight_layout() 
    figName = savePath + str_ident + 'Sup_Comp' + '.png'
    plt.savefig(figName)
    plt.show()   
    return