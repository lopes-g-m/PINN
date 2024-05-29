import numpy as np
from matplotlib import artist
import matplotlib.axes as axes
from matplotlib import axes
import matplotlib.pyplot as plt 
import matplotlib.tri as tri 

def plot_2d_domains(data_sets, titles, x_label, y_label, str_ident, savePath):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.3)
    for i, (ob_x, ob_y, ob_u) in enumerate(data_sets):
        max_value = max(ob_u)
        min_value = min(ob_u)
        plot_x = ob_x.flatten()
        plot_y = ob_y.flatten()
        plot_u = ob_u.flatten()
        triang = tri.Triangulation(plot_x, plot_y) 
        ax = axs[i]
        ax.set_aspect("equal")
        tcf = ax.tricontourf(
            triang,
            plot_u,
            levels=100,
            cmap="jet",
            vmin=min_value,
            vmax=max_value )

        ticks_arr = np.linspace(min_value, max_value, 8).flatten()      
        
        fig.colorbar(tcf, ax=ax, location="right", format='%.1e', ticks=ticks_arr, shrink=0.8)

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
    # plot output variable
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

    fig, axs = plt.subplots(num_plots, 3, figsize=(6 * 3, 5 * num_plots))

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
        axs[i, 1].set_title(titles[i] + ' ' + '$_{PINN}$', fontdict=title_font)

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
        axs[i, 1].set_title(titles[i] + ' ' + '$_{PINN}$', fontdict=title_font)

        axs[i, 2].set_xlabel(x_label)
        axs[i, 2].set_ylabel(y_label)
        axs[i, 2].set_title(titles[i] + ' ' + '$_{NMAPE}$', fontdict=title_font)

    figName = savePath + titles[0] + str_ident + '_all_variables.png'
    fig.savefig(figName)
    return

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
    for value in range(min_value, max_value + 1):
        value_min = value
        value_max = value + 1
        percentage = percentage_between(array, value_min, value_max)
        percentage_ranges[(value_min, value_max)] = percentage
    return [(value[0], value[1], percentage) for value, percentage in percentage_ranges.items()]

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
    