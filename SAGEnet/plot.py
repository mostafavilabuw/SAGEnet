import pandas as pd
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import os
import seaborn as sb
from sklearn.metrics import r2_score
import tangermeme.plot

def plot_attribs(arr,title='',annotations=None,center_line=False,save_dir=None,save_name='',shorten_motif_name=0,fig_width=15): 
    """
    Plot model attributions, optionally labeled with annotations. Arr is attribution array of shape (4, seq_lenth). 
    """
    plt.figure(figsize=(fig_width, 1))
    ax = plt.subplot(111)
    tangermeme.plot.plot_logo(arr,ax=ax)
    plt.title(title)
    abs_max = np.abs(arr).max()
    ax.set_ylim(-abs_max, abs_max)
    
    # plot annotations
    if annotations is not None: 
        for i in range(len(annotations)): 
            curr_annotation = annotations.iloc[i]
            plt.hlines(y=0, xmin=curr_annotation['start'], xmax=curr_annotation['end'], color='purple', linewidth=2)
            if shorten_motif_name:
                motif_label = curr_annotation['motif_match'].split('.')[0]
            else: 
                motif_label=curr_annotation['motif_match']
            plt.text(x=(curr_annotation['start']+curr_annotation['end'])//2, y=-1.25*abs_max, s=motif_label, color='purple', ha='center', fontsize=10)
    
    if center_line:
        x_range = ax.get_xlim()  
        x_mid = (x_range[0] + x_range[1]) // 2  
        plt.axvline(x=x_mid, color='black', linestyle='--', linewidth=1)

    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    if save_dir: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf',format='pdf', dpi=300,bbox_inches='tight')   
    else: 
        plt.show()


def plot_heatmap(arr, center_line=False, save_dir=None,save_name=''): 
    """
    Plot heatmap of model attributions. Arr is attribution array of shape (4, seq_lenth). 
    """
    num_columns = arr.shape[1]
    plt.figure(figsize=(15, 0.5))
    ax = plt.subplot(111)

    abs_max = np.abs(arr).max()
    heatmap = ax.imshow(arr, aspect='auto', cmap='RdBu_r', interpolation='nearest', vmin=-abs_max, vmax=abs_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["A", "C", "G", "T"], fontsize=8)

    if center_line:
        x_range = ax.get_xlim()  
        x_mid = (x_range[0] + x_range[1]) // 2  
        plt.axvline(x=x_mid, color='black', linestyle='--', linewidth=1)
        
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    if save_dir: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf',format='pdf', dpi=300,bbox_inches='tight')   
    else: 
        plt.show()


def line_plot_compare(arra, xlabels, arrb=None, x=None,
                      arra_t='', arrb_t='',
                      xlabel='', ylabel='', title='',
                      shade_std=True,
                      fig_width=6, fig_height=6,
                      legend_x=0.7, legend_y=0.95,
                      save_dir=None, save_name=None,
                      plot_y0=True, include_line=True):
    """
    Plot line plot of array (and optional additional array). Each array can be 1D or 2D:
    - If 2D, the first dimension determines the number of x-ticks,
      and the second dimension is plotted as median ± std.
    """
    plt.rcParams.update({'font.size': 14})  # Set global font size
    plt.clf()
    
    if save_name is None: 
        save_name = title

    plt.figure(figsize=(fig_width, fig_height))
    if x is None:
        x = xlabels

    if len(arra.shape) == 1:
        shade_std = False
        arra_mean = arra
        if arrb is not None:
            arrb_mean = arrb
    else:
        arra_mean = np.nanmedian(arra, axis=1)
        if arrb is not None:
            arrb_mean = np.nanmedian(arrb, axis=1)

    if shade_std:
        arra_std = np.nanstd(arra, axis=1)
        plt.fill_between(x, arra_mean - arra_std, arra_mean + arra_std,
                         color="#1f77b4", alpha=0.2)
        if arrb is not None:
            arrb_std = np.nanstd(arrb, axis=1)
            plt.fill_between(x, arrb_mean - arrb_std, arrb_mean + arrb_std,
                             color="#ff7f0e", alpha=0.2)

    if include_line:
        plt.plot(x, arra_mean, label=arra_t, color="#1f77b4")
    plt.scatter(x, arra_mean, color="#1f77b4", zorder=3)

    if arrb is not None:
        if include_line:
            plt.plot(x, arrb_mean, label=arrb_t, color="#ff7f0e")
        plt.scatter(x, arrb_mean, color="#ff7f0e", zorder=3)

    plt.xticks(ticks=x, labels=xlabels)
    if plot_y0:
        plt.axhline(y=0, color='black', linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(legend_x, legend_y), frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(title)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    
def sb_plot(res_df, xlabel='', ylabel='', title='', 
                     save_dir=None, save_name=None,
                     plot_type='box',  # options: 'box', 'bar', 'violin'
                     scatter=True, dot_size=5, dot_alpha=0.1, 
                     custom_palette=None, long_format=False,
                     x_col='Group', y_col='Pearson R',
                     split_by_col_val=False,
                     fontsize=16, ax=None,
                     fig_width=11, fig_height=6,
                     save_fig=True,
                     label_avgs=False,
                     label_x_offset=None, 
                     label_y_offset=None,
                     nan_to_zero=True,
                     hue=None,
                     title_pad=0):
    """
    Combined seaborn plot function for box, bar, violin, and scatter plots.

    Parameters:
    - res_df: pandas DataFrame.
    - long_format: if True, use x_col and y_col; else assume wide format.
    - plot_type: 'box', 'bar', or 'violin'
    - scatter: if True, overlay stripplot
    - split_by_col_val: whether to plot using x_col/y_col when not in long_format
    - ax: if provided, plot on this axis
    - save_dir: directory to save figure
    - save_name: name of the saved file (without extension)
    - save_fig: if True and save_dir is provided, saves figure as PDF
    """
    
    if nan_to_zero:
        res_df = res_df.replace([np.nan, np.inf, -np.inf], 0)

    if custom_palette is None:
        custom_palette = 'colorblind'
    if save_name is None:
        save_name = title

    if ax is None:
        plt.clf()
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if long_format:
        plot_df = res_df.dropna(subset=[y_col])
    else:
        plot_df = res_df.dropna(how='any')
        if not split_by_col_val:
            plot_df = plot_df.fillna(0)

    # main plot type
    if plot_type == 'box':
        if long_format or split_by_col_val:
            sb.boxplot(x=x_col, y=y_col, data=plot_df, whis=np.inf, width=0.8,
                       palette=custom_palette, ax=ax,hue=hue)
        else:
            sb.boxplot(data=plot_df, whis=np.inf, width=0.8,
                       palette=custom_palette, ax=ax,hue=hue)

    elif plot_type == 'bar':
        if long_format or split_by_col_val:
            sb.barplot(x=x_col, y=y_col, data=plot_df, width=0.8,
                       palette=custom_palette, ax=ax)
        else:
            sb.barplot(data=plot_df, width=0.8,
                       palette=custom_palette, ax=ax)

    elif plot_type == 'violin':
        if long_format or split_by_col_val:
            sb.violinplot(x=x_col, y=y_col, data=plot_df,
                          palette=custom_palette, width=0.8, ax=ax)
        else:
            sb.violinplot(data=plot_df, palette=custom_palette,
                          width=0.8, ax=ax)

    if scatter:
        non_na_plot_df = plot_df.dropna(subset=[y_col]) if (long_format or split_by_col_val) else plot_df.dropna(how='any')

        palette_args = {'palette': 'colorblind'} if len(non_na_plot_df) == 1 else {}

        if long_format or split_by_col_val:
            sb.stripplot(x=x_col, y=y_col, data=non_na_plot_df, color='black',
                        size=dot_size, jitter=True, alpha=dot_alpha, ax=ax, **palette_args)
        else:
            sb.stripplot(data=non_na_plot_df, color='black',
                        size=dot_size, jitter=True, alpha=dot_alpha, ax=ax, **palette_args)
            
    x_min, x_max = ax.get_xlim()
    buffer = 0.1
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.axhline(0, color='black', linewidth=1)

    ax.set_title(title, fontsize=fontsize,pad=title_pad)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    sb.despine(ax=ax)

    if label_avgs: 

        if label_x_offset is None: 
            if plot_type=='box':
                label_x_offset=0
            else:
                label_x_offset=0.14
        
        if label_y_offset is None: 
            label_y_offset=0

        if plot_type == 'box' or plot_type=='violin' or plot_type=='bar':
            if long_format or split_by_col_val:
                if plot_type=='bar':
                    group_avgs = plot_df.groupby(x_col)[y_col].mean()
                    #print(f'means={group_avgs}')
                else:
                    group_avgs = plot_df.groupby(x_col)[y_col].median()
                    #print(f'medians={group_avgs}')
        
                xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
                for i, cat in enumerate(xtick_labels):
                    avg_val = group_avgs[cat]
                    ax.text(i + label_x_offset, avg_val+label_y_offset, f'{avg_val:.4f}',
                            ha='center', va='bottom', fontsize=fontsize * 0.8)

            else:
                for i, col in enumerate(plot_df.columns):
                    if plot_type=='bar':
                        avg_val = plot_df[col].mean()
                    else:
                        avg_val = plot_df[col].median()
                    ax.text(i+label_x_offset, avg_val+label_y_offset,
                            f'{avg_val:.4f}', ha='center', va='bottom', fontsize=fontsize * 0.8)

    if save_dir and save_fig:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f'{save_dir}{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    elif ax is None:
        plt.show()

    legend = ax.get_legend()
    if legend:
        legend.set_frame_on(False)
        legend.set_title(None)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    return ax

        
def boxplot_compare(arra, arrb, arra_t='', arrb_t='', save_dir=None, title='',
                    stat_test='ks', ylabel='', fig_width=7, fig_height=6,
                    include_stat_test=True, save_name=None, fontsize=16):
    """
    Plot two arrays as boxplots. Each array should be 1D.
    """
    plt.clf()
    if save_name is None: 
        save_name = title

    fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 

    # Label text with n, mean, median
    arra_t = f"{arra_t} \n n={len(arra)} \n mean={np.mean(arra):.2f} \n median={np.median(arra):.2f}"
    arrb_t = f"{arrb_t} \n n={len(arrb)} \n mean={np.mean(arrb):.2f} \n median={np.median(arrb):.2f}"

    # Plot boxplots
    ax.boxplot([arra, arrb], showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([arra_t, arrb_t], fontsize=fontsize)

    # Statistical test
    if include_stat_test: 
        if stat_test == 'wilcoxon':
            stat, p_value = scipy.stats.wilcoxon(arra, arrb)
            stat_test_label = 'Wilcoxon'
        elif stat_test == 'ks':
            stat, p_value = scipy.stats.ks_2samp(arra, arrb)
            stat_test_label = 'K-S'

        if title != '':
            ax.set_title(f"{title} \n {stat_test_label} p={p_value:.2e}", fontsize=fontsize)
        else:
            ax.set_title(f"{stat_test_label} p={p_value:.2e}", fontsize=fontsize)
    else:
        if title != '':
            ax.set_title(title, fontsize=fontsize)

    # Axis label
    ax.set_ylabel(ylabel, fontsize=fontsize)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tick font size
    ax.tick_params(axis='y', labelsize=fontsize)

    # Layout and save/show
    plt.tight_layout()
    if save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')   
    else: 
        plt.show()


def scatter_compare(arra, arrb, arra_t='', arrb_t='', save_dir=None, title='', 
                    add_identity_line=False, nan_to_zero=True, save_name=None, 
                    plot_density=True, axes_lines=0, names=None, 
                    color_arr=None, discrete_color_labels=None, hline_y=None, 
                    colorbar_label=None, auto_title=True, s=30, alpha=.7, fontsize=16,
                    fig_width=11, fig_height=6, ax=None, highlight_pts=None):
    """
    Plot two arrays as a scatterplot. Each array should be 1d. Optionally plot on a provided matplotlib Axes.
    highlight_pts should be an array of the same len as arra & arrb of 0s and 1s, 1s are points to highlight 
    
    """
    if save_name is None: 
        save_name = title
    if nan_to_zero: 
        arra = np.nan_to_num(arra, nan=0)
        arrb = np.nan_to_num(arrb, nan=0)
        arra[~np.isfinite(arra)] = 0
        arrb[~np.isfinite(arrb)] = 0
    else:
        nan_positions = np.isnan(arra) | np.isnan(arrb)
        arra = arra[~nan_positions]
        arrb = arrb[~nan_positions]

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure  # Use the provided axis' figure

    if plot_density and highlight_pts is None: 
        if arra.shape[0] > 3500: 
            print('sub sampling density for speed')
            mask = np.random.permutation(arra.shape[0])[:3500]
            colors = scipy.stats.gaussian_kde(np.vstack([arra[mask], arrb[mask]]))(np.vstack([arra, arrb]))
            scatter = ax.scatter(arra, arrb, c=colors, cmap='viridis', s=s, alpha=alpha)
        else: 
            colors = scipy.stats.gaussian_kde(np.vstack([arra, arrb]))(np.vstack([arra, arrb]))
            scatter = ax.scatter(arra, arrb, c=colors, cmap='viridis', s=s, alpha=alpha)
    elif color_arr is not None and highlight_pts is None:
        color_arr = np.array(color_arr)
        sort_idx = np.argsort(color_arr)
        arra = np.array(arra)[sort_idx]
        arrb = np.array(arrb)[sort_idx]
        color_arr = color_arr[sort_idx]
        scatter = ax.scatter(arra, arrb, c=color_arr, s=s, alpha=alpha, cmap='viridis')
        if discrete_color_labels is not None: 
            handles = []
            for label, color in discrete_color_labels.items():
                handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label, 
                                          markerfacecolor=color, markersize=10))
            ax.legend(handles=handles, fontsize=fontsize)
        if colorbar_label is not None: 
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label, rotation=270, labelpad=15, fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.outline.set_visible(False)
    else: 
        ax.scatter(arra, arrb, c='cornflowerblue', s=s, alpha=alpha)
        if highlight_pts is not None: 
            ax.scatter(arra[highlight_pts], arrb[highlight_pts], c='#4B0082', s=s, alpha=alpha)

    ax.set_xlabel(arra_t, fontsize=fontsize)
    ax.set_ylabel(arrb_t, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    if hline_y is not None:
        ax.axhline(hline_y, color='red', linestyle='--', linewidth=1)
    
    if add_identity_line: 
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, color='red', linestyle='--', linewidth=2)
    
    if axes_lines:
        ax.axvline(0, color='black', linewidth=1)
        ax.axhline(0, color='black', linewidth=1)

    pearson, p_val = scipy.stats.pearsonr(arra, arrb)
    spearman, p_val = scipy.stats.spearmanr(arra, arrb)
    title_str = f"n = {str(arra.shape[0])} \n Pearson R = {pearson:.6f} \n Spearman R = {spearman:.6f}"
    if title != '':
        ax.set_title(f"{title}\n{title_str}" if auto_title else title, fontsize=fontsize)
    else:
        if auto_title:
            ax.set_title(title_str, fontsize=fontsize)

    if names is not None:
        texts = []
        for i, name in enumerate(names):
            if name != '': 
                text = ax.annotate(name, (arra[i], arrb[i]), fontsize=fontsize-2, alpha=0.8)
                texts.append(text)
        print('adjusting text')
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_dir is not None and ax is None: 
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f'{save_dir}{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')   
    elif ax is None: 
        fig.tight_layout()
        fig.show()


def plot_hist(data, xlabel='', ylabel='', title='', save_dir=None, logscale=False, bins=100,
              xlim_min=None, ylim_min=None, xlim_max=None, ylim_max=None, show_x0=True,
              show_mean=True, additional_data=None, data_t=None, additional_data_t=None,
              show_n=True, save_name=None, fontsize=16): 
    """
    Plot one (or two, if additional_data is not None) arrays as histograms. 
    """
    plt.clf()
    if save_name is None: 
        save_name = title
    fig, ax = plt.subplots() 

    # Plot histograms
    if additional_data is not None:
        ax.hist(additional_data, color='darkgreen', bins=bins, label=additional_data_t, alpha=0.5)
        ax.hist(data, color='darkred', bins=bins, label=data_t, alpha=0.5)
        ax.legend(fontsize=fontsize)
    else: 
        ax.hist(data, color='cornflowerblue', bins=bins, alpha=1)

    # Axis labels
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    # Title
    if show_mean:
        ax.set_title(f"{title} \n n={len(data)} \n mean={np.nanmean(data):.2f} \n median={np.median(data):.2f}", fontsize=fontsize)
        ax.axvline(np.nanmean(data), color='darkorange', linestyle='--', linewidth=1)
    else: 
        if show_n: 
            ax.set_title(f"{title} \n n={len(data)}", fontsize=fontsize)  
        else: 
            ax.set_title(f"{title}", fontsize=fontsize)  

    # Optional vertical lines
    if show_x0: 
        ax.axvline(0, color='black', linewidth=1)

    # Log scale
    if logscale:
        ax.set_yscale('log')
    
    # Axis limits
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    if xlim_min is not None or xlim_max is not None:
        new_xlim_min = xlim_min if xlim_min is not None else current_xlim[0]
        new_xlim_max = xlim_max if xlim_max is not None else current_xlim[1]
        ax.set_xlim(new_xlim_min, new_xlim_max)
    
    if ylim_min is not None or ylim_max is not None:
        new_ylim_min = ylim_min if ylim_min is not None else current_ylim[0]
        new_ylim_max = ylim_max if ylim_max is not None else current_ylim[1]
        ax.set_ylim(new_ylim_min, new_ylim_max)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set tick font sizes
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Save or show
    if save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')   
    else: 
        plt.show()