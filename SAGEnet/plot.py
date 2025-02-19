import pandas as pd
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import os
import seaborn as sb
from sklearn.metrics import r2_score
import tangermeme.plot
from adjustText import adjust_text

def plot_attribs(arr,title,annotations=None,center_line=False,save_dir=None,save_name=''): 
    """
    Plot model attributions, optionally labeled with annotations. Arr is attribution array of shape (4, seq_lenth). 
    """
    plt.figure(figsize=(15, 1))
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
            plt.text(x=(curr_annotation['start']+curr_annotation['end'])//2, y=-1.25*abs_max, s=curr_annotation['match_rank_0'].split('.')[0], color='purple', ha='center', fontsize=10)
    
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



def line_plot_compare(arra,arrb,xlabels,x=None,arra_t='',arrb_t='',xlabel='',ylabel='',title='',shade_std=True,fig_width=6,fig_height=6,legend_x=.7,legend_y=.95,save_dir=None,save_name=None): 
    """
    Plot Line plot comparing two arrays. Each array can be 1d or 2d, if 2d, the first dimension determines the number of x-ticks, the second dimension will be plotted as median and standard deviation. 
    """
    plt.clf()
    if save_name is None: 
        save_name=title
    
    plt.figure(figsize=(fig_width, fig_height))
    if x is None:
        x=xlabels
    if len(arra.shape)==1: #1d
        shade_std=False
        arra_mean=arra
        arrb_mean=arrb
    else: 
        arra_mean = np.nanmedian(arra, axis=1) 
        arrb_mean = np.nanmedian(arrb, axis=1)  
    if shade_std: 
        arra_std = np.nanstd(arra, axis=1) 
        arrb_std = np.nanstd(arrb, axis=1) 
        plt.fill_between(x, arra_mean - arra_std, arra_mean + arra_std, color="#1f77b4", alpha=0.2)
        plt.fill_between(x, arrb_mean - arrb_std, arrb_mean + arrb_std, color="#ff7f0e", alpha=0.2)
    
    plt.plot(x, arra_mean, label=arra_t, color="#1f77b4")
    plt.plot(x, arrb_mean, label=arrb_t, color="#ff7f0e")
    plt.scatter(x, arra_mean, color="#1f77b4", zorder=3)  
    plt.scatter(x, arrb_mean, color="#ff7f0e", zorder=3)  
    
    plt.xticks(ticks=x, labels=xlabels) 
    plt.axhline(y=0, color='black', linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=10,bbox_to_anchor=(legend_x, legend_y),frameon=False)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(title)
    plt.tight_layout()
    
    if save_dir: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf',format='pdf', dpi=300,bbox_inches='tight')
    else: 
        plt.show()
    
def sb_boxplot(res_df, xlabel='', ylabel='',save_dir=None,save_name=None,title='',fig_width=11,fig_height=6,scatter=True,dot_size=5,dot_alpha=.1,boxplot=True,custom_palette=None,save_fig=None):
    """
    Plot seaborn boxplots of one or more arrays. res_df should be a pandas dataframe, with each column being an array to plot. 
    """
    plt.clf()
    if custom_palette is None: 
        custom_palette='colorblind'

    if save_name is None: 
        save_name=title
    res_df = res_df.fillna(0)
    plt.figure(figsize=(fig_width, fig_height))
    if boxplot: 
        sb.boxplot(data=res_df.dropna(how='any'), whis=np.inf, width=0.8,palette=custom_palette)
    if scatter:
        if len(res_df)==1: 
            sb.stripplot(data=res_df.dropna(how='any'), color='black', size=dot_size, jitter=True, alpha=dot_alpha,palette='colorblind')
        else: 
            sb.stripplot(data=res_df.dropna(how='any'), color='black', size=dot_size, jitter=True, alpha=dot_alpha)
    x_min, x_max = plt.xlim()
    buffer = 0.1
    plt.xlim(x_min - buffer, x_max + buffer)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)    
    plt.tight_layout()
    sb.despine()
    if save_dir: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf',format='pdf', dpi=300,bbox_inches='tight')
    else: 
        plt.show()

        
def boxplot_compare(arra, arrb, arra_t='',arrb_t='',save_dir=None,title='',stat_test='ks',ylabel='',fig_width=7,fig_height=6,include_stat_test=True, save_name=None):
    """
    Plot two arrays as boxplots. Each array should be 1d. 
    """
    plt.clf()
    if save_name is None: 
        save_name=title
    fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 
    arra_t = f"{arra_t} \n n={len(arra)} \n mean={np.mean(arra):.2f} \n median={np.median(arra):.2f}"
    arrb_t = f"{arrb_t} \n n={len(arrb)} \n mean={np.mean(arrb):.2f} \n median={np.median(arrb):.2f}"
    plt.boxplot([arra, arrb], showfliers=False)
    plt.xticks([1, 2], [arra_t, arrb_t])
    if include_stat_test: 
        if stat_test == 'wilcoxon':
            stat, p_value = scipy.stats.wilcoxon(arra, arrb)
            stat_test_label = 'Wilcoxon'
        elif stat_test == 'ks':
            stat, p_value = scipy.stats.ks_2samp(arra, arrb)
            stat_test_label = 'K-S'
    if title!='':
        plt.title(f"{title} \n {stat_test_label} p={p_value}")
    else: 
        plt.title(f"{stat_test_label} p={p_value:.4e}")
    plt.tight_layout()
    plt.ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf',format='pdf', dpi=300,bbox_inches='tight')   
    else: 
        plt.show()


def scatter_compare(arra, arrb,arra_t='', arrb_t='', save_dir=None,title='', add_identity_line=False,nan_to_zero=True,save_name=None,plot_density=True,axes_lines=0,names=None,show_r2=False,color_arr=None,color_labels=None,corr_label=True,hline_y=None): 
    """
    Plot two arrays as a scatterplot. Each array should be 1d. 
    """
    plt.clf()
    if save_name is None: 
        save_name=title
    if nan_to_zero: 
        arra = np.nan_to_num(arra, nan=0)
        arrb = np.nan_to_num(arrb, nan=0)
    else:
        nan_positions = np.isnan(arra) | np.isnan(arrb)
        arra = arra[~nan_positions]
        arrb = arrb[~nan_positions]
    fig, ax = plt.subplots(figsize=(6, 6))
    if plot_density: 
        if arra.shape[0] > 3500: 
            print('sub sampling density for speed')
            mask = np.random.permutation(arra.shape[0])[:3500]
            colors = scipy.stats.gaussian_kde(np.vstack([arra[mask],arrb[mask]]))(np.vstack([arra,arrb]))
            plt.scatter(arra ,arrb, c = colors, cmap='viridis', s=50,alpha=.2)
        else: 
            colors = scipy.stats.gaussian_kde(np.vstack([arra,arrb]))(np.vstack([arra,arrb]))
            plt.scatter(arra ,arrb, c = colors, cmap='viridis', s=15,alpha=.5)
    elif color_arr is not None:
        plt.scatter(arra ,arrb, c = color_arr, s=50,alpha=.6)
        handles = []
        for label, color in color_labels.items():
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label, 
                                    markerfacecolor=color, markersize=10))
        plt.legend(handles=handles)
    else: 
        plt.scatter(arra ,arrb, c = 'cornflowerblue', s=50,alpha=.6)
        
    plt.xlabel(arra_t)
    plt.ylabel(arrb_t)

    if hline_y is not None:
        plt.axhline(hline_y, color='red', linestyle='--', linewidth=1)
    
    if add_identity_line: 
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
        ax.plot(lims, lims,color='red', linestyle='--', linewidth=2)
    
    if axes_lines:
        plt.axvline(0, color='black', linewidth=1)
        plt.axhline(0, color='black', linewidth=1)
    pearson, p_val = scipy.stats.pearsonr(arra, arrb)
    spearman, p_val = scipy.stats.spearmanr(arra, arrb)
    if corr_label: 
        title_str = f"n = {str(arra.shape[0])} \n Pearson R = {pearson:.6f} \n Spearman R = {spearman:.6f}"
    else: 
        title_str = f"n = {str(arra.shape[0])}"
    if show_r2: 
        r2_sklearn = r2_score(arra, arrb)
        title_str=title_str+ f'\n R2 = {r2_sklearn:.2f}'
    if title != '':
        plt.title(f"{title} \n {title_str}")
    else:
        plt.title(title_str)
    if names is not None:
        texts = []
        for i, name in enumerate(names):
            if name!='': 
                text = ax.annotate(name, (arra[i], arrb[i]), fontsize=10, alpha=0.8)
                texts.append(text)
        print('adjusting text')
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf',format='pdf', dpi=300,bbox_inches='tight')   
    else: 
        plt.show()


def plot_hist(data,xlabel='', ylabel='',title='',save_dir=None,logscale=False,bins=100,xlim_min=None,ylim_min=None,xlim_max=None, ylim_max=None,show_x0=True,show_mean=True,additional_data=None,data_t=None, additional_data_t=None,show_n=True,save_name=None): 
    """
    Plot one (or two, if additional_data is not None) arrays as histograms. 
    """
    plt.clf()
    if save_name is None: 
        save_name=title
    fig, ax = plt.subplots() 
    if additional_data is not None:
        plt.hist(additional_data, color='darkgreen', bins=bins,label=additional_data_t,alpha=.5)
        plt.hist(data,color='darkred',bins=bins,label=data_t,alpha=.5)
        plt.legend()
    else: 
        plt.hist(data,color='cornflowerblue',bins=bins,alpha=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)  
    if show_mean:
        plt.title(f"{title} \n n={str(len(data))} \n mean={np.nanmean(data):.2f} \n median={np.median(data):.2f}",fontsize=10)
        plt.axvline(np.nanmean(data), color='darkorange', linestyle='--', linewidth=1)
    else: 
        if show_n: 
            plt.title(f"{title} \n n={str(len(data))}",fontsize=10)  
        else: 
            plt.title(f"{title}",fontsize=10)  
    if show_x0: 
        plt.axvline(0, color='black', linewidth=1)
    if logscale:
        plt.yscale('log')
    
    current_xlim = plt.gca().get_xlim()
    current_ylim = plt.gca().get_ylim()

    # set xlim and ylim, modifying only the provided boundaries
    if xlim_min is not None or xlim_max is not None:
        new_xlim_min = xlim_min if xlim_min is not None else current_xlim[0]
        new_xlim_max = xlim_max if xlim_max is not None else current_xlim[1]
        plt.xlim(new_xlim_min, new_xlim_max)
    
    if ylim_min is not None or ylim_max is not None:
        new_ylim_min = ylim_min if ylim_min is not None else current_ylim[0]
        new_ylim_max = ylim_max if ylim_max is not None else current_ylim[1]
        plt.ylim(new_ylim_min, new_ylim_max)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save_dir is not None: 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{save_name}.pdf',format='pdf', dpi=300,bbox_inches='tight')   
    else: 
        plt.show()
