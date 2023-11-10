import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def enter_bucket_wrapper(th, method_a = 'map',method_b = 'raw1to1'):
    def enter_bucket(sr):
        res = 0
        if sr[f'{method_a}'] > th:
            res += 1
        if sr[f'{method_b}'] > th:
            res += 10
        # else:
            # if sr[f'{chan}SS_{t}_{method_a}'] > th:
                # res += 1
            # if sr[f'{chan}SS_{t}_{method_b}'] > th:
                # res += 10


        if res == 1:
            return method_a
        if res == 10:
            return method_b
        if res == 11:
            return 'both'

        return 'none'
    
    return enter_bucket

def enter_bucket_wrapper_ss(t, th, chan, method_a = 'map',method_b = 'raw1to1'):
    def enter_bucket(sr):
        res = 0
        if len(chan) >0:
            if sr[f'{chan}_SS_{t}_{method_a}'] > th:
                res += 1
            if sr[f'{chan}_SS_{t}_{method_b}'] > th:
                res += 10
        else:
            if sr[f'{chan}SS_{t}_{method_a}'] > th:
                res += 1
            if sr[f'{chan}SS_{t}_{method_b}'] > th:
                res += 10

        

        if res == 1:
            return method_a
        if res == 10:
            return method_b
        if res == 11:
            return 'both'

        return 'none'
    
    return enter_bucket

##############################################################################################################

def plot_complementary(title, cs, method_a = 'map',method_b = 'raw1to1',save_dir=None):
    nrow = 1
    # ts = [2,4,6]#[2,6,10,14]
    ts = list(cs.keys())
    ncol = len(ts)
#     ylims = [10000, 1200, 700, 550]
    d={'xlabel':"", 'ylabel':"Amount of Hits"}

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*8,nrow*5), subplot_kw=d, facecolor='white')
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    fig.suptitle(f'Distributions of hits {title}', fontsize = 32, y = 1)
    

    def get_bar_heights(df, category):
        if category in df.index:
            return df.loc[category]
        
        return np.zeros(df.shape[1])


    for i, t in enumerate(ts):
        # i=t-2
#         counts = pd.concat({f'{th:.1f}': res.apply(enter_bucket_wrapper(t, th * len(cols[chan]), chan), axis=1).value_counts() for th in np.arange(0, 1.01, 0.1)}, axis=1)
        counts = cs[t]

        labels = counts.columns
        width = 0.55      # the width of the bars: can also be len(x) sequence
        
        c_a = get_bar_heights(counts, method_a)
        c_both = get_bar_heights(counts, 'both')
        c_b = get_bar_heights(counts, method_b)
        if ncol>1:
            ax = axs[i]
        else:
            ax = axs
        r1=ax.bar(labels, c_a, width, label=method_a)
        r2=ax.bar(labels, c_both, width, bottom=c_a, label='both')
        r3=ax.bar(labels, c_b, width, bottom=c_a+c_both, label=method_b)
        # ax[i//ncol,i%ncol].bar_label(r1, padding=3)
        ax.bar_label(r1,label_type='center')
        ax.bar_label(r2,label_type='center')
        ax.bar_label(r3,label_type='center',padding=3)

        # ax[i//ncol,i%ncol].set_yscale('symlog', base=10)
        # ax[i].set_xlim([0.5,6.5])
        # ax[i].set_ylim([0, cs[t].loc[[i for i in cs[t].index if i != 'none']].sum().max()*0.65])
        ax.set_ylabel('Amount of Hits')
        ax.set_title(f'Hits Distribution - Norm-SS({t})', fontsize = 24)
        ax.legend()

    if save_dir is not None:
        
        _=fig.savefig(f'{save_dir}/{title.replace(".","")}.png', format='png')
    # _ = fig.savefig(f'/sise/assafzar-group/g-and-n/tabular_models_results/plots/complementary/{title.replace(".","")}.eps', format='eps')
    plt.tight_layout()

    plt.show()
    # plt.close()
    # counts.head()


# https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def plot_latent_effect(scores, configs):

  for metric in ['mse','r2','pcc']:
    # plots = defaultdict(dict)
    # for dl in configs.eval.scores:
        # dl_metric = np.array(configs.eval.scores[dl][metric])
  # latent_dims = sorted(k for k in model_dict)
  # val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]
        # fig = plt.figure(figsize=(6, 4))
        # plt.plot(
        #   latent_dims, dl_metric, "--", color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y",
        #   markersize=16, label=dl
        # )
        fig = sns.relplot(data=scores,x='latent_dim',y=metric,hue='set', kind="line")
        # plt.plot(
          # latent_dims, dl_metric, label=dl
        # )

        plt.xscale("log")
        plt.xticks(scores['latent_dim'].unique(), labels = scores['latent_dim'].unique())
        plt.title(f"{metric} over latent dim size", fontsize=14)
        plt.xlabel("Latent dimensionality")
        plt.ylabel(metric)
        plt.legend()
        plt.minorticks_off()
        if metric != 'mse':
          plt.ylim(0, 1)
        plt.show()
        fig.savefig(os.path.join(configs.general.res_dir,f'{metric}_by_ldim.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_p_vs_median(df, path, file_name,x_col="fraction_score",y_col ="log10_p_val",plot = 'scatter' ):
    
    """plot p_values vs median correlation scores for each compound for all doses (1-6)"""
    
    if not os.path.exists(path):
        os.mkdir(path)
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12,8)) 
    plt.xlabel(f"{x_col} scores of pairwise correlation btw cpds")
    plt.ylabel("Non-parametric -log10 P-values")
    plt.title(f"P-values vs {x_col} scores for compound replicates")
    # fig = sns.scatterplot(data=df, x="median_scores", y="p_values", hue="dose", 
    #                       style="dose", palette = "viridis")
    if plot == 'scatter':
        fig = sns.scatterplot(data=df, x=x_col, y=y_col, hue="method", 
                        style="method", palette = "viridis",alpha=0.5)
    else:
        fig = sns.kdeplot(data=df, x=x_col, y=y_col, hue="method",fill=True, alpha=0.3)
                        # style="method", palette = "viridis",alpha=0.5)
        
        
    log_val = -np.round(np.log10(0.05),5)
    fig.axhline(log_val, ls='--', c='black')

    plt.show()

# def plot_p_vs_median(df, path, file_name,x_col="fraction_score"):
    
#     """plot p_values vs median correlation scores for each compound for all doses (1-6)"""
    
#     if not os.path.exists(path):
#         os.mkdir(path)
#     plt.rcParams.update({'font.size': 14})
#     plt.figure(figsize=(12,8)) 
#     plt.xlabel("Median scores of pairwise correlation btw cpds")
#     plt.ylabel("Non-parametric P-values")
#     plt.title("P-values vs median scores for compound replicates")
#     # fig = sns.scatterplot(data=df, x="median_scores", y="p_values", hue="dose", 
#     #                       style="dose", palette = "viridis")
#     fig = sns.scatterplot(data=df, x=x_col, y="p_val", hue="method", 
#                       style="method", palette = "viridis",alpha=0.5)

#     fig.axhline(0.05, ls='--', c='black')
#     fig.legend(loc = 'upper right')
#     fig.text(-0.18,0.07, "Significance level (0.05)")
#     plt.savefig(os.path.join(path, file_name))
#     plt.show()

def rename_cols(df):
    'Rename columns from dose number to actual doses'
    
    df.sortname(columns= {'dose_1' : '0.04 uM', 'dose_2':'0.12 uM', 'dose_3':'0.37 uM',
                        'dose_4': '1.11 uM', 'dose_5':'3.33 uM', 'dose_6':'10 uM'}, inplace = True)
    return df

def melt_df(df, col_name):
    """
    This function returns a reformatted dataframe with 
    3 columns: cpd, dose number and dose_values(median score or p-value)
    """
    # df = df.melt(id_vars=['cpd', 'no_of_replicates'], var_name="dose", value_name=col_name)
    df = df.melt(id_vars=['cpd'], var_name="p_val", value_name=col_name)
    return df

def merge_p_median_vals(df_cpd_vals, df_null):
    """
    This function merge p_values and median scores 
    dataframes for each compound for all doses(1-6) 
    """
    df_p_vals = melt_df(df_null, 'p_values')
    df_cpd_vals = melt_df(df_cpd_vals, 'median_scores')
    df_cpd_vals['p_values'] = df_p_vals['p_values']
    return df_cpd_vals