from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd


# From https://github.com/carpenter-singh-lab/2023_vanDijk_CytoSummaryNet/tree/master/Jupyter_scripts
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import average_precision_score
import copy
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
import numpy as np
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# sns.set_style("whitegrid")
sns.set(rc={"lines.linewidth": 2})



################################################

def run_pr_plot(methods,data_reps,fig_dir,savename=None):

    sns.set_context("paper",font_scale = 1.8, rc={"font.size":8,"axes.titlesize":16,"axes.labelsize":16,"axes.facecolor": "white","axes.style": "ticks"})
    sns.set_style("ticks")

    ncols = len(data_reps)
    # plot distribution of replicate and random correlations
    max_y_val = 0
    fig, axes = plt.subplots(nrows=1,ncols=ncols,figsize=(5*ncols,3.8),sharey=True, sharex=True)

    if len(data_reps)>1:
        for i, m in enumerate(data_reps):

            # plot distribution of replicate and random correlations
            # repC = methods[m]['rep_corr']
            # randC_v2 = methods[m]['rand_corr']
            # repCorrDf = methods[m]['corr_df']   
        
            perc90=np.percentile(methods[m]['rand_corr'], 90);

            rgb_values = get_color_from_palette("Set2", i)
            sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, label=f"{methods[m]['name']}",ax=axes[i],color=rgb_values,linewidth=2, fill =True);
            sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, ax=axes[i],color=rgb_values,linewidth=1, fill =False);

            curr_max_y_val = np.max(axes[i].lines[0].get_data())
            max_y_val = max(max_y_val,curr_max_y_val)
            axes[i].set_xlabel('Correlation');
            
            # axes[i].set_title(METHOD_NAME_MAPPING[methods[m]['name']])

            if i == 1:
                axes[i].axvline(x=methods[m]['rand90'],linestyle=':',color = 'firebrick', linewidth=2, label='corr threshold')
                sns.kdeplot(methods[m]['rand_corr'], bw_method=.2, label="Random pairs",ax=axes[i],color='darkgrey')
            else:
                axes[i].axvline(x=methods[m]['rand90'],linestyle=':',color = 'firebrick', linewidth=2)
                sns.kdeplot(methods[m]['rand_corr'], bw_method=.2, ax=axes[i],color='darkgrey')
            axes[i].axvline(x=0,linestyle=':',color='k', linewidth=1);

        y_lim = np.max(max_y_val * 1.33)
        text_y_loc = y_lim*0.65
        for i, m in enumerate(data_reps):
            axes[i].text(0.08+methods[m]['rand90'], text_y_loc,str(int(np.round(methods[m]['percent_replicating'],2)))+ '%>t',fontsize=14) #add text

    else:
        m='l1k'
        ncols =1 
        fig, axes = plt.subplots(nrows=1,ncols=ncols,figsize=(4.5*ncols,4))
                

        # repCorrDf = methods[m]['corr_df']
        sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, label="replicate pairs",ax=axes);
        sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, label=f"{methods[m]['name']}",ax=axes,linewidth=2, fill =True);
        axes.axvline(x=methods[m]['rand90'],linestyle=':',color = 'firebrick', linewidth=2, label='corr threshold')
        sns.kdeplot(methods[m]['rand_corr'], bw_method=.2, label="Random pairs",ax=axes,color='darkgrey')
        axes.axvline(x=0,linestyle=':',color='k', linewidth=1);

        axes.set_xlabel('Correlation');

        curr_max_y_val = np.max(axes.lines[0].get_data())
        max_y_val = max(max_y_val,curr_max_y_val)
        y_lim = np.max(max_y_val * 1.33)
        text_y_loc = y_lim*0.65
        # for i, m in enumerate(data_reps):
        axes.text(0.08+methods[m]['rand90'], text_y_loc,str(int(np.round(methods[m]['percent_replicating'],2)))+ '%>t',fontsize=14) #add text

        
    if savename is None:
        savename = f'{m}_RC'

    plt.ylim(0,y_lim)
    plt.xlim(-1,1.2)
    plt.tight_layout() 
    debug = False
    if not debug:
        plt.savefig(f'{fig_dir}/{savename}.png',dpi=500)
    plt.close()



def compute_venn2_subsets(a, b):

    if not (type(a) == type(b)):
        raise ValueError("Both arguments must be of the same type")
    set_size = len if type(a) != Counter else lambda x: sum(x.values())   # We cannot use len to compute the cardinality of a Counter
    return (set_size(a - b), set_size(b - a), set_size(a & b))



# input is a list of dfs--> [cp,l1k,cp_cca,l1k_cca]
#######
def plotRepCorrs(allData,pertName):
    corrAll=[]
    for d in range(len(allData)):
        df=allData[d][0];
        features=allData[d][1];
        uniqPert=df[pertName].unique().tolist()
        repC=[]
        randC=[]
        for u in uniqPert:
            df1=df[df[pertName]==u].drop_duplicates().reset_index(drop=True)
            df2=df[df[pertName]!=u].drop_duplicates().reset_index(drop=True)
            repCorr=np.sort(np.unique(df1.loc[:,features].T.corr().values))[:-1].tolist()
#             print(repCorr)
            repC=repC+repCorr
            randAllels=df2[pertName].drop_duplicates().sample(df1.shape[0],replace=True).tolist()
            df3=pd.concat([df2[df2[pertName]==i].reset_index(drop=True).iloc[0:1,:] for i in randAllels],ignore_index=True)
            randCorr=df1.corrwith(df3, axis = 1,method='pearson').values.tolist()
            randC=randC+randCorr

        corrAll.append([randC,repC]);
    return corrAll


##############################################################################################################

def get_color_from_palette(name, index):
    # Get the colormap
    cmap = cm.get_cmap(name)

    # Get the RGB values at the specified index
    rgb_values = cmap.colors[index]

    return rgb_values