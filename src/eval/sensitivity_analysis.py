
from scipy import stats
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import utils.global_variables as gv
sns.set(style="ticks")



def standardize_per_catX(df,column_name,cp_features):
# column_name='Metadata_Plate'
#     cp_features=df.columns[df.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
    df_scaled_perPlate=df.copy()
    df_scaled_perPlate[cp_features]=\
    df[cp_features+[column_name]].groupby(column_name)\
    .transform(lambda x: (x - x.mean()) / x.std()).values
    return df_scaled_perPlate


def analyze_reconstruction_reproducibility(
    df: pd.DataFrame,
    feature_columns: list,
    compound_id_column: str,
    reproducible_cpds: list = None,  # New optional parameter
    n_bins: int = 5,
    threshold_percentile: float = 95,
    show_stats: bool = False,
    show_fig: bool = True,
    output_dir: str = None
):
    """
    Analyze the relationship between reconstruction quality and reproducibility by compounds.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the reconstruction errors and replicate information
    feature_columns : list
        List of column names containing the feature reconstruction errors
    replicate_group_column : str
        Name of the column containing replicate group labels
    compound_id_column : str
        Name of the column containing compound identifiers
    n_bins : int
        Number of bins to divide compounds into based on reconstruction quality
    threshold_percentile : float
        Percentile to use as threshold for "poorly" reconstructed features
    """

    # Calculate threshold for each feature
    thresholds = {col: np.percentile(df[col], threshold_percentile)
                  for col in feature_columns}

    # Calculate poor reconstruction fraction for each compound
    compound_scores = []

    # Get unique compounds
    unique_compounds = df[compound_id_column].unique()

    for compound in unique_compounds:
        compound_data = df[df[compound_id_column] == compound][feature_columns]

        # Calculate fraction of poorly reconstructed features for this compound
        poor_recon_fraction = np.mean([
            (compound_data[col] > thresholds[col]).mean()
            for col in feature_columns
        ])

        compound_scores.append({
            'compound_id': compound,
            'poor_reconstruction_fraction': poor_recon_fraction
        })

    # Create DataFrame with compound scores
    compound_scores_df = pd.DataFrame(compound_scores)

    # Create bins based on poor reconstruction fraction
    try:
        compound_scores_df['bin'] = pd.qcut(
            compound_scores_df['poor_reconstruction_fraction'],
            n_bins,
            labels=False,
            duplicates='drop'
        )
    except ValueError:
        unique_values = len(compound_scores_df['poor_reconstruction_fraction'].unique())
        actual_bins = min(n_bins, unique_values)
        compound_scores_df['bin'] = pd.qcut(
            compound_scores_df['poor_reconstruction_fraction'],
            actual_bins,
            labels=False,
            duplicates='drop'
        )
        print(f"Warning: Reduced number of bins to {actual_bins} due to duplicate values")

    # Calculate reproducibility for each bin
    bin_stats = []

    for bin_idx in compound_scores_df['bin'].unique():
        # Get compounds in this bin
        bin_compounds = compound_scores_df[compound_scores_df['bin'] == bin_idx]['compound_id']

        # Calculate correlations between replicates for compounds in this bin
        bin_correlations = []

        for compound in bin_compounds:
            # Get replicate data for this compound
            compound_data = df[df[compound_id_column] == compound][feature_columns]

            if len(compound_data) > 1:  # Only calculate if we have multiple replicates
                # Calculate correlation matrix for all features
                corr_matrix = np.corrcoef(compound_data)
                # Get upper triangle correlations
                upper_triangle = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]
                if len(upper_triangle) > 0:
                    bin_correlations.append(np.mean(upper_triangle))

        if bin_correlations:
            stats_dict = {
                'bin': bin_idx,
                'n_compounds': len(bin_correlations),
                'mean_correlation': np.mean(bin_correlations),
                'std_correlation': np.std(bin_correlations) if len(bin_correlations) > 1 else np.nan,
                'mean_poor_reconstruction': compound_scores_df[compound_scores_df['bin'] == bin_idx][
                    'poor_reconstruction_fraction'].mean()
            }

            # Add percentage of reproducible compounds if list is provided
            if reproducible_cpds is not None:
                bin_compounds_list = list(bin_compounds)
                n_reproducible = sum(1 for cpd in bin_compounds_list if cpd in reproducible_cpds)
                stats_dict['pct_reproducible'] = (n_reproducible / len(bin_compounds_list)) * 100

            bin_stats.append(stats_dict)

    bin_results = pd.DataFrame(bin_stats)

    # Sort bin_results by mean_poor_reconstruction
    bin_results = bin_results.sort_values('mean_poor_reconstruction')

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Distribution of poor reconstruction fractions
    sns.histplot(compound_scores_df['poor_reconstruction_fraction'], ax=ax1)
    ax1.set_xlabel('Fraction of poorly reconstructed features per compound')
    ax1.set_ylabel('Count of compounds')
    ax1.set_title('Distribution of compound reconstruction quality')

    # Reproducibility vs reconstruction quality
    ax2.scatter(bin_results['mean_poor_reconstruction'],
                bin_results['mean_correlation'])
    ax2.plot(bin_results['mean_poor_reconstruction'],
             bin_results['mean_correlation'], '--')

    # Add error bars
    ax2.errorbar(bin_results['mean_poor_reconstruction'],
                 bin_results['mean_correlation'],
                 yerr=bin_results['std_correlation'],
                 fmt='none',
                 capsize=5)

    ax2.set_xlabel('Mean fraction of poorly reconstructed features')
    ax2.set_ylabel('Reproducibility score (correlation)')
    ax2.set_title('Reproducibility vs Reconstruction Quality')

    # Calculate correlation and p-value
    if len(bin_results) > 1:
        correlation, p_value = stats.pearsonr(
            bin_results['mean_poor_reconstruction'],
            bin_results['mean_correlation']
        )
    else:
        correlation, p_value = np.nan, np.nan

    plt.tight_layout()
    if show_fig:
        plt.show()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'reproducibility_vs_reconstruction_quality.png'),
                    dpi=300, bbox_inches='tight')

    plt.close()

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Distribution of poor reconstruction fractions
    sns.histplot(compound_scores_df['poor_reconstruction_fraction'], ax=ax1)
    ax1.set_xlabel('Fraction of poorly reconstructed features per compound')
    ax1.set_ylabel('Count of compounds')
    ax1.set_title('Distribution of compound reconstruction quality')

    # Bar plot of reproducibility vs reconstruction quality
    bars = ax2.bar(range(len(bin_results)),
                   bin_results['mean_correlation'],
                   yerr=bin_results['std_correlation'],
                   capsize=5)

    if show_stats:
        # Add number of compounds on top of each bar
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'n={int(bin_results.iloc[idx]["n_compounds"])}',
                     ha='center', va='bottom')

    # Add mean poor reconstruction fraction as x-tick labels
    ax2.set_xticks(range(len(bin_results)))
    ax2.set_xticklabels([f'{x:.3f}' for x in bin_results['mean_poor_reconstruction']],
                        rotation=45)

    ax2.set_xlabel('Mean fraction of poorly reconstructed features')
    ax2.set_ylabel('Reproducibility score (correlation)')
    ax2.set_title('Reproducibility vs Reconstruction Quality')

    # Add grid for better readability
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Calculate correlation and p-value
    if len(bin_results) > 1:
        correlation, p_value = stats.pearsonr(
            bin_results['mean_poor_reconstruction'],
            bin_results['mean_correlation']
        )
        # Add correlation and p-value to the plot
        ax2.text(0.05, 0.95,
                 f'r = {correlation:.3f}\np = {p_value:.3e}',
                 transform=ax2.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        correlation, p_value = np.nan, np.nan

    plt.tight_layout()
    if show_fig:
        plt.show()

    plt.close()


    return {
        'compound_scores': compound_scores_df,
        'bin_results': bin_results,
        'correlation': correlation,
        'p_value': p_value,
        'figure': fig
    }


def analyze_multiple_threshold_sensitivity(
        df: pd.DataFrame,
        feature_columns: list,
        compound_id_column: str,
        thresholds_percentiles: list = [50, 75, 90, 95, 97.5, 99],
        n_bins: int = 10,
        reproducible_cpds: list = None,
        show_fig: bool = True,
        output_dir: str = None

):
    """
    Analyze how the reproducibility-reconstruction relationship changes across different thresholds.

    Parameters:
    -----------
    [Previous parameters remain the same]
    thresholds_percentiles : list
        List of percentile thresholds to test
    """

    # Store results for each threshold
    threshold_results = []

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds_percentiles)))

    # For each threshold
    for thresh_idx, thresh in enumerate(thresholds_percentiles):
        # Run the analysis with current threshold
        results = analyze_reconstruction_reproducibility(
            df=df,
            feature_columns=feature_columns,
            compound_id_column=compound_id_column,
            reproducible_cpds=reproducible_cpds,
            n_bins=n_bins,
            threshold_percentile=thresh,
            show_fig=False
        )


        bin_results = results['bin_results']

        # Store correlation results
        threshold_results.append({
            'threshold': thresh,
            'correlation': results['correlation'],
            'p_value': results['p_value']
        })

        # Plot correlation line for this threshold
        ax1.plot(bin_results['mean_poor_reconstruction'],
                 bin_results['mean_correlation'],
                 '-o', color=colors[thresh_idx],
                 label=f'{thresh}th percentile',
                 alpha=0.7)

    ax1.set_xlabel('Mean fraction of poorly reconstructed features')
    ax1.set_ylabel('Reproducibility score (correlation)')
    ax1.set_title('Reproducibility vs Reconstruction Quality\nAcross Different Thresholds')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title='Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.legend(title='Threshold', loc='upper right')

    # Create summary plot of correlations
    thresh_df = pd.DataFrame(threshold_results)
    ax2.plot(thresh_df['threshold'], thresh_df['correlation'], '-o')

    # Add significance markers
    significant = thresh_df['p_value'] < 0.05
    if any(significant):
        ax2.plot(thresh_df.loc[significant, 'threshold'],
                 thresh_df.loc[significant, 'correlation'],
                 'o', color='red', markersize=10, fillstyle='none',
                 label='p < 0.05')
        ax2.legend()

    ax2.set_xlabel('Threshold percentile')
    ax2.set_ylabel('Correlation coefficient (% poorly reconstructed & % Reproducible')
    ax2.set_title('Strength of Reproducibility-Reconstruction\nRelationship vs Threshold')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if show_fig:
        plt.show()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'reproducibility_vs_reconstruction_quality_for_thresholds.png'),
                    dpi=300, bbox_inches='tight')


    return {
        'threshold_results': pd.DataFrame(threshold_results),
        'figure': fig
    }



# fill in the path to the directory where the data is stored

BASE_DIR = '/sise/assafzar-group/assafzar/genesAndMorph'
BASE_DIR = '/Users/alonshp/rosetta-data/'
DATA_DIR = f'{BASE_DIR}/preprocessed_data'
PROCESSED_DATA_DIR = f'{BASE_DIR}/anomaly_output'
RESULTS_DIR = f'{BASE_DIR}/results'

with_reproduced = True

exp_dir = 'test_code_refactor'
exp_dir = '2003_t'
dataset = 'CDRP-bio'
# dataset = 'TAORF'

datasets = ["CDRP-bio","TAORF", "LINCS", "LUAD"]

for dataset in datasets:

    if 'nvs' in exp_dir:
        profile_type = 'normalized_variable_selected'
    else:
        profile_type = 'augmented'
    # profile_type = 'normalized_variable_selected'


    base_dir = f'{PROCESSED_DATA_DIR}/{dataset}/CellPainting/{exp_dir}/'
    res_dir = f'{RESULTS_DIR}/{dataset}/CellPainting/{exp_dir}/'
    fig_dir = f'{res_dir}/figs/feature_importance/'
    os.makedirs(fig_dir, exist_ok=True)
    # base_dir = './'
    anomaly_path = f'{base_dir}replicate_level_cp_{profile_type}_ae_diff.csv'
    preds_path = f'{base_dir}replicate_level_cp_{profile_type}_preds.csv'
    cellprofiler_path = f'{base_dir}replicate_level_cp_{profile_type}_baseline.csv'
    pow_path = f'{base_dir}replicate_level_cp_{profile_type}_ae_diff_power.csv'


    reps = {
        # 'CellProfiler':  pd.read_csv(cellprofiler_path, compression='gzip'),
        # 'Predictions': pd.read_csv(preds_path, compression='gzip'),
        'Anomaly': pd.read_csv(anomaly_path, compression='gzip'),
        # 'MSE': pd.read_csv(pow_path, compression='gzip')
    }
    hue_order = reps.keys()
    features = reps['Anomaly'].columns[reps['Anomaly'].columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()



    if gv.DS_INFO_DICT[dataset]['has_moa']:
        moa_col = gv.DS_INFO_DICT[dataset]['CellPainting']['moa_col']
    cpd_col = gv.DS_INFO_DICT[dataset]['CellPainting']['cpd_col']
    plate_col = gv.DS_INFO_DICT[dataset]['CellPainting']['plate_col']
    # load the reproducibility data
    # reproducibility_df = pd.read_csv(f'{res_dir}reproducibility.csv')
    if 'nvs' in exp_dir:
        corr_path = f'{res_dir}reproducible_cpds_nvs_ba.csv'
    else:
        corr_path = f'{res_dir}reproducible_cpds_a_ba.csv'
    corr_path = f'{res_dir}RepCorrDF.xlsx'


    if with_reproduced:
        repCorrDF = pd.read_excel(corr_path, sheet_name=None)
        reproducible = {}
        for i in repCorrDF.keys():
            perc90 = repCorrDF[i]['Rand90Perc'][0]
            reproducible[i] = list(repCorrDF[i][repCorrDF[i]['RepCor'] > perc90]['index'])

        reproducible_cpds = set()
        for i in reproducible.keys():
            reproducible_cpds = reproducible_cpds.union(set(reproducible[i]))

    reps_normalized = {}
    for r in reps:
        reps_normalized[r] = standardize_per_catX(reps[r],plate_col,features)

    for r in reps:
        # data_moa = reps_normalized[r][reps_normalized[r][moa_col]==m]
        # data_moa = reps[r][reps[r][moa_col].isin([m])]
        data_for_analysis = reps[r]
        data_only_treat = data_for_analysis[data_for_analysis['Metadata_set'] == 'test_treat']
        # print(f'{r} {m} {data_moa.shape[0]}')

        # Run analysis
        results = analyze_reconstruction_reproducibility(
            data_only_treat,
            features,
            n_bins=10,
            threshold_percentile=95,
            compound_id_column ='Metadata_broad_sample',
            reproducible_cpds = reproducible_cpds,
            show_stats = False,
            show_fig = True,
            output_dir=res_dir
        )

        # Run analysis
        results = analyze_multiple_threshold_sensitivity(
            data_only_treat,
            features,
            n_bins=10,
            compound_id_column='Metadata_broad_sample',
            reproducible_cpds=reproducible_cpds,
            output_dir=res_dir
            # output_dir=res_dir
        )

