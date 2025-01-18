import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def analyze_reconstruction_reproducibility(
        error_matrix,
        n_bins=5,
        threshold_percentile=95,
        replicate_groups=None
):
    """
    Analyze the relationship between reconstruction quality and reproducibility.

    Parameters:
    -----------
    error_matrix : numpy.ndarray or pandas.DataFrame
        Matrix of reconstruction errors (samples x features)
    n_bins : int
        Number of bins to divide samples into based on reconstruction quality
    threshold_percentile : float
        Percentile to use as threshold for "poorly" reconstructed features
    replicate_groups : list or numpy.ndarray
        Labels indicating which samples are replicates

    Returns:
    --------
    dict containing results and plots
    """

    # Convert to numpy array if DataFrame
    if isinstance(error_matrix, pd.DataFrame):
        error_matrix = error_matrix.values

    # Calculate threshold for "poorly" reconstructed features
    threshold = np.percentile(error_matrix, threshold_percentile)

    # Calculate fraction of poorly reconstructed features for each sample
    poor_reconstruction_fraction = np.mean(error_matrix > threshold, axis=1)

    # Create bins based on poor reconstruction fraction
    bins = pd.qcut(poor_reconstruction_fraction, n_bins, labels=False)

    # Calculate reproducibility for each bin
    reproducibility_scores = []
    bin_centers = []

    for bin_idx in range(n_bins):
        bin_mask = (bins == bin_idx)
        bin_samples = error_matrix[bin_mask]

        # If replicate groups are provided, calculate correlation between replicates
        if replicate_groups is not None:
            bin_replicates = np.array(replicate_groups)[bin_mask]
            unique_groups = np.unique(bin_replicates)
            correlations = []

            for group in unique_groups:
                group_samples = bin_samples[bin_replicates == group]
                if len(group_samples) > 1:
                    # Calculate average correlation between all pairs of replicates
                    corr_matrix = np.corrcoef(group_samples)
                    correlations.extend(corr_matrix[np.triu_indices(len(corr_matrix), k=1)])

            reproducibility_scores.append(np.mean(correlations))
        else:
            # If no replicate groups, use average correlation between all samples in bin
            corr_matrix = np.corrcoef(bin_samples)
            reproducibility_scores.append(
                np.mean(corr_matrix[np.triu_indices(len(corr_matrix), k=1)])
            )

        bin_centers.append(np.mean(poor_reconstruction_fraction[bin_mask]))

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Distribution of poor reconstruction fractions
    sns.histplot(poor_reconstruction_fraction, ax=ax1)
    ax1.set_xlabel('Fraction of poorly reconstructed features')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of reconstruction quality')

    # Reproducibility vs reconstruction quality
    ax2.scatter(bin_centers, reproducibility_scores)
    ax2.plot(bin_centers, reproducibility_scores, '--')
    ax2.set_xlabel('Mean fraction of poorly reconstructed features')
    ax2.set_ylabel('Reproducibility score (correlation)')
    ax2.set_title('Reproducibility vs Reconstruction Quality')

    # Calculate correlation and p-value
    correlation, p_value = stats.pearsonr(bin_centers, reproducibility_scores)

    plt.tight_layout()

    return {
        'poor_reconstruction_fraction': poor_reconstruction_fraction,
        'bins': bins,
        'reproducibility_scores': reproducibility_scores,
        'bin_centers': bin_centers,
        'correlation': correlation,
        'p_value': p_value,
        'figure': fig
    }


# Example usage:
"""
# Generate some example data
n_samples = 100
n_features = 50
n_replicates = 4

# Create synthetic error matrix
np.random.seed(42)
error_matrix = np.random.exponential(scale=1.0, size=(n_samples, n_features))

# Create replicate groups
replicate_groups = np.repeat(range(n_samples // n_replicates), n_replicates)

# Run analysis
results = analyze_reconstruction_reproducibility(
    error_matrix,
    n_bins=5,
    threshold_percentile=95,
    replicate_groups=replicate_groups
)

print(f"Correlation: {results['correlation']:.3f}")
print(f"P-value: {results['p_value']:.3f}")
plt.show()
"""