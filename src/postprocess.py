import numpy as np
import pandas as pd

class QuestPostprocessor:
    def __init__(self, train_labels, denominator=60):
        """
        Initialize with training labels to calculate optimal quantiles.
        
        Args:
            train_labels (pd.DataFrame or np.array): The ground truth labels from the training set.
                                                     Should be the flattened array of all target values
                                                     or the DataFrame of targets.
            denominator (int): The number of intervals to divide the distribution into.
        """
        # Flatten the labels if it's a DataFrame
        if isinstance(train_labels, pd.DataFrame):
            self.y_labels = train_labels.values.flatten()
        else:
            self.y_labels = np.array(train_labels).flatten()
            
        self.unique_labels = np.array(sorted(np.unique(self.y_labels)))
        self.denominator = denominator
        self.exp_labels = self._compute_expected_labels()

    def _compute_expected_labels(self):
        """
        Create optimal percentiles (bins) based on the unique labels.
        """
        q = np.arange(0, 101, 100 / self.denominator)
        exp_labels = np.percentile(self.unique_labels, q)
        return exp_labels

    def optimize_ranks(self, preds):
        """
        Snap predictions to the nearest expected label to optimize Spearman correlation.
        
        Args:
            preds (np.array): The continuous predictions from the model.
            
        Returns:
            np.array: The optimized predictions.
        """
        new_preds = np.zeros(preds.shape)
        
        # Use self.exp_labels as the bins
        bins = self.exp_labels
        
        # Iterate over each column (target)
        for i in range(preds.shape[1]):
            # np.digitize returns indices where bins[i-1] <= x < bins[i]
            # If values are beyond the last bin, it returns len(bins)
            interpolate_bins = np.digitize(preds[:, i], bins=bins, right=False)
            
            # Clip indices to prevent IndexError if prediction >= max(bins)
            # This maps anything larger than the largest bin to the largest bin
            interpolate_bins = np.minimum(interpolate_bins, len(bins) - 1)
            # Also clip 0 just in case, though digitize usually returns >= 0
            interpolate_bins = np.maximum(interpolate_bins, 0)

            # Check for the "single unique value" edge case mentioned in the magic
            if len(np.unique(interpolate_bins)) == 1:
                # Use original preds to avoid zero std dev (Spearman NaN)
                new_preds[:, i] = preds[:, i]
            else:
                new_preds[:, i] = bins[interpolate_bins]

        return new_preds
