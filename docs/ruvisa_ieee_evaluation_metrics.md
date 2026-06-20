# Ruvisa — IEEE Conference Paper: Evaluation Methodology and Metrics (Draft)

> **Note on citations.** The bracketed source tags below (e.g., [MultilabelStratified]) are placeholders; assign them final reference numbers once the bibliography is unified with the rest of the paper.

## Evaluation Methodology and Metrics

The ingredient-based concern classifier is evaluated as a multi-label problem over the seven-concern taxonomy. We use 4-fold cross-validation with *MultilabelStratifiedKFold*, which preserves the joint label distribution across folds [MultilabelStratified], training each fold for three epochs with micro-F1 early stopping. The seven-dimensional logits are mapped to independent probabilities via the *sigmoid* (not softmax, since labels are non-exclusive) and thresholded per label. Because class prevalence is uneven (38.9% for acne scars/texture to 85.6% for redness), a single 0.5 cut-off is suboptimal; we instead tune each label's threshold by sweeping 19 values in [0.05, 0.95] to maximize its validation F1, which improves minority-class performance.

We report four complementary metrics over the resulting N×7 prediction matrix. *Label accuracy* is the fraction of individual (product, concern) cells predicted correctly; *subset (exact-match) accuracy* is the stricter fraction of products whose full seven-label vector matches the ground truth; *micro* precision/recall/F1 pool predictions across all N×7 cells, reflecting frequency-weighted aggregate quality; and *macro-F1* averages the seven per-label F1 scores, weighting each concern equally regardless of prevalence. Together they capture aggregate quality, rare-concern performance, and the strict all-seven case. All metrics are reported as mean ± standard deviation across the four folds.
