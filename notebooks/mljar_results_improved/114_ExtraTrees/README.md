# Summary of 114_ExtraTrees

[<< Go back](../README.md)


## Extra Trees Classifier (Extra Trees)
- **n_jobs**: -1
- **criterion**: entropy
- **max_features**: 1.0
- **min_samples_split**: 40
- **max_depth**: 7
- **eval_metric_name**: auc
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

8.1 seconds

## Metric details
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0134396 | nan         |
| auc       | 0.971965  | nan         |
| f1        | 0.217822  |   0.0702154 |
| accuracy  | 0.988401  |   0.0702154 |
| precision | 0.141026  |   0.0702154 |
| recall    | 1         |   0         |
| mcc       | 0.291571  |   0.033504  |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0134396 | nan         |
| auc       | 0.971965  | nan         |
| f1        | 0.217822  |   0.0702154 |
| accuracy  | 0.988401  |   0.0702154 |
| precision | 0.141026  |   0.0702154 |
| recall    | 0.478261  |   0.0702154 |
| mcc       | 0.255383  |   0.0702154 |


## Confusion matrix (at threshold=0.070215)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6721 |               67 |
| Labeled as 1 |               12 |               11 |

## Learning curves
![Learning curves](learning_curves.png)

## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)


## ROC Curve

![ROC Curve](roc_curve.png)


## Kolmogorov-Smirnov Statistic

![Kolmogorov-Smirnov Statistic](ks_statistic.png)


## Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)


## Calibration Curve

![Calibration Curve](calibration_curve_curve.png)


## Cumulative Gains Curve

![Cumulative Gains Curve](cumulative_gains_curve.png)


## Lift Curve

![Lift Curve](lift_curve.png)



[<< Go back](../README.md)
