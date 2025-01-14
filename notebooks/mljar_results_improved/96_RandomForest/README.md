# Summary of 96_RandomForest

[<< Go back](../README.md)


## Random Forest
- **n_jobs**: -1
- **criterion**: gini
- **max_features**: 1.0
- **min_samples_split**: 30
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

10.6 seconds

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0127348 | nan           |
| auc       | 0.965764  | nan           |
| f1        | 0.316832  |   0.0574909   |
| accuracy  | 0.989869  |   0.0574909   |
| precision | 0.205128  |   0.0574909   |
| recall    | 1         |   0.000319696 |
| mcc       | 0.374313  |   0.0574909   |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0127348 | nan         |
| auc       | 0.965764  | nan         |
| f1        | 0.316832  |   0.0574909 |
| accuracy  | 0.989869  |   0.0574909 |
| precision | 0.205128  |   0.0574909 |
| recall    | 0.695652  |   0.0574909 |
| mcc       | 0.374313  |   0.0574909 |


## Confusion matrix (at threshold=0.057491)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6726 |               62 |
| Labeled as 1 |                7 |               16 |

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
