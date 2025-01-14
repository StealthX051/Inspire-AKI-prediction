# Summary of 8_Default_RandomForest

[<< Go back](../README.md)


## Random Forest
- **n_jobs**: -1
- **criterion**: gini
- **max_features**: 0.9
- **min_samples_split**: 30
- **max_depth**: 4
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

10.5 seconds

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0137167 | nan           |
| auc       | 0.929188  | nan           |
| f1        | 0.316832  |   0.0552285   |
| accuracy  | 0.989869  |   0.0552285   |
| precision | 0.205128  |   0.0552285   |
| recall    | 1         |   0.000897675 |
| mcc       | 0.374313  |   0.0552285   |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0137167 | nan         |
| auc       | 0.929188  | nan         |
| f1        | 0.316832  |   0.0552285 |
| accuracy  | 0.989869  |   0.0552285 |
| precision | 0.205128  |   0.0552285 |
| recall    | 0.695652  |   0.0552285 |
| mcc       | 0.374313  |   0.0552285 |


## Confusion matrix (at threshold=0.055229)
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
