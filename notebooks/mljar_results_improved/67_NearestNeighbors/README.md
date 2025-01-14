# Summary of 67_NearestNeighbors

[<< Go back](../README.md)


## k-Nearest Neighbors (Nearest Neighbors)
- **n_jobs**: -1
- **n_neighbors**: 7
- **weights**: distance
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

4.5 seconds

## Metric details
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0541664 |  nan        |
| auc       | 0.514079  |  nan        |
| f1        | 0.019802  |    0.136452 |
| accuracy  | 0.985465  |    0.136452 |
| precision | 0.0128205 |    0.136452 |
| recall    | 0.0434783 |    0        |
| mcc       | 0.017521  |    0.136452 |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0541664 |  nan        |
| auc       | 0.514079  |  nan        |
| f1        | 0.019802  |    0.136452 |
| accuracy  | 0.985465  |    0.136452 |
| precision | 0.0128205 |    0.136452 |
| recall    | 0.0434783 |    0.136452 |
| mcc       | 0.017521  |    0.136452 |


## Confusion matrix (at threshold=0.136452)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6711 |               77 |
| Labeled as 1 |               22 |                1 |

## Learning curves
![Learning curves](learning_curves.png)
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
