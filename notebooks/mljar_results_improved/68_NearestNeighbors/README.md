# Summary of 68_NearestNeighbors

[<< Go back](../README.md)


## k-Nearest Neighbors (Nearest Neighbors)
- **n_jobs**: -1
- **n_neighbors**: 7
- **weights**: uniform
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

4.7 seconds

## Metric details
|           |      score |   threshold |
|:----------|-----------:|------------:|
| logloss   | 0.054184   |  nan        |
| auc       | 0.514002   |  nan        |
| f1        | 0.0155039  |    0        |
| accuracy  | 0.996476   |    0.142857 |
| precision | 0.00943396 |    0        |
| recall    | 0.0434783  |    0        |
| mcc       | 0.0131278  |    0        |


## Metric details with threshold from accuracy metric
|           |        score |   threshold |
|:----------|-------------:|------------:|
| logloss   |  0.054184    |  nan        |
| auc       |  0.514002    |  nan        |
| f1        |  0           |    0.142857 |
| accuracy  |  0.996476    |    0.142857 |
| precision |  0           |    0.142857 |
| recall    |  0           |    0.142857 |
| mcc       | -0.000705374 |    0.142857 |


## Confusion matrix (at threshold=0.142857)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6787 |                1 |
| Labeled as 1 |               23 |                0 |

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
