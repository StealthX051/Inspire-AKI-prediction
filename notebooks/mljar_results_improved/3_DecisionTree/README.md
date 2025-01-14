# Summary of 3_DecisionTree

[<< Go back](../README.md)


## Decision Tree
- **n_jobs**: -1
- **criterion**: gini
- **max_depth**: 4
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

13.3 seconds

## Metric details
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0169609 |  nan        |
| auc       | 0.815826  |  nan        |
| f1        | 0.320988  |    0.025825 |
| accuracy  | 0.991925  |    0.025825 |
| precision | 0.224138  |    0.025825 |
| recall    | 0.956522  |    0        |
| mcc       | 0.352666  |    0.025825 |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0169609 |  nan        |
| auc       | 0.815826  |  nan        |
| f1        | 0.320988  |    0.025825 |
| accuracy  | 0.991925  |    0.025825 |
| precision | 0.224138  |    0.025825 |
| recall    | 0.565217  |    0.025825 |
| mcc       | 0.352666  |    0.025825 |


## Confusion matrix (at threshold=0.025825)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6743 |               45 |
| Labeled as 1 |               10 |               13 |

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
