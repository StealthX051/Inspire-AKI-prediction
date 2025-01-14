# Summary of 93_NeuralNetwork

[<< Go back](../README.md)


## Neural Network
- **n_jobs**: -1
- **dense_1_size**: 64
- **dense_2_size**: 16
- **learning_rate**: 0.08
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

15.6 seconds

## Metric details
|           |     score |      threshold |
|:----------|----------:|---------------:|
| logloss   | 0.0163769 | nan            |
| auc       | 0.942443  | nan            |
| f1        | 0.118812  |   0.0589129    |
| accuracy  | 0.986933  |   0.0589129    |
| precision | 0.0769231 |   0.0589129    |
| recall    | 1         |   1.53064e-299 |
| mcc       | 0.178115  |   0.0248358    |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0163769 | nan         |
| auc       | 0.942443  | nan         |
| f1        | 0.118812  |   0.0589129 |
| accuracy  | 0.986933  |   0.0589129 |
| precision | 0.0769231 |   0.0589129 |
| recall    | 0.26087   |   0.0589129 |
| mcc       | 0.136452  |   0.0589129 |


## Confusion matrix (at threshold=0.058913)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6716 |               72 |
| Labeled as 1 |               17 |                6 |

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
