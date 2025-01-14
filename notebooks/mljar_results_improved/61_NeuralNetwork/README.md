# Summary of 61_NeuralNetwork

[<< Go back](../README.md)


## Neural Network
- **n_jobs**: -1
- **dense_1_size**: 64
- **dense_2_size**: 4
- **learning_rate**: 0.01
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

13.8 seconds

## Metric details
|           |     score |      threshold |
|:----------|----------:|---------------:|
| logloss   | 0.0157496 | nan            |
| auc       | 0.960218  | nan            |
| f1        | 0.217822  |   0.194642     |
| accuracy  | 0.988401  |   0.194642     |
| precision | 0.141026  |   0.194642     |
| recall    | 1         |   1.91345e-119 |
| mcc       | 0.270973  |   0.105642     |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0157496 |  nan        |
| auc       | 0.960218  |  nan        |
| f1        | 0.217822  |    0.194642 |
| accuracy  | 0.988401  |    0.194642 |
| precision | 0.141026  |    0.194642 |
| recall    | 0.478261  |    0.194642 |
| mcc       | 0.255383  |    0.194642 |


## Confusion matrix (at threshold=0.194642)
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
