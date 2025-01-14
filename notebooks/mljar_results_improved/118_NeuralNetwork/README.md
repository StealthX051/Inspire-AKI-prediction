# Summary of 118_NeuralNetwork

[<< Go back](../README.md)


## Neural Network
- **n_jobs**: -1
- **dense_1_size**: 32
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

13.6 seconds

## Metric details
|           |     score |      threshold |
|:----------|----------:|---------------:|
| logloss   | 0.0149483 | nan            |
| auc       | 0.944262  | nan            |
| f1        | 0.217822  |   0.0650117    |
| accuracy  | 0.988401  |   0.0650117    |
| precision | 0.141026  |   0.0650117    |
| recall    | 1         |   1.94759e-102 |
| mcc       | 0.255383  |   0.0650117    |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0149483 | nan         |
| auc       | 0.944262  | nan         |
| f1        | 0.217822  |   0.0650117 |
| accuracy  | 0.988401  |   0.0650117 |
| precision | 0.141026  |   0.0650117 |
| recall    | 0.478261  |   0.0650117 |
| mcc       | 0.255383  |   0.0650117 |


## Confusion matrix (at threshold=0.065012)
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
