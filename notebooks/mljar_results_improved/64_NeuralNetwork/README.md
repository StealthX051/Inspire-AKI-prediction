# Summary of 64_NeuralNetwork

[<< Go back](../README.md)


## Neural Network
- **n_jobs**: -1
- **dense_1_size**: 16
- **dense_2_size**: 32
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

12.8 seconds

## Metric details
|           |     score |      threshold |
|:----------|----------:|---------------:|
| logloss   | 0.0167186 | nan            |
| auc       | 0.941108  | nan            |
| f1        | 0.0537772 |   0.0227334    |
| accuracy  | 0.996476  |   0.0269065    |
| precision | 0.0277045 |   0.0227334    |
| recall    | 1         |   2.40077e-296 |
| mcc       | 0.148935  |   0.0182998    |


## Metric details with threshold from accuracy metric
|           |        score |   threshold |
|:----------|-------------:|------------:|
| logloss   |  0.0167186   | nan         |
| auc       |  0.941108    | nan         |
| f1        |  0           |   0.0269065 |
| accuracy  |  0.996476    |   0.0269065 |
| precision |  0           |   0.0269065 |
| recall    |  0           |   0.0269065 |
| mcc       | -0.000705374 |   0.0269065 |


## Confusion matrix (at threshold=0.026906)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6787 |                1 |
| Labeled as 1 |               23 |                0 |

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
