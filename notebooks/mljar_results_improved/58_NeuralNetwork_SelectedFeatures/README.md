# Summary of 58_NeuralNetwork_SelectedFeatures

[<< Go back](../README.md)


## Neural Network
- **n_jobs**: -1
- **dense_1_size**: 64
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

15.8 seconds

## Metric details
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0182846 | nan         |
| auc       | 0.923913  | nan         |
| f1        | 0.138614  |   0.0528911 |
| accuracy  | 0.987227  |   0.0528911 |
| precision | 0.0897436 |   0.0528911 |
| recall    | 1         |   0         |
| mcc       | 0.160238  |   0.0528911 |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0182846 | nan         |
| auc       | 0.923913  | nan         |
| f1        | 0.138614  |   0.0528911 |
| accuracy  | 0.987227  |   0.0528911 |
| precision | 0.0897436 |   0.0528911 |
| recall    | 0.304348  |   0.0528911 |
| mcc       | 0.160238  |   0.0528911 |


## Confusion matrix (at threshold=0.052891)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6717 |               71 |
| Labeled as 1 |               16 |                7 |

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
