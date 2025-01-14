# Summary of 75_CatBoost

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.05
- **depth**: 8
- **rsm**: 0.9
- **loss_function**: Logloss
- **eval_metric**: AUC
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

9.1 seconds

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0113812 | nan           |
| auc       | 0.983897  | nan           |
| f1        | 0.356436  |   0.0330458   |
| accuracy  | 0.990457  |   0.0330458   |
| precision | 0.230769  |   0.0330458   |
| recall    | 1         |   6.56473e-06 |
| mcc       | 0.421886  |   0.0330458   |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0113812 | nan         |
| auc       | 0.983897  | nan         |
| f1        | 0.356436  |   0.0330458 |
| accuracy  | 0.990457  |   0.0330458 |
| precision | 0.230769  |   0.0330458 |
| recall    | 0.782609  |   0.0330458 |
| mcc       | 0.421886  |   0.0330458 |


## Confusion matrix (at threshold=0.033046)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6728 |               60 |
| Labeled as 1 |                5 |               18 |

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
