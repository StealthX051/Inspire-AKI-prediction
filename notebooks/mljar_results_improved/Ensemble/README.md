# Summary of Ensemble

[<< Go back](../README.md)


## Ensemble structure
| Model                        |   Weight |
|:-----------------------------|---------:|
| 16_Xgboost                   |        2 |
| 1_DecisionTree               |        1 |
| 26_LightGBM                  |        4 |
| 29_CatBoost_SelectedFeatures |        2 |
| 64_NeuralNetwork             |        1 |
| 75_CatBoost                  |        1 |
| 77_CatBoost_SelectedFeatures |        2 |

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0111883 | nan           |
| auc       | 0.988381  | nan           |
| f1        | 0.336634  |   0.0318892   |
| accuracy  | 0.990163  |   0.0318892   |
| precision | 0.217949  |   0.0318892   |
| recall    | 1         |   0.000108363 |
| mcc       | 0.3981    |   0.0318892   |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0111883 | nan         |
| auc       | 0.988381  | nan         |
| f1        | 0.336634  |   0.0318892 |
| accuracy  | 0.990163  |   0.0318892 |
| precision | 0.217949  |   0.0318892 |
| recall    | 0.73913   |   0.0318892 |
| mcc       | 0.3981    |   0.0318892 |


## Confusion matrix (at threshold=0.031889)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6727 |               61 |
| Labeled as 1 |                6 |               17 |

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
