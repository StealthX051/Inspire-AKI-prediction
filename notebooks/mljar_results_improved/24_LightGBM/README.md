# Summary of 24_LightGBM

[<< Go back](../README.md)


## LightGBM
- **n_jobs**: -1
- **objective**: binary
- **num_leaves**: 63
- **learning_rate**: 0.2
- **feature_fraction**: 0.5
- **bagging_fraction**: 1.0
- **min_data_in_leaf**: 30
- **metric**: auc
- **custom_eval_metric_name**: None
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

11.8 seconds

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0285069 | nan           |
| auc       | 0.967968  | nan           |
| f1        | 0.237624  |   7.35926e-05 |
| accuracy  | 0.988695  |   7.35926e-05 |
| precision | 0.153846  |   7.35926e-05 |
| recall    | 1         |   2.85442e-17 |
| mcc       | 0.279169  |   7.35926e-05 |


## Metric details with threshold from accuracy metric
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0285069 | nan           |
| auc       | 0.967968  | nan           |
| f1        | 0.237624  |   7.35926e-05 |
| accuracy  | 0.988695  |   7.35926e-05 |
| precision | 0.153846  |   7.35926e-05 |
| recall    | 0.521739  |   7.35926e-05 |
| mcc       | 0.279169  |   7.35926e-05 |


## Confusion matrix (at threshold=7.4e-05)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |             6722 |               66 |
| Labeled as 1 |               11 |               12 |

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



## SHAP Importance
![SHAP Importance](shap_importance.png)

## SHAP Dependence plots

### Dependence (Fold 1)
![SHAP Dependence from Fold 1](learner_fold_0_shap_dependence.png)

## SHAP Decision plots


[<< Go back](../README.md)
