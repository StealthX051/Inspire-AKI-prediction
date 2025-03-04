from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


def performance_dict(y_binary_test, y_pred, y_prob, bool_print=False, plot=False):
    print("----------")
    rtn = {}
    report = classification_report(y_binary_test, y_pred, output_dict=True)
    rtn['Precision'] = report['True']['precision']
    rtn['Sensitivity'] = report['True']['recall']
    rtn["Accuracy"] = accuracy_score(y_binary_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_binary_test, y_prob)
    rtn["rc_auc"] = auc(fpr, tpr)
    prec, rec, thresholds = precision_recall_curve(y_binary_test, y_prob)
    rtn["pr_auc"] = auc(rec, prec)
    rtn['Specificity'] = report['False']['recall']
    rtn['Negative Predictive Value'] = report['False']['precision']
    rtn['F1 Score'] = report['True']['f1-score']

    if bool_print:
        for item in rtn.items():
            print(item)
    
    if plot:
        # Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {rtn["rc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Using Regression Output)')
        plt.legend(loc='lower right')
        plt.show()

        # Plot PR curve
        plt.figure()
        plt.plot(prec, rec, color='yellow', lw=2, label=f'PR curve (area = {rtn["pr_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower right')
        plt.show()
        print("----------")

    return rtn