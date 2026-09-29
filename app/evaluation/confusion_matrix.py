"""
Confusion Matrix Generator module formatting 2x2 confusion matrix outputs.
"""

from typing import Dict, Any
from sklearn.metrics import confusion_matrix


class ConfusionMatrixGenerator:
    """
    Generates structured 2x2 Confusion Matrix dictionary and markdown representation.
    """

    @staticmethod
    def generate(y_true: list, y_pred: list) -> Dict[str, Any]:
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        if cm.shape == (2, 2):
            tp = int(cm[0, 0])
            fn = int(cm[0, 1])
            fp = int(cm[1, 0])
            tn = int(cm[1, 1])
        else:
            tp, fn, fp, tn = 0, 0, 0, 0

        cm_table = f"""
                 Predicted WIN    Predicted LOSS
Actual WIN           {tp:<15} {fn:<15}
Actual LOSS          {fp:<15} {tn:<15}
"""
        return {
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "formatted_matrix": cm_table.strip(),
        }
