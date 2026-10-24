Malignant is the positive class.

| model | cv auc | test auc | precision @0.5 | recall @0.5 |
|---|---|---|---|---|
| logreg | 0.992 | 0.996 | 0.980 | 0.925 |
| random_forest | 0.986 | 0.996 | 1.000 | 0.925 |
| svc_rbf | 0.993 | 0.997 | 1.000 | 0.962 |

Best by cross-validation: **svc_rbf**. Threshold lowered to 0.13 to aim for 98% recall on malignant cases: test precision 0.929, test recall 0.981.
