# breast-cancer-classifier

Screening only works if it doesn't miss people. This project predicts whether a tumour is malignant or benign from measurements taken off a fine needle aspirate (radius, texture, concavity and so on), with the decision threshold tuned so that very few malignant cases slip through.

Uses the Wisconsin Diagnostic dataset bundled with scikit-learn (569 samples, 30 features).

## Run it

```bash
pip install -r requirements.txt
python src/train.py
```

## What it does

1. Splits 75/25, stratified.
2. Compares logistic regression, random forest and an RBF SVM with 5-fold CV on ROC AUC.
3. For the winner, finds the highest probability threshold that still catches about 98% of malignant cases in cross-validated predictions on the training set.
4. Reports precision and recall on the untouched test set at that threshold.

## Notes

- Learning project. It isn't a medical tool and shouldn't be used as one.
- The dataset is small and clean, so almost every model scores very high. The interesting part is the threshold trade-off, not the AUC.
- With only ~140 test samples, one misclassified patient moves recall by a lot. Treat the numbers as rough.

## Results

From a run on the default split (seed 42):

Malignant is the positive class.

| model | cv auc | test auc | precision @0.5 | recall @0.5 |
|---|---|---|---|---|
| logreg | 0.992 | 0.996 | 0.980 | 0.925 |
| random_forest | 0.986 | 0.996 | 1.000 | 0.925 |
| svc_rbf | 0.993 | 0.997 | 1.000 | 0.962 |

Best by cross-validation: **svc_rbf**. Threshold lowered to 0.13 to aim for 98% recall on malignant cases: test precision 0.929, test recall 0.981.
