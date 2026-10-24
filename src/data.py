from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load(test_size=0.25, seed=42):
    """Wisconsin diagnostic data. sklearn codes malignant as 0, I flip it so malignant = 1."""
    ds = load_breast_cancer(as_frame=True)
    y = 1 - ds.target
    return train_test_split(ds.data, y, test_size=test_size, stratify=y, random_state=seed)
