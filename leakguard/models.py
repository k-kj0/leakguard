"""Interpretable baseline models with fold-local preprocessing."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_factories():

    return {

        "logistic_regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=0,
            ),
        ),

        "hist_gradient_boosting": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=40,
                max_leaf_nodes=4,
                learning_rate=0.03,
                l2_regularization=10.0,
                min_samples_leaf=800,
                class_weight="balanced",
                random_state=0,
            ),
        ),
    }
