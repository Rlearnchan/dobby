import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class Classifier:
    def __init__(self, model_type="logistic"):

        self.best_params = None

        # raw models
        if model_type == "logistic":
            self.model = LogisticRegression()
            self.param_grid = {
                "C": [0.1, 1, 10],  # regularization strength
                "solver": ["liblinear", "saga"],  # optimization algorithm
            }
        elif model_type == "rf":
            self.model = RandomForestClassifier()
            self.param_grid = {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
        elif model_type == "xgb":
            self.model = XGBClassifier()
            self.param_grid = {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
        # elif model_type == 'lightgbm':
        #     self.model = lgb.LGBMClassifier(force_row_wise=True)
        #     self.param_grid = {
        #         'n_estimators': [50, 100],
        #         'learning_rate': [0.01, 0.1]
        #     }

        # regarding class imbalance
        elif model_type == "logistic_balance":
            self.model = LogisticRegression(class_weight="balanced")
            self.param_grid = {"C": [0.1, 1, 10], "solver": ["liblinear", "saga"]}
        elif model_type == "rf_balance":
            self.model = RandomForestClassifier(class_weight="balanced")
            self.param_grid = {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
        # elif model_type == "xgb_balance":
        #     self.model = XGBClassifier()
        #     self.param_grid = {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
        # elif model_type == 'lightgbm_balance':
        #     self.model = lgb.LGBMClassifier(scale_pos_weight=3, force_row_wise=True)
        #     self.param_grid = {
        #         'n_estimators': [50, 100],
        #         'learning_rate': [0.01, 0.1]
        #     }

    def fit(self, X, y):
        grid_search = GridSearchCV(self.model, self.param_grid, cv=5)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions, zero_division=1)
        return accuracy, report
