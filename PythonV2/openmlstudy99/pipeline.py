import random

import numpy as np
import scipy
import scipy.stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from openmlstudy99.distributions import loguniform, loguniform_int, random_size_loguniform_int
from openmlstudy99.preprocessing import CategoricalImputer, NumericalImputer


class EstimatorFactory:

    def __init__(self, n_folds_inner_cv, n_iter, n_jobs=-1):
        scoring = 'accuracy'
        error_score = 'raise'
        self.grid_arguments = dict(
            scoring=scoring,
            error_score=error_score,
            cv=n_folds_inner_cv,
            n_jobs=n_jobs,
            verbose=100,
        )
        self.rs_arguments = dict(
            scoring=scoring,
            error_score=error_score,
            cv=n_folds_inner_cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            verbose=100,
        )

        self.estimator_mapping = {
            'naive_bayes': self.get_naive_bayes,
            'decision_tree': self.get_decision_tree,
            'logreg': self.get_logistic_regression,
            'gradient_boosting': self.get_gradient_boosting,
            'svm': self.get_svm,
            'random_forest': self.get_random_forest,
            'mlp': self.get_mlp,
            'knn': self.get_knn,
            'xgboost': self.get_xgboost,
        }

    @staticmethod
    def _get_pipeline(nominal_indices, num_features, sklearn_model):
        with_mean = False if len(nominal_indices) > 0 else True
        numerical_indices = [i for i in range(num_features) if i not in nominal_indices]

        numerical_preprocessing = (
            'NumericalPreprocessing',
            Pipeline([
                ('NumericalImputer', NumericalImputer()),
                ('Scaler', StandardScaler(with_mean=with_mean)),
            ]),
            numerical_indices,
        )

        categorical_preprocessing = (
            'CategoricalPreprocessing',
            Pipeline([
                ('CategoricalImputer', CategoricalImputer()),
                (
                    'OneHotEncoder',
                    OneHotEncoder(
                        categories='auto',
                        sparse=True,
                        handle_unknown='ignore',
                    )
                ),
            ]),
            nominal_indices,
        )

        joint_preprocessing = (
            'Preprocessing',
            ColumnTransformer(
                [
                    numerical_preprocessing,
                    categorical_preprocessing,
                ],
                n_jobs=1
            ),
        )

        if len(nominal_indices) > 0 and len(numerical_indices) > 0:
            steps = [joint_preprocessing]
        elif len(nominal_indices) > 0:
            steps = [categorical_preprocessing[:2]]
        else:
            steps = [numerical_preprocessing[:2]]
        steps.extend([
            ('VarianceThreshold', VarianceThreshold()),
            ('Estimator', sklearn_model),
        ])

        return Pipeline(steps=steps)

    def get_naive_bayes(self, nominal_indices, num_features):
        return self._get_pipeline(nominal_indices, num_features, GaussianNB())

    def get_decision_tree(self, nominal_indices, num_features):
        param_dist = {
            'Estimator__min_samples_split': loguniform_int(low=2**1, high=2**7),
            'Estimator__min_samples_leaf': loguniform_int(low=2**0, high=2**6),
            'Estimator__ccp_alpha': loguniform(low=10**-4, high=10**-1),
        }
        decision_tree = self._get_pipeline(
            nominal_indices,
            num_features,
            DecisionTreeClassifier(),
        )
        random_search = RandomizedSearchCV(
            estimator=decision_tree,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return random_search

    def get_svm(self, nominal_indices, num_features):
        param_dist = {
            'Estimator__C': loguniform(low=2**-12, high=2**12),
            'Estimator__gamma': loguniform(low=2**-12, high=2**12),
        }
        svm = self._get_pipeline(
            nominal_indices,
            num_features,
            SVC(kernel='rbf', probability=True, cache_size=2000),
        )
        random_search = RandomizedSearchCV(
            estimator=svm,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return random_search

    def get_gradient_boosting(self, nominal_indices, num_features):
        param_dist = {
            'Estimator__learning_rate': loguniform(low=10**-4, high=10**-1),
            'Estimator__max_depth': scipy.stats.randint(1, 5 + 1),
            'Estimator__n_estimators': scipy.stats.randint(500, 10001),
        }
        boosting = self._get_pipeline(
            nominal_indices,
            num_features,
            GradientBoostingClassifier(),
        )
        random_search = RandomizedSearchCV(
            estimator=boosting,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return random_search

    def get_knn(self, nominal_indices, num_features):
        param_dist = {'Estimator__n_neighbors': list(range(1, 50 + 1))}
        knn = self._get_pipeline(
            nominal_indices,
            num_features,
            KNeighborsClassifier(),
        )
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_dist,
            **self.grid_arguments
        )
        return grid_search

    def get_mlp(self, nominal_indices, num_features):
        # Simplification of the search space to a single layer due to scikit-learn issue 15568
        # https://github.com/scikit-learn/scikit-learn/issues/15568
        param_dist = {
            'Estimator__hidden_layer_sizes': random_size_loguniform_int(
                low=2**5,
                high=2**11,
                min_size=1,
                max_size=2,
            ),
            'Estimator__learning_rate_init': loguniform(low=10**-5, high=10**0),
            'Estimator__alpha': loguniform(low=10**-7, high=10**-4),
            'Estimator__momentum': scipy.stats.uniform(loc=0.1, scale=0.8),
        }

        mlp = self._get_pipeline(
            nominal_indices,
            num_features,
            MLPClassifier(activation='tanh', solver='adam', max_iter=200),
        )
        random_search = RandomizedSearchCV(
            estimator=mlp,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return random_search

    def get_logistic_regression(self, nominal_indices, num_features):
        param_dist = {'Estimator__C': loguniform(low=2**-12, high=2**12)}
        logreg = self._get_pipeline(
            nominal_indices,
            num_features,
            LogisticRegression(
                n_jobs=1,
                solver='saga',
                multi_class='multinomial',
                max_iter=500,
            ),
        )
        random_search = RandomizedSearchCV(
            estimator=logreg,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return random_search

    def get_random_forest(self, nominal_indices, num_features):
        lower = np.floor(np.power(num_features, 0.1))
        upper = np.ceil(np.power(num_features, 0.9))
        param_dist = {
            'Estimator__max_features': loguniform_int(low=lower, high=upper),
        }
        randomforest = self._get_pipeline(
            nominal_indices,
            num_features,
            RandomForestClassifier(n_estimators=500, n_jobs=1),
        )
        grid_search = RandomizedSearchCV(
            estimator=randomforest,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return grid_search

    def get_xgboost(self, nominal_indices, num_features):

        param_dist = {
            'Estimator__max_depth': scipy.stats.randint(1, 20 + 1),
            'Estimator__learning_rate': loguniform(low=10**-4, high=10**-1),
            'Estimator__booster': ['gbtree', 'gblinear', 'dart'],
            'Estimator__subsample': scipy.stats.uniform(loc=0.1, scale=0.9),
            'Estimator__min_child_weight': scipy.stats.randint(1, 20 + 1),
            'Estimator__colsample_by_tree': scipy.stats.uniform(0.1, 0.9),
            'Estimator__colsample_by_level': scipy.stats.uniform(0.1, 0.9),
            'Estimator__reg_alpha': loguniform(10**-11, 10**-2),
            'Estimator__reg_lambda': loguniform(10**-11, 10**-2),
            'Estimator__sample_type': ['uniform', 'weighted'],
            'Estimator__normalize_type': ['forest', 'tree'],
            'Estimator__rate_drop': loguniform(10**-11, 1 - (10**-11)),
        }
        xgb = self._get_pipeline(
            nominal_indices,
            num_features,
            XGBClassifier(n_estimators=512, n_jobs=1),
        )
        grid_search = RandomizedSearchCV(xgb, param_dist, **self.rs_arguments)
        return grid_search

    def get_random_estimator(self, nominal_indices, num_features):
        estimator = random.choice(list(self.estimator_mapping.values()))
        return estimator(nominal_indices, num_features)

    def get_all_flows(self, nominal_indices, num_features):
        return [estimator(nominal_indices, num_features)
                for estimator in self.estimator_mapping.values()]

    def get_flow_mapping(self):
        return self.estimator_mapping
