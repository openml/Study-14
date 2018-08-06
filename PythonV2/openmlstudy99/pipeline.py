import random
import numpy as np
import scipy

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from openmlstudy99.distributions import loguniform, loguniform_int
from openmlstudy99.preprocessing import (
    CategoricalImputer,
    NumericalImputer,
    CategoricalFeatureSelector,
    NumericalFeatureSelector,
)


supported_classifiers = [
    #'decision_tree',
    #'gradient_boosting',
    #'knn',
    'logreg',
    #'mlp',
    #'naive_bayes',
    #'random_forest',
    #'svm',
    #'xgboost',
]


class EstimatorFactory():

    def __init__(self, n_folds_inner_cv=3, n_iter=200, n_jobs=-1):
        scoring = 'accuracy'
        error_score = 'raise'
        self.grid_arguments = dict(
            scoring=scoring,
            error_score=error_score,
            cv=n_folds_inner_cv,
            n_jobs=n_jobs,
        )
        self.rs_arguments = dict(
            scoring=scoring,
            error_score=error_score,
            cv=n_folds_inner_cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            verbose=10,
        )

        self.all_estimators = [
            self.get_naive_bayes,
            self.get_decision_tree,
            self.get_logistic_regression,
            self.get_gradient_boosting,
            self.get_svm,
            self.get_random_forest,
            self.get_mlp,
            self.get_knn,
            self.get_xgboost,
        ]

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
        if len(nominal_indices) > 0:
            with_mean = False
        else:
            with_mean = True
        numerical_indices = [i for i in range(num_features)
                             if i not in nominal_indices]

        numerical_preprocessing = ('NumericalPreprocessing', Pipeline([
            ('SelectNumerical', NumericalFeatureSelector(
                numerical_indices,
            )),
            (
                'NumericImputationFU',
                FeatureUnion([
                    (
                        'MissingIndicator',
                        MissingIndicator(
                            error_on_new=False
                        ),
                    ),
                    (
                        'NumericalImputer',
                        NumericalImputer(),
                    )
                ])
            ),
            ('Scaler', StandardScaler(with_mean=with_mean)),
        ]))

        categorical_preprocessing = ('CategoricalPreprocessing', Pipeline([
            ('SelectCategorical', CategoricalFeatureSelector(
                nominal_indices,
            )),
            (
                'CategoricalImputer',
                CategoricalImputer()
            ),
            (
                'OneHotEncoder',
                OneHotEncoder(
                    categories='auto',
                    sparse=True,
                    handle_unknown='ignore',
                )
            ),
        ]))

        joint_preprocessing = ('Preprocessing', FeatureUnion([
            numerical_preprocessing,
            categorical_preprocessing,
        ]))

        if len(nominal_indices) > 0 and len(numerical_indices) > 0:
            steps = [joint_preprocessing]
        elif len(nominal_indices) > 0:
            steps = [categorical_preprocessing]
        else:
            steps = [numerical_preprocessing]
        steps.extend([
            ('VarianceThreshold', VarianceThreshold()),
            ('Estimator', sklearn_model),
        ])

        return Pipeline(steps=steps)

    def get_naive_bayes(self, nominal_indices, num_features):
        return self._get_pipeline(nominal_indices, num_features, GaussianNB())

    def get_decision_tree(self, nominal_indices, num_features):
        param_dist = {
            'Estimator__min_samples_split': loguniform_int(base=2, low=2**1, high=2**7),
            'Estimator__min_samples_leaf': loguniform_int(base=2, low=2**0, high=2**6),
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
            'Estimator__C': loguniform(base=2, low=2**-12, high=2**12),
            'Estimator__gamma': loguniform(base=2, low=2**-12, high=2**12),
        }
        svm = self._get_pipeline(
            nominal_indices,
            num_features,
            SVC(kernel='rbf', probability=True),
        )
        random_search = RandomizedSearchCV(
            estimator=svm,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return random_search

    def get_gradient_boosting(self, nominal_indices, num_features):
        param_dist = {
            'Estimator__learning_rate': loguniform(base=10, low=10**-4, high=10**-1),
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
        param_dist = {
            'Estimator__hidden_layer_sizes': loguniform_int(base=2, low=2**5, high=2**11),
            'Estimator__learning_rate_init': loguniform(base=10, low=10**-5, high=10**0),
            'Estimator__alpha': loguniform(base=10, low=10**-7, high=10**-4),
            'Estimator__max_iter': scipy.stats.randint(2, 1001),
            'Estimator__momentum': scipy.stats.uniform(loc=0.1, scale=0.8),
        }
        mlp = self._get_pipeline(
            nominal_indices,
            num_features,
            MLPClassifier(activation='tanh', solver='adam'),
        )
        random_search = RandomizedSearchCV(
            estimator=mlp,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return random_search

    def get_logistic_regression(self, nominal_indices, num_features):
        param_dist = {'Estimator__C': loguniform(base=2, low=2**-12, high=2**12)}
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
        lower = np.power(num_features, 0.1)
        upper = np.power(num_features, 0.9)
        scale = upper - lower
        lower = lower / num_features
        scale = scale / num_features
        param_dist = {
            'Estimator__max_features':
                scipy.stats.uniform(loc=lower, scale=scale)
        }
        randomforest = self._get_pipeline(
            nominal_indices,
            num_features,
            RandomForestClassifier(n_estimators=500),
        )
        grid_search = RandomizedSearchCV(
            estimator=randomforest,
            param_distributions=param_dist,
            **self.rs_arguments
        )
        return grid_search

    def get_xgboost(self, nominal_indices, num_features):

        # TODO add categorical features here
        # TODO use sklearn searchspace as the default searchspace!

        param_dist = {'max_depth': scipy.stats.randint(1, 11)}
        xgb = XGBClassifier()
        grid_search = RandomizedSearchCV(xgb, param_dist, **self.rs_arguments)
        return grid_search

    def get_random_estimator(self, nominal_indices, num_features):
        estimator = random.choice(self.all_estimators)
        return estimator(nominal_indices, num_features)

    def get_all_flows(self, nominal_indices, num_features):
        return [estimator(nominal_indices, num_features)
                for estimator in self.all_estimators]

    def get_flow_mapping(self):
        return self.estimator_mapping
