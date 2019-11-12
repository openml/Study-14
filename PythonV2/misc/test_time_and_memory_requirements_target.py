"""Script to run a task on a learner. Called by test_time_and_memory_requirements.py"""

import os
import resource
from typing import cast
import warnings

this_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(this_dir, '..'))

import sys
sys.path.append(parent_dir)

import numpy as np
import openml
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.tree
import sklearn.pipeline
import xgboost

import openml
import openml.extensions.sklearn
import openmlstudy99.pipeline

model_name = sys.argv[1]
task_id = int(sys.argv[2])


limit = 16000 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

task = openml.tasks.get_task(task_id)
task = cast(openml.tasks.OpenMLClassificationTask, task)
num_features = task.get_X_and_y()[0].shape[1]

# Check number of levels (for categorical features)
dataset = task.get_dataset()
X, _, _, _ = dataset.get_data()
total_num_categories = 0
for col_name, col in X.select_dtypes('category').iteritems():
    total_num_categories += len(col.dtype.categories)
if total_num_categories > 5000:
    raise ValueError('Dataset not supported by study 99!')

models = {'decision_tree': sklearn.tree.DecisionTreeClassifier(
             min_samples_split=2, min_samples_leaf=1, ccp_alpha=10e-4),
         'gradient_boosting': sklearn.ensemble.GradientBoostingClassifier(
             max_depth=5, n_estimators=10000),
         'knn': sklearn.neighbors.KNeighborsClassifier(n_neighbors=50),
         'logreg': sklearn.linear_model.LogisticRegression(C=2**12),
         'mlp': sklearn.neural_network.MLPClassifier(
             hidden_layer_sizes=(2048, 2048), max_iter=200, learning_rate_init=1, alpha=10e-4,
             momentum=0.9),
         'naive_bayes': sklearn.naive_bayes.GaussianNB(),
         'random_forest': sklearn.ensemble.RandomForestClassifier(
             n_estimators=500, max_features=int(np.rint(X.shape[1]**0.9))),
         'svm': sklearn.svm.SVC(C=2e-12, gamma=2e-12),
         'xgboost': xgboost.XGBClassifier(
             n_estimators=512,
             n_jobs=1,
             max_depth=20,
             booster='dart',
             subsample=1.0,
             min_child_weight=1,
             colsample_by_tree=1.0,
             colsample_by_level=1.0,
             reg_alpha=10**-11,
             reg_lambda=10**-11,
             sample_type='uniform',
             normalize_type='forest',
             rate_drop=1 - (10**-11),
         )}

# retrieve classifier
indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
classifierfactory = openmlstudy99.pipeline.EstimatorFactory(3, 200, 1)
pipeline = classifierfactory._get_pipeline(indices, num_features, models[model_name])

print(pipeline)

n_rep, n_folds, n_samples = task.get_split_dimensions()
for repeat in range(n_rep):
    for fold in range(n_folds):
        X, y = task.get_X_and_y(dataset_format='array')
        train_indices, test_indices = task.get_train_test_split_indices(fold=fold, repeat=repeat)
        trainX = X[train_indices]
        trainY = y[train_indices]
        testX = X[test_indices]
        testY = y[test_indices]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            pipeline.fit(trainX, trainY)
            pipeline.score(testX, testY)
        break
