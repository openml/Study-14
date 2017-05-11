"""Script to run a task on a learner. Called by test_time_and_memory_requirements.py"""


import os
import resource
import sys
import warnings

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

model_name = sys.argv[1]
task_id = int(sys.argv[2])

this_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_directory, '..', 'run_scripts'))
import estimators

limit = 16000 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

models = {'decision_tree': sklearn.tree.DecisionTreeClassifier(
    min_samples_split=2, min_samples_leaf=1),
         'gradient_boosting': sklearn.ensemble.GradientBoostingClassifier(
             max_depth=5, n_estimators=10000),
         'knn': sklearn.neighbors.KNeighborsClassifier(n_neighbors=50),
         'logreg': sklearn.linear_model.LogisticRegression(),
         'mlp': sklearn.neural_network.MLPClassifier(
             hidden_layer_sizes=(2048, 2048), max_iter=1000),
         'naive_bayes': sklearn.naive_bayes.GaussianNB(),
         'random_forest': sklearn.ensemble.RandomForestClassifier(
             n_estimators=500),
         'svm': sklearn.svm.SVC()}

task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y()
_, nominal_indices = task.get_dataset().get_data(
    return_categorical_indicator=True)
nominal_indices = nominal_indices[:-1]

model = models[model_name]
steps = estimators._get_preprocessors(nominal_indices) + [('model', model)]
pipeline = sklearn.pipeline.Pipeline(steps=steps)

for rep in task.iterate_repeats():
    for fold in rep:
        train_indices, test_indices = fold
        trainX = X[train_indices]
        trainY = y[train_indices]
        testX = X[test_indices]
        testY = y[test_indices]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            pipeline.fit(trainX, trainY)
        break