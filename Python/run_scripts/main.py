import argparse
import os
import pickle

# Necessary to use dask parallel
import distributed.joblib
from distributed import Executor
import numpy as np
import sklearn.externals.joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import scipy.sparse

import openml
import openml.tasks

import openml2016.pipeline as pipeline
import openml2016.util as util


def run_task(seed, tmp_dir, task_id, estimator_name, n_iter, n_jobs,
             n_folds_inner_cv, scheduler_host):

    task = openml.tasks.get_task(task_id)

    # Inform the OneHotEncoder about categorical features, use feature
    # indices instead of boolean mask because of a more compact string
    # representation
    dataset = task.get_dataset()
    X, _, categorical_indicator = dataset.get_data(
        target=task.target_name, return_categorical_indicator=True)

    categorical_indicator = np.array(categorical_indicator)
    categorical_features = [int(x) for x in np.where(categorical_indicator)[0]]
    numerical_features = [int(x) for x in np.where(~categorical_indicator)[0]]

    print('Categorical features', categorical_features)
    print('Numerical features', numerical_features)

    assert len(categorical_features + numerical_features) == X.shape[1]
    assert len(np.unique(categorical_features + numerical_features)) == X.shape[1]

    if len(categorical_features) > 0 and len(numerical_features) > 0:
        version = 'mixed'
    elif len(categorical_features) == 0:
        version = 'numerical'
    elif len(numerical_features) == 0:
        version = 'categorical'
    else:
        raise Exception()

    num_features = X.shape[0]

    if estimator_name == 'SVC':
        param_dist, clf = pipeline.get_SVM(version)
    elif estimator_name == 'DecisionTreeClassifier':
        param_dist, clf = pipeline.get_decision_tree(version)
    elif estimator_name == 'GradientBoostingClassifier':
        param_dist, clf = pipeline.get_gradient_boosting(version)
    elif estimator_name == 'KNeighborsClassifier':
        param_dist, clf = pipeline.get_kNN(version)
    elif estimator_name == 'NaiveBayes':
        param_dist, clf = pipeline.get_naive_bayes(version)
    elif estimator_name == 'MLPClassifier':
        param_dist, clf = pipeline.get_neural_network(version)
    elif estimator_name == 'LogisticRegression':
        param_dist, clf = pipeline.get_logistic_regression(version)
    elif estimator_name == 'RandomForestClassifier':
        param_dist, clf = pipeline.get_random_forest(version, num_features)
    else:
        raise ValueError('Unknown classifier %s.' % args.classifier)

    kfold = StratifiedKFold(n_splits=n_folds_inner_cv, shuffle=True,
                            random_state=1)

    if scipy.sparse.isspmatrix(X) or np.sum(categorical_features) > 0:
        standard_scaler_center = False
    else:
        standard_scaler_center = True

    if n_iter > 0:
        estimator = RandomizedSearchCV(estimator=clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter,
                                       n_jobs=n_jobs,
                                       pre_dispatch='2*n_jobs',
                                       iid=True,
                                       cv=kfold,
                                       refit=True,
                                       verbose=10,
                                       random_state=1,
                                       error_score=0.0)

        if len(categorical_features) != 0 and len(numerical_features) != 0:
            estimator.set_params(
                estimator__FeatureUnion__numerical__ItemSelector__indices=
                categorical_features)
            estimator.set_params(
                estimator__FeatureUnion__numerical__ItemSelector__indices=
                numerical_features)

        if not standard_scaler_center and 'estimator__scaler' in \
                estimator.get_params():
            estimator.set_params(estimator__scaler__with_mean=False)

        print([item for item in estimator.get_params().items() if 'n_jobs' in item[0]])
    else:
        estimator = clf

        if not standard_scaler_center and 'scaler' in estimator.get_params():
            estimator.set_params(scaler__with_mean=False)

        if len(categorical_features) != 0 and len(numerical_features) != 0:
            estimator.set_params(
                Preprocessing__numerical__ItemSelector__indices=
                numerical_features)
            estimator.set_params(
                Preprocessing__categorical__ItemSelector__indices=
                categorical_features)

    print('Running task with ID %d.' % task_id)
    print('Arguments: random search iterations: %d, inner CV folds %d, '
          'n parallel jobs: %d, seed %d' %
          (n_iter, n_folds_inner_cv, n_jobs, seed))
    print('Model: %s' % str(estimator))

    import time
    start_time = time.time()

    # TODO generate a flow first
    if scheduler_host is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='sklearn\.externals\.joblib\.parallel')
            run = openml.runs.run_task(task, estimator)

        try:
            print(util.report(estimator.cv_results_))
        except:
            pass
    else:
        # Register with scikit-leran joblib since scikit-learn uses the builtin
        # version to distribute it's work
        print('Using dask parallel with host %s' % scheduler_host)
        with sklearn.externals.joblib.parallel_backend('distributed',
                                                       scheduler_host=scheduler_host):
            run = openml.runs.run_task(task, estimator)

    end_time = time.time()
    print('READTHIS', estimator_name, task_id, end_time-start_time)

    output_file = os.path.join(tmp_dir, '%d_%s_%d.pkl' % (
        task_id, estimator_name, seed))

    #return_code, return_value = run.publish()
    #if return_code != 200:
    #    print(return_value)
    #    exit(1)

    with open(output_file, 'wb') as fh:
        pickle.dump(run, fh)
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', required=True,
                        choices=['SVC', 'DecisionTreeClassifier',
                                 'GradientBoostingClassifier',
                                 'KNeighborsClassifier', 'NaiveBayes',
                                 'MLPClassifier',
                                 'LogisticRegression', 'RandomForestClassifier'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--task_id', required=True, type=int)
    parser.add_argument('--tmp_dir', required=True, type=str)
    parser.add_argument('--n_iter_inner_loop', default=200, type=int,
                        help='Number of iterations random search.')
    parser.add_argument('--n_folds_inner_cv', default=10, type=int,
                        help='Number of folds for inner CV')
    parser.add_argument('--n_jobs', default=-1, type=int,
                        help='Number of cores used by random search. Defaults '
                             'to all (-1).')
    parser.add_argument('--openml_server', default=openml.config.server)
    parser.add_argument('--scheduler_host', default=None, type=str,
                        help='Use distributed backend if specified')

    args = parser.parse_args()

    openml_server = args.openml_server
    openml.config.server = openml_server

    seed = args.seed
    tmp_dir = args.tmp_dir
    classifier = args.classifier
    n_iter_random_search = args.n_iter_inner_loop
    n_folds_inner_cv = args.n_folds_inner_cv
    n_jobs = args.n_jobs
    task_id = args.task_id
    scheduler_host = args.scheduler_host

    run = run_task(seed=seed,
                   tmp_dir=tmp_dir,
                   task_id=task_id,
                   estimator_name=classifier,
                   n_iter=n_iter_random_search,
                   n_jobs=n_jobs,
                   n_folds_inner_cv=n_folds_inner_cv,
                   scheduler_host=scheduler_host)
    print(run)

