import argparse
import openml
import openmlstudy14.pipeline

import sklearn.externals.joblib


def run_task(seed, tmp_dir, task_id, estimator_name, n_iter, n_jobs,
             n_folds_inner_cv, scheduler_host):


    # retrieve dataset / task
    task = openml.tasks.get_task(task_id)
    indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])

    # retrieve classifier
    classifierfactory = openmlstudy14.pipeline.EstimatorFactory(n_folds_inner_cv, n_iter, n_jobs)
    if classifier == 'SVC':
        estimator = classifierfactory.get_svm(indices)
    elif classifier == 'DecisionTreeClassifier':
        estimator = classifierfactory.get_decision_tree(indices)
    elif classifier == 'GradientBoostingClassifier':
        estimator = classifierfactory.get_gradient_boosting(indices)
    elif classifier == 'KNeighborsClassifier':
        estimator = classifierfactory.get_knn(indices)
    elif classifier == 'NaiveBayes':
        estimator = classifierfactory.get_naive_bayes(indices)
    elif classifier == 'MLPClassifier':
        estimator = classifierfactory.get_mlp(indices)
    elif classifier == 'LogisticRegression':
        estimator = classifierfactory.get_logistic_regression(indices)
    elif classifier == 'RandomForestClassifier':
        estimator = classifierfactory.get_random_forest(indices)
    else:
        raise ValueError('Unknown classifier %s.' % args.classifier)

    print('Running task with ID %d.' % task_id)
    print('Arguments: random search iterations: %d, inner CV folds %d, '
          'n parallel jobs: %d, seed %d' %(n_iter, n_folds_inner_cv, n_jobs, seed))
    print('Model: %s' % str(estimator))
    flow = openml.flows.sklearn_to_flow(estimator)
    flow.tags.append('study_14')

    import time
    start_time = time.time()


    # TODO generate a flow first
    if scheduler_host is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='sklearn\.externals\.joblib\.parallel')
            run = openml.runs.run_flow_on_task(task, flow, seed=seed)
    else:
        # Register with scikit-leran joblib since scikit-learn uses the builtin
        # version to distribute it's work
        print('Using dask parallel with host %s' % scheduler_host)
        with sklearn.externals.joblib.parallel_backend('distributed', scheduler_host=scheduler_host):
            run = openml.runs.run_flow_on_task(task, flow, seed=seed)

    end_time = time.time()
    run_prime = run.publish()
    print('READTHIS', estimator_name, task_id, run_prime.run_id, end_time-start_time)

    # TODO: why do we need this? it crashes on
    # AttributeError: Can't pickle local object '_run_task_get_arffcontent.<locals>.<lambda>'
    # output_file = os.path.join(tmp_dir, '%d_%s_%d.pkl' % (task_id, estimator_name, seed))
    # with open(output_file, 'wb') as fh:
    #     pickle.dump(run, fh)
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    supported_classifiers = ['SVC', 'DecisionTreeClassifier',
                             'GradientBoostingClassifier',
                             'KNeighborsClassifier', 'NaiveBayes',
                             'MLPClassifier',
                             'LogisticRegression', 'RandomForestClassifier']
    parser.add_argument('--classifier', choices=supported_classifiers)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--task_id', required=True, type=int)
    parser.add_argument('--tmp_dir', required=True, type=str)
    parser.add_argument('--n_iter_inner_loop', default=200, type=int,
                        help='Number of iterations random search.')
    parser.add_argument('--n_folds_inner_cv', default=3, type=int,
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

    if classifier is None:
        for classifier in supported_classifiers:
            run = run_task(seed=seed,
                           tmp_dir=tmp_dir,
                           task_id=task_id,
                           estimator_name=classifier,
                           n_iter=n_iter_random_search,
                           n_jobs=n_jobs,
                           n_folds_inner_cv=n_folds_inner_cv,
                           scheduler_host=scheduler_host)
            print(run)
    else:
        run = run_task(seed=seed,
                       tmp_dir=tmp_dir,
                       task_id=task_id,
                       estimator_name=classifier,
                       n_iter=n_iter_random_search,
                       n_jobs=n_jobs,
                       n_folds_inner_cv=n_folds_inner_cv,
                       scheduler_host=scheduler_host)
        print(run)
