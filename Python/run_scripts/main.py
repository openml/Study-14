import argparse
import time

import yaml
import openml
import openmlstudy14.pipeline

# Necessary to register the backend with joblib!
import distributed.joblib
import sklearn.externals.joblib



def run_task(seed, task_id, estimator_name, n_iter, n_jobs, n_folds_inner_cv,
             scheduler_file):

    # retrieve dataset / task
    task = openml.tasks.get_task(task_id)
    num_features = task.get_X_and_y()[0].shape[1]
    indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])

    # retrieve classifier
    classifierfactory = openmlstudy14.pipeline.EstimatorFactory(n_folds_inner_cv, n_iter, n_jobs)
    estimator = classifierfactory.get_flow_mapping()[estimator_name](indices,
                                                                     num_features=num_features)

    print('Running task with ID %d.' % task_id)
    print('Arguments: random search iterations: %d, inner CV folds %d, '
          'n parallel jobs: %d, seed %d' %(n_iter, n_folds_inner_cv, n_jobs, seed))
    print('Model: %s' % str(estimator))
    flow = openml.flows.sklearn_to_flow(estimator)
    flow.tags.append('study_14')

    import time
    start_time = time.time()

    # TODO generate a flow first
    if scheduler_file is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='sklearn\.externals\.joblib\.parallel')
            run = openml.runs.run_flow_on_task(task, flow, seed=seed)
    else:
        print('Using dask parallel with scheduler file %s' % scheduler_file)

        scheduler_host = None
        for i in range(1000):
            try:
                with open(scheduler_file) as fh:
                    scheduler_information = yaml.load(fh)
                scheduler_host = scheduler_information['address']
            except FileNotFoundError:
                print('scheduler file %s not found. sleeping ... zzz' % scheduler_file)
                time.sleep(1)

        if scheduler_host is None:
            raise ValueError('Could not read scheduler_host file!')

        with sklearn.externals.joblib.parallel_backend('distributed',
                                                       scheduler_host=scheduler_host):
            run = openml.runs.run_flow_on_task(task, flow, seed=seed)

    end_time = time.time()
    run.tags.append('study_14')
    run_prime = run.publish()
    print('READTHIS', estimator_name, task_id, run_prime.run_id, end_time-start_time)

    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    supported_classifiers = ['svm', 'decision_tree', 'gradient_boosting',
                             'knn', 'naive_bayes', 'mlp', 'logreg',
                             'random_forest']
    parser.add_argument('--classifier', choices=supported_classifiers,
                        required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--task_id', required=True, type=int)
    parser.add_argument('--n_iter_inner_loop', default=200, type=int,
                        help='Number of iterations random search.')
    parser.add_argument('--n_folds_inner_cv', default=3, type=int,
                        help='Number of folds for inner CV')
    parser.add_argument('--n_jobs', default=-1, type=int,
                        help='Number of cores used by random search. Defaults '
                             'to all (-1).')
    parser.add_argument('--openml_server', default=openml.config.server)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--scheduler_file', default=None, type=str,
                        help='Use distributed backend if specified')

    args = parser.parse_args()

    openml_server = args.openml_server
    openml.config.server = openml_server
    cache_dir = args.cache_dir

    if cache_dir is not None:
        openml.config.set_cache_directory(cache_dir)

    seed = args.seed
    classifier = args.classifier
    n_iter_random_search = args.n_iter_inner_loop
    n_folds_inner_cv = args.n_folds_inner_cv
    n_jobs = args.n_jobs
    task_id = args.task_id
    scheduler_file = args.scheduler_file


    run = run_task(seed=seed,
                   task_id=task_id,
                   estimator_name=classifier,
                   n_iter=n_iter_random_search,
                   n_jobs=n_jobs,
                   n_folds_inner_cv=n_folds_inner_cv,
                   scheduler_file=scheduler_file)
    print(run)
