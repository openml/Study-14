import argparse
import os

import arff
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend
import numpy as np
from sklearn.externals.joblib import register_parallel_backend, \
    parallel_backend
from sklearn.externals.joblib.parallel import BatchedCalls
import yaml

import openml
import openmlstudy14.pipeline
import openmlstudy14.util


class NPCachingIpyParallelBackend(IPythonParallelBackend):
    """Joblib backend which distributes numpy arrays via a shared file system."""

    def __init__(self, view, tmp_dir):
        super().__init__(view=view)
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)

    def apply_async(self, func, callback=None):
        assert isinstance(func, BatchedCalls)
        calls = []
        for batched_call in func.items:
            f = openmlstudy14.util.CallbackFunction(batched_call[0])
            args = list(batched_call[1])
            kwargs = batched_call[2]

            for i in range(len(args)):
                # Save numpy arrays to disk and replace them by a dummy!
                if isinstance(args[i], np.ndarray):
                    args[i] = openmlstudy14.util.NumpyMock(self.tmp_dir,
                                                           args[i])

            for key, item in kwargs.items():
                # This could easily be changed to also store numpy arrays on
                # disk but is so far not necessary
                if isinstance(item, np.ndarray):
                    raise NotImplementedError(type(item))

            calls.append((f, tuple(args), kwargs))
        calls = BatchedCalls(calls)
        return super().apply_async(func=calls, callback=callback)


def run_task(seed, task_id, estimator_name, n_iter, n_jobs, n_folds_inner_cv,
             profile, joblib_tmp_dir, run_tmp_dir):

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
    if profile is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='sklearn\.externals\.joblib\.parallel')
            run = openml.runs.run_flow_on_task(task, flow, seed=seed)
    else:
        print('Using ipython parallel with scheduler file %s' % profile)

        for i in range(1000):
            profile_file = os.path.join(os.path.expanduser('~'),
                                        '.ipython',
                                        'profile_%s' % profile, 'security',
                                        'ipcontroller-engine.json')
            try:
                with open(profile_file) as fh:
                    scheduler_information = yaml.load(fh)
                break
            except FileNotFoundError:
                print('scheduler file %s not found. sleeping ... zzz' % profile_file)
                time.sleep(1)
                continue

        c = Client(profile=profile)
        bview = c.load_balanced_view()
        register_parallel_backend('ipyparallel',
                                  lambda: NPCachingIpyParallelBackend(
                                      view=bview, tmp_dir=joblib_tmp_dir))

        with parallel_backend('ipyparallel'):
            run = openml.runs.run_flow_on_task(task, flow, seed=seed)

    end_time = time.time()
    run.tags.append('study_14')

    tmp_dir = os.path.join(run_tmp_dir, '%s_%s' % (str(task_id), estimator_name))
    print(tmp_dir)
    try:
        os.makedirs(tmp_dir)
    except Exception as e:
        print(e)
    run_xml = run._create_description_xml()
    predictions_arff = arff.dumps(run._generate_arff_dict())

    with open(tmp_dir + '/run.xml', 'w') as f:
        f.write(run_xml)
    with open(tmp_dir + '/predictions.arff', 'w') as f:
        f.write(predictions_arff)

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
    parser.add_argument('--joblib-tmp-dir', default='/tmp/.ipyparallel',
                        type=str,
                        help='Temporary directory to store numpy arrays to. '
                             'Must be accesible by all workers!')
    parser.add_argument('--openml_server', default=openml.config.server)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--profile', default=None, type=str,
                        help='Use ipython parallel backend if specified')
    parser.add_argument('--run_tmp_dir', type=str, default='/tmp/',
                        help='Store runs temporarilty.')

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
    profile = args.profile
    joblib_tmp_dir = args.joblib_tmp_dir
    run_tmp_dir = args.run_tmp_dir


    run = run_task(seed=seed,
                   task_id=task_id,
                   estimator_name=classifier,
                   n_iter=n_iter_random_search,
                   n_jobs=n_jobs,
                   n_folds_inner_cv=n_folds_inner_cv,
                   profile=profile,
                   joblib_tmp_dir=joblib_tmp_dir,
		   run_tmp_dir=run_tmp_dir,
    )
    print(run)
