import argparse
import os
from typing import cast

this_dir = os.path.abspath(os.path.dirname(__file__))
import sys
#sys.path.append('/home/feurerm/projects/openml/Study-14/PythonV2/')
sys.path.append(this_dir)

import numpy as np
import openml
import openml.extensions.sklearn
import openmlstudy99.pipeline


def run_task(
    seed,
    task_id,
    estimator_name,
    n_iter,
    n_jobs,
    n_folds_inner_cv,
    run_tmp_dir,
):

    # retrieve dataset / task
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
    del X
    del dataset

    # retrieve classifier
    indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
    classifierfactory = openmlstudy99.pipeline.EstimatorFactory(n_folds_inner_cv, n_iter, n_jobs)
    estimator = classifierfactory.get_flow_mapping()[estimator_name](
        indices,
        num_features=num_features,
    )

    print('Running task with ID %d.' % task_id)
    print('Arguments: random search iterations: %d, inner CV folds %d, '
          'n parallel jobs: %d, seed %d' %(n_iter, n_folds_inner_cv, n_jobs, seed))
    print('Model: %s' % str(estimator))
    sklearn_extension = openml.extensions.sklearn.SklearnExtension()
    flow = sklearn_extension.model_to_flow(estimator)
    print(flow.name)
    assert flow.name is not None

    import time
    start_time = time.time()

    run = openml.runs.run_flow_on_task(flow, task, seed=seed)

    end_time = time.time()

    tmp_dir = os.path.join(run_tmp_dir, '%s_%s' % (str(task_id), estimator_name))
    print(tmp_dir)
    try:
        os.makedirs(tmp_dir)
    except Exception as e:
        print(e)

    # TODO disable store model functionality!
    run.to_filesystem(tmp_dir)

    run_prime = run.publish()
    print('READTHIS', estimator_name, task_id, run_prime.run_id, end_time-start_time)

    return run


if __name__ == '__main__':
    # Example call:
    # --classifier xgboost --seed 1 --task_id 119 --openml-server https://test.openml.org/api/v1 --run-tmp-dir /tmp/openml-study99/

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--classifier',
        choices=openmlstudy99.pipeline.supported_classifiers,
        required=True,
    )
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--task-id', required=True, type=int)
    parser.add_argument(
        '--n-iter-inner-loop',
        default=200,
        type=int,
        help='Number of iterations random search.',
    )
    parser.add_argument(
        '--n-folds-inner-cv',
        default=3,
        type=int,
        help='Number of folds for inner CV',
    )
    parser.add_argument(
        '--n-jobs',
        default=-1,
        type=int,
        help='Number of cores used by random search. Defaults to all (-1).',
    )
    parser.add_argument('--openml-server', default=openml.config.server)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument(
        '--run-tmp-dir',
        type=str,
        default='/tmp/',
        help='Store runs temporarilty.',
    )

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
    run_tmp_dir = args.run_tmp_dir

    run = run_task(
        seed=seed,
        task_id=task_id,
        estimator_name=classifier,
        n_iter=n_iter_random_search,
        n_jobs=n_jobs,
        n_folds_inner_cv=n_folds_inner_cv,
        run_tmp_dir=run_tmp_dir,
    )

    print(run)
