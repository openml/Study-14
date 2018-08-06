import argparse
import os
import sys

this_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(this_dir, '..'))
sys.path.append(parent_dir)
from openmlstudy99.pipeline import supported_classifiers

import openml

################################################################################
# Command line parsing
parser = argparse.ArgumentParser()
parser.add_argument('--output-file', type=str, required=True)
parser.add_argument('--openml-server', type=str, default=None)
parser.add_argument('--run-tmp-dir', type=str, default=None)
parser.add_argument('--n-jobs', type=int, default=-1)
args = parser.parse_args()
output_file = args.output_file
openml_server = args.openml_server
if openml_server is not None:
    openml.config.server = openml_server
run_tmp_dir = args.run_tmp_dir
n_jobs = args.n_jobs

################################################################################
# Constants

num_random_search = 200
num_inner_folds = 3
num_outer_folds = 10

################################################################################
# dask distributed commands

main_command = ' '.join([
    'python ',
    os.path.join(parent_dir, 'main.py'),
    '--seed 1',
    '--task-id %d',
    '--classifier %s',
    '--n-jobs %d' % n_jobs,
])
if openml_server is not None:
    main_command += (' --openml-server %s' % openml_server)
if run_tmp_dir is not None:
    main_command += (' --run-tmp-dir %s' % run_tmp_dir)

################################################################################

tasks = []
# Generate all command files
for task_id in list(openml.tasks.list_tasks(tag='study_99').keys())[:10]:
    for estimator_name in supported_classifiers:

        if (
            openml.tasks.get_task(task_id).get_dataset().format.lower() == 'sparse_arff'
            and estimator_name == 'naive_bayes'
        ):
            continue
        else:
            print(task_id, estimator_name)
            tasks.append((estimator_name, task_id))

with open(output_file, 'w') as fh:
    for estimator_name, task_id in sorted(tasks):
        fh.write(main_command % (task_id, estimator_name))
        fh.write('\n')

