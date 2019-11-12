"""Script to test the runtime and memory requirement of classifier.
"""

import argparse
from collections import defaultdict
import json
import os
import subprocess

this_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(this_dir, '..'))

import sys
sys.path.append(parent_dir)

import openml
import openml.extensions.sklearn
import openmlstudy99.pipeline


parser = argparse.ArgumentParser()
parser.add_argument('--output-directory', type=str, required=True)
parser.add_argument('--task-id', type=int, required=True)
parser.add_argument(
    '--classifier',
    choices=openmlstudy99.pipeline.EstimatorFactory(
        n_folds_inner_cv=None,
        n_iter=None,
    ).estimator_mapping,
    required=True,
)
args = parser.parse_args()

output_directory = args.output_directory
task_id = args.task_id
model_name = args.classifier

try:
    os.makedirs(output_directory)
except Exception as e:
    pass

all_tasks = openml.study.get_suite(99).tasks
assert task_id in all_tasks
output_file = os.path.join(output_directory, '%d_%s.json' % (task_id, model_name))

maximum_ram_used = None
wallclock_time = None

this_directory = os.path.abspath(os.path.dirname(__file__))
target_file = os.path.join(this_directory, "test_time_and_memory_requirements_target.py")
cmd_call = 'time -v python3 %s' % target_file

task = openml.tasks.get_task(task_id)
indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])

# As of scikit-learn==0.18, gradient boosting and gaussian naive bayes
# do not work on sparse data (which is produced by the OneHotEncoder).
if len(indices) > 0 and model_name in ['gradient_boosting', 'naive_bayes']:
    with open(output_file, 'w') as fh:
        json.dump({'possible': False}, fh)

cmd = '%s %s %d' % (cmd_call, model_name, task_id)

pipe = subprocess.PIPE
rval = subprocess.run(cmd, shell=True, stdout=pipe, stderr=pipe)
if rval.returncode > 0:
    with open(output_file, 'w') as fh:
        json.dump({'possible': True, 'success': False}, fh)
    print(rval.stdout)
    print(rval.stderr)
    exit(1)

for line in rval.stderr.splitlines():
    line = str(line)
    if 'Maximum resident set size' in line:
        maximum_ram_used = int(line.replace("'", "").split(' ')[-1]) / 1024.
    if 'Elapsed (wall clock) time' in line:
        wallclock_time = line.replace("'", "").split(' ')[-1]
        wallclock_time = wallclock_time.split(":")
        wallclock_time = [float(t) for t in wallclock_time]
        if len(wallclock_time) == 2:
            wallclock_time = 60 * wallclock_time[0] + wallclock_time[1]
        else:
            wallclock_time = 3600 * wallclock_time[0] + \
                             60 * wallclock_time[1] + \
                             wallclock_time[2]

print('################################################################')
print('################################################################')
print('################################################################')
print('Task %d: %s' % (task_id, model_name))
print('Memory usage: %f; time taken: %f' % (maximum_ram_used, wallclock_time))

with open(output_file, 'w') as fh:
    json.dump({'possible': True, 'success': True, 'memory': maximum_ram_used, 'time': wallclock_time}, fh)
