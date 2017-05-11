"""Script to test the runtime and memory requirement of a set of classifiers.

This script sequentially runs the cross-product of all tasks of study 14 and
the classifiers used in study 14 (decision tree, gradient boosting, kNN,
logistic regression, multilayer perceptron, naive bayes, random forest and
support vector machines). For each combination of task and learner it spawns
a subprocess and measures time and memory usage via `time -v`. The script
outputs a csv representation of all runs so far after each call to a machine
learning algorithm.
"""

from collections import defaultdict
import os
import subprocess

import openml
import pandas


model_names = ['decision_tree', 'gradient_boosting', 'knn', 'logreg', 'mlp',
               'naive_bayes', 'random_forest', 'svm']

all_tasks = openml.tasks.list_tasks(task_type_id=1, tag='study_14')

memory_per_task = defaultdict(dict)
runtime_per_task = defaultdict(dict)

this_directory = os.path.abspath(os.path.dirname(__file__))
target_file = os.path.join(this_directory, "test_time_and_memory_requirements_target.py")
cmd_call = 'time -v python3 %s' % target_file

n_runs = 0
for task_id in all_tasks:
    n_runs += 1
    #if n_runs > 2:
    #    break
    for model_name in model_names:
        cmd = '%s %s %d' % (cmd_call, model_name, task_id)
        #print(cmd)
        pipe = subprocess.PIPE
        rval = subprocess.run(cmd, shell=True, stdout=pipe, stderr=pipe)
        for line in rval.stderr.splitlines():
            line = str(line)
            if 'Maximum resident set size' in line:
                maximum_ram_used = int(line.replace("'", "").split(' ')[-1]) / 1024.
                memory_per_task[task_id][model_name] = maximum_ram_used
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
                runtime_per_task[task_id][model_name] = wallclock_time

        print('################################################################')
        print('################################################################')
        print('################################################################')
        print('Task %d/%d' % (n_runs, len(all_tasks)))
        print('Memory requirement')
        print(pandas.DataFrame(memory_per_task).transpose().to_csv())
        print('runtimes')
        print(pandas.DataFrame(runtime_per_task).transpose().to_csv())
