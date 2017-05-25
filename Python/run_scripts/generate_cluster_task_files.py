import argparse
from collections import defaultdict
import csv
import os

from openmlstudy14.pipeline import EstimatorFactory
import openml

################################################################################
# Command line parsing
parser = argparse.ArgumentParser()
parser.add_argument('--jobfile-directory', type=str, required=True)
parser.add_argument('--n-workers', type=int, required=True)
parser.add_argument('--openml-server', type=str, default=None)
parser.add_argument('--cache-dir', type=str, default=None)
args = parser.parse_args()
jobfile_directory = args.jobfile_directory
try:
    os.makedirs(jobfile_directory)
except:
    pass
n_workers = args.n_workers
openml_server = args.openml_server
if openml_server is not None:
    openml.config.server = openml_server
cache_dir = args.cache_dir
if cache_dir is not None:
    openml.config.set_cache_directory(cache_dir)

################################################################################
# Constants
memory_allowances = [6000, 12000, 18000, 24000]
time_allowances_in_hours = [1, 2, 6, 12, 24, 48, 72, 96]
time_allowances = [t * 3600 for t in time_allowances_in_hours]
max_parallel_jobs = 50
num_random_search = 200
num_inner_folds = 3
num_outer_folds = 10

################################################################################
# Read expected runtime and memory usage from previous probe
tasks = list(openml.tasks.list_tasks(tag='study_14').keys())
estimator_factory = EstimatorFactory()
estimators = estimator_factory.get_flow_mapping()

def read_csv_file(filename):
    memory_probes = defaultdict(dict)
    with open(filename) as fh:
        reader = csv.DictReader(fh)
        for line in reader:
            key = int(line[''])
            del line['']
            line = {k: float(v) for k, v in line.items()}
            memory_probes[key] = line

    return memory_probes

this_file_path = os.path.abspath(os.path.dirname(__file__))
memory_probes_file = os.path.abspath(os.path.join(this_file_path, '..', 'misc',
                                                  'memory_probes.csv'))
memory_probes = read_csv_file(memory_probes_file)
runtime_probes_file = os.path.abspath(os.path.join(this_file_path, '..', 'misc',
                                                   'runtime_probes.csv'))
runtime_probes = read_csv_file(runtime_probes_file)

################################################################################
# dask distributed commands

scheduler_command = 'dask-scheduler --scheduler-file %s'
main_command = 'python ' + os.path.join(this_file_path, 'main.py') + \
               ' --scheduler_file %s --seed 1 --task_id %d --classifier %s'
if openml_server is not None:
    main_command += (' --openml_server %s' % openml_server)
if cache_dir is not None:
    main_command += (' --cache_dir %s' % cache_dir)
worker_command = 'dask-worker --nprocs 1 --nthreads 1 --death-timeout %d ' \
                 '--no-nanny --scheduler-file %s'
################################################################################


tasks_by_size = defaultdict(list)

# Generate all command files
for task_id in tasks:
    for estimator_name in estimators:
        try:
            estimated_memory_usage = memory_probes[task_id][estimator_name]
        except Exception as e:
            estimated_memory_usage = 3072
        required_memory = [ma for ma in memory_allowances
                           if ma >= (estimated_memory_usage * 1.5)][0]
        try:
            estimated_max_runtime = runtime_probes[task_id][estimator_name]
        except Exception as e:
            estimated_max_runtime = 1
        estimated_runtime = num_outer_folds * \
                            (num_random_search * num_inner_folds /
                             max_parallel_jobs + 1) * estimated_max_runtime
        required_cluster_time = [ta for ta in time_allowances
                                 if ta >= (estimated_runtime * 2.0)]
        required_cluster_time = time_allowances[-1] if len(
            required_cluster_time) == 0 else required_cluster_time[0]
        tasks_by_size[(required_cluster_time, required_memory)].append((task_id, estimator_name))


for tbs in sorted(tasks_by_size):
    required_cluster_time, required_memory = tbs

    dask_scheduler_file = os.path.expanduser(os.path.join(
        '~', '.dask_scheduler_%d_%d' % (required_cluster_time, required_memory)))

    jobfile = os.path.join(jobfile_directory, '%d_%d_scheduler_command.txt' %
                           (required_cluster_time, required_memory))
    with open(jobfile, 'w') as fh:
        fh.write(scheduler_command % dask_scheduler_file)
        fh.write('\n')

    jobfile = os.path.join(jobfile_directory, '%d_%d.txt' % (tbs[0], tbs[1]))
    with open(jobfile, 'w') as fh:
        for job in tasks_by_size[tbs]:
            fh.write(main_command % (dask_scheduler_file, job[0], job[1]))
            fh.write('\n')

    # lots of workers
    jobfile = os.path.join(jobfile_directory, '%d_%d_worker_commands.txt' % (
        required_cluster_time, required_memory))
    with open(jobfile, 'w') as fh:
        for i in range(n_workers):
            fh.write(worker_command % (3600, dask_scheduler_file))
            fh.write('\n')