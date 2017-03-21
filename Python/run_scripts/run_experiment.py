import random

from run_scripts.estimators import get_random_estimator
from openml import tasks,runs, datasets

# download all tasks
all_tasks = tasks.list_tasks(task_type_id=1, tag='study_14')
# pick one
task_id = random.choice(list(all_tasks.keys()))
task = tasks.get_task(task_id)
dataset = task.get_dataset()

# TODO: indexing should not be part of setup
indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
estimator = get_random_estimator(indices)

try:
    run = runs.run_task(task, estimator)
    run.tags.append('study_14')
    run.publish()
    print('Run uploaded with id %d' %run.run_id)
except Exception as e:
    print(e)
