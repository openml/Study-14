import random

from openmlstudy14.pipeline import get_random_estimator

from openml import tasks,runs, flows

# download all tasks
all_tasks = tasks.list_tasks(task_type_id=1, tag='study_14')
# pick one
task_id = 115 # random.choice(list(all_tasks.keys()))
task = tasks.get_task(task_id)
print("Task %d, dataset %d" %(task_id, task.dataset_id))
dataset = task.get_dataset()

# TODO: indexing should not be part of setup
indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
estimator = get_random_estimator(indices)

flow = flows.sklearn_to_flow(estimator)

run = runs.run_flow_on_task(task, flow, flow_tags=['study_14'])
run.tags.append('study_14')
run.publish()
print('Run uploaded with id %d' %run.run_id)
