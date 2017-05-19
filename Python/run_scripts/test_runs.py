import openml
import openmlstudy14.pipeline

openml.config.server = "https://test.openml.org/api/v1/"

# download all tasks
all_tasks = openml.tasks.list_tasks(task_type_id=1, tag='study_14')
# pick one
task_id = 115 # random.choice(list(all_tasks.keys()))
task = openml.tasks.get_task(task_id)
print("Task %d, dataset %d" %(task_id, task.dataset_id))
dataset = task.get_dataset()

# TODO: indexing should not be part of setup
indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
factory = openmlstudy14.pipeline.EstimatorFactory(3, 20, 1)

estimators = factory.get_all_flows(indices)

for estimator in estimators:
    flow = openml.flows.sklearn_to_flow(estimator)
    flow.tags.append('study_14')
    run = openml.runs.run_flow_on_task(task, flow)
    run.tags.append('study_14')
    run.publish()
    print('Flow %s Task %d uploaded with id %d' %(flow.name, task_id, run.run_id))
