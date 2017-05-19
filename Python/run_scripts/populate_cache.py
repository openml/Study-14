import openml

tasks = list(openml.tasks.list_tasks(tag='study_14').keys())
openml.populate_cache(task_ids=tasks)
print('Using %d tasks', len(tasks))
