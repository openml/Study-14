import openml

all_tasks = openml.tasks.list_tasks(task_type_id=1, tag='study_14').keys()
openml.populate_cache()