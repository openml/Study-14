import openml

#openml.config.server = 'https://test.openml.org/api/v1/'
#openml.config.set_cache_directory('/home/feurerm/tmp/openml_test/')

all_tasks = openml.tasks.list_tasks(task_type_id=1, tag='study_14').keys()
openml.populate_cache(task_ids=all_tasks)