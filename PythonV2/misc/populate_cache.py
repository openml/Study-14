import openml

#openml.config.server = 'https://test.openml.org/api/v1/'
#openml.config.set_cache_directory('/home/feurerm/tmp/openml_test/')

study = openml.study.get_study(99)
for task in study.tasks:
    openml.tasks.get_task(task)