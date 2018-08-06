import sys
sys.path.append('..')
import main
import openmlstudy99.pipeline

import openml

server = "https://test.openml.org/api/v1/xml"
openml.config.server = server
cache_dir = "/tmp/openml_study99"


for classifier in openmlstudy99.pipeline.supported_classifiers:
    for task_id in [577, 307, 391, 565, 53]:
        run = main.run_task(
            seed=1,
            task_id=task_id,
            estimator_name=classifier,
            n_iter=10,
            n_jobs=-1,
            n_folds_inner_cv=3,
            run_tmp_dir=cache_dir,
        )
        run_prime = openml.runs.get_run(run.run_id)
        model = openml.runs.initialize_model_from_run(run.run_id)

        # Test that the parameters are exactly equal!
        run_parameter_settings = {}
        run_prime_parameter_settings = {}
        for lst in run.parameter_settings:
            run_parameter_settings[lst['oml:component']] = lst
        run_parameter_settings = {
            key: run_parameter_settings[key]
            for key in sorted(run_parameter_settings.keys())
        }
        for lst in run_prime.parameter_settings:
            run_prime_parameter_settings[lst['oml:component']] = lst
        run_prime_parameter_settings = {
            key: run_prime_parameter_settings[key]
            for key in sorted(run_prime_parameter_settings.keys())
        }
        assert sorted(run_parameter_settings) == sorted(run_prime_parameter_settings)

        if classifier != 'knn':
            # TODO fix the sorting issue with parameter distributions?
            params = run.flow.model.get_params()
            del params['param_distributions']
            params_prime = model.get_params()
            del params_prime['param_distributions']
            # Keep in mind that scikit-learn does not support direct comparison of models as we
            # intend to do here
            assert str(params) == str(params_prime)
        else:
            params = run.flow.model.get_params()
            del params['param_grid']
            params_prime = model.get_params()
            del params_prime['param_grid']
            # Keep in mind that scikit-learn does not support direct comparison of models as we
            # intend to do here
            assert str(params) == str(params_prime)