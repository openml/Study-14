"""Register all flows prior to working with them to have no categorical
feature indicators for the OneHotEncoder on the server."""

import inspect

import openml
import openml2016.pipeline
import sklearn

openml.config.server = "http://test.openml.org/api/v1/xml"

_, svm = openml2016.pipeline.get_SVM()
_, dt = openml2016.pipeline.get_decision_tree()
_, gb = openml2016.pipeline.get_gradient_boosting()
_, knn = openml2016.pipeline.get_kNN()
_, nb = openml2016.pipeline.get_naive_bayes()
_, mlp = openml2016.pipeline.get_neural_network()
_, lr = openml2016.pipeline.get_logistic_regression()
_, rf = openml2016.pipeline.get_random_forest()

pipelines = [svm, dt, gb, knn, nb, mlp, lr, rf]

external_version = 'sklearn_' + sklearn.__version__
dependencies = '\n'.join(openml.runs.run._get_version_information())

# First, without RandomizedSearchCV
for pipeline in pipelines:
    classifier = pipeline.get_params()['classifier']
    for param, value in pipeline.get_params().items():

        # All components in the pipeline
        if (hasattr(value, 'fit') and hasattr(value, 'predict') and
            hasattr(value, 'get_params') and hasattr(value, 'set_params')) or \
           (hasattr(value, 'fit') and hasattr(value, 'transform') and
            hasattr(value, 'get_params') and hasattr(value, 'set_params')):
            first_line = inspect.getdoc(value).split('\n')[0]

            flow = openml.OpenMLFlow(model=value,
                                     external_version=external_version,
                                     dependencies=dependencies,
                                     description=first_line)
            flow.init_parameters_and_components()
            flow._ensure_flow_exists()

    # The full pipeline
    description = 'Simple pipeline to benchmark %s.' % \
                  classifier.__class__.__name__
    flow = openml.OpenMLFlow(model=pipeline,
                             external_version=external_version,
                             dependencies=dependencies,
                             description=description)
    flow.init_parameters_and_components()
    print(flow.description, flow.name)
    flow._ensure_flow_exists()
