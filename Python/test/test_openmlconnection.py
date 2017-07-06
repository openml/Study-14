
from openmlstudy14.pipeline import EstimatorFactory

import unittest
import openml
import hashlib
import time

class OpenMLTaskTest(unittest.TestCase):

    def _add_sentinel_to_flow_name(self, flow, sentinel=None):
        if sentinel is None:
            # Create a unique prefix for the flow. Necessary because the flow is
            # identified by its name and external version online. Having a unique
            #  name allows us to publish the same flow in each test run
            md5 = hashlib.md5()
            md5.update(str(time.time()).encode('utf-8'))
            sentinel = md5.hexdigest()[:10]
            sentinel = 'TEST%s' % sentinel

        flows_to_visit = list()
        flows_to_visit.append(flow)
        while len(flows_to_visit) > 0:
            current_flow = flows_to_visit.pop()
            current_flow.name = '%s%s' % (sentinel, current_flow.name)
            for subflow in current_flow.components.values():
                flows_to_visit.append(subflow)

        return flow, sentinel

    def test_existing_flow_exists(self):
        openml.config.server = "https://test.openml.org/api/v1/"
        # create a flow
        classifier = EstimatorFactory().get_decision_tree(None)

        flow = openml.flows.sklearn_to_flow(classifier)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)
        # publish the flow
        flow = flow.publish()
        # redownload the flow
        flow = openml.flows.get_flow(flow.flow_id)

        # check if flow exists can find it
        flow = openml.flows.get_flow(flow.flow_id)
        downloaded_flow_id = openml.flows.flow_exists(flow.name, flow.external_version)
        self.assertEquals(downloaded_flow_id, flow.flow_id)
