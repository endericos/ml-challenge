from locust import HttpUser, task, between
import json
import os

example_request_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "example_request.json"
)


class PerformanceTests(HttpUser):
    # Set wait time between requests to 0 for max-throughput: wait_time = locust.constant(0)
    wait_time = between(1, 3)

    @task(1)
    def test_tf_predict(self):

        with open(example_request_path, "r") as jf:
            data = json.load(jf)

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        response = self.client.post(
            url="prediction",
            data=json.dumps({"text": data["text"]}),
            headers=headers,
        )
