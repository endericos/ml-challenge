# ML Challenge

## Requirements

1. Provide an appropriate evaluation of the model performance on the test data.
2. Implement a way to persist the trained model to local disk.
3. Implement an API according to the open api specification.
4. Create a web service (in Python 3) to serve the persisted model.
5. Deploy the model locally.
6. Create a container with your solution that can be run on Kubernetes.
7. Provide some sample curl commands or a Postman collection.
8. *Stretch Goal 1* - Suggest and/or implement improvements to the model.
9. *Stretch Goal 2* - Testing of the API before deployment.
10. *Stretch Goal 3* - Metrics API for inspecting current metrics of the service.

**Note**: We expect a web service that can run indefinitely, linearly scale and process as many requests as possible until stopped. The service should be able to accept as many requests as the model can process.

## Write-up

This write-up attempts to explain the implementation, and the taught process behind in this repo. Given the personal time constraints, there were some missing points, referenced at the end of the write-up.

### Model Evaluation

Model evaluation and persistance done under `research/` directory. Here's the directory structure for the important files:

```dir
├── research/          
│   ├── evaluation.json       <- the results of model performance on test data
│   ├── model.py              <- main script for model training, prediction, evaluation and persistance
│   ├── utils.py              <- the code that defines sklearn pipeline, evaluates, converts and persists the trained model 
│   ├── text_classifier.onnx  <- persisted onnx model

```

Since the model is a classification model, micro-, macro- and weighted-average of `precision`, `recall`, and `f1-score` metrics calculated on test data and stored in `evaluation.json`.

### Model Persistance

`Model` class omitted and rewritten as a composite [sklearn.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) object. Usually, it's a good practice chaining sklearn estimators and transformers for easier maintainability. This object converted into a ONNX format and persisted to disk as such using [skl2onnx](https://github.com/onnx/sklearn-onnx).

Joblib is not chosen, since serialising the model as python object makes it strongly coupled to the Python training environment. In this case, an inference service, would still require the training framework (i.e. sklearn, pytorch).

Instead, we can only maintain [onnxruntime](https://github.com/microsoft/onnxruntime) package in our inference environment. The only packages needed for the service `fastapi[all], onnxruntime`.

My reasoning was ONNX models are **portable to many different languages (i.e. C++, go)** and **different computing architectures (i.e. mobile, edge, cloud)**. They can provide greater flexibility in different customer environments.

ONNX models are also **optimized for performance**, they can run faster and consume fewer resources than some of the other formats. Trained model persisted as **joblib occupied 5.5 mb**, but **ONNX only 2.0 mb**. Also in an auto-scaling system, **ONNX may reduce cold start latency**.

Trade-off is sometimes it might take a day or two to write a converter for custom model objects, but the effort would be worth it, considering the eliminated problems.

### Inference Service

The service is implemented based on provided OpenAPI specification in FastAPI, which provides built-in support for **asynchronous request handling** ensures that the server can handle multiple requests concurrently without blocking other calls.

**Note**: FastAPI version currently supports OAS 3.1, to fully reflect OAS 2.0 shared, we'd need to identify an older FastAPI version but the latest before switching to OAS 3.1 from FastAPI releases. Neglected doing so, due to minor differences.

Here's the directory structure for the service code:

```dir
├── src/          
│   ├── classifier.py         <- functions to load the model and make predictions
│   ├── main.py               <- the inference service code
│   ├── schemas.py            <- Pydantic data models and persists the trained model 
│   ├── text_classifier.onnx  <- persisted onnx model
│   ├── example_request.json  <- an example request file for curl commands

```

The `prediction/` endpoint is defined as an asynchronous function `async def prediction(...)` to **handle requests concurrently**. The underlying `classifier.predict_labels` function uses using `asyncio` to efficiently process requests while waiting for I/O operations (such as database queries or external API calls in other cases) to complete, maximizing throughput.

The code can be deployed **behind a load balancer, allowing us to scale horizontally** by adding more instances of the service to handle increased load. The load balancer distributes incoming requests across multiple instances, enabling linear scalability.

Currently, [ONNX Runtime Architecture](https://onnxruntime.ai/docs/reference/high-level-design.html) only supports synchronous execution. However, we can still benefit from FastAPI's asynchronous capabilities by running the prediction logic in a separate thread pool using the `run_in_executor` method. This approach allows us to offload CPU-bound tasks to a thread pool while keeping the event loop free to handle other requests.

ONNX model is only 2.0 mb. If the model object would be very large, then reloading the model in every restart would cause some trouble, such as in automated simple tests. To address this we'd ideally need a "startup-shutdown" logic using [Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) whereas we load the model before the requests are handled, but only right before the application starts receiving requests, not while the code is being loaded.

The advantage of doing this as an event is that **our service won't start until the model is loaded and so no requests will be prematurely processed and cause errors**.

One tradeoff is lifespan events currently only supported by `FastAPI` but not `APIRouter`.

### Deployment

`Makefile` targets hold the required commands for deployment and to examine the service.

To run the service locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make run
make example-request
```

to view the auto-generated API documentation, and to make requests directly from the browser:

```bash
make open-docs
```

to build and run the container:

```bash
make build
make run-container
make example-request
```

`example-request` target just holds an example curl commmand

```bash
curl -X POST http://localhost:8080/prediction -H "Content-Type: application/json" -d "@./src/example_request.json"`
```

Typically, we'd run `uvicorn --reload` during local development and `gunicorn -k uvicorn.workers.UvicornWorker` during production while we're dealing with meaningful traffic.

However, the container will be ran in Kubernetes (K8s). **Since, K8s handle replication of containers** while still supporting load balancing for the incoming requests, all at the cluster level, I preferred building the Docker image by running a single Uvicorn process. So, instead of using a process manager like Gunicorn with workers, we'd run **a single Uvicorn process per container, and handle the workload by the means provided by K8s** (i.e. `HorizontalPodAutoscaler`).

And since the container will be run in K8s behind a TLS Termination Proxy (load balancer) like Nginx, `--proxy-headers` option added to `Dockerfile` to tell Uvicorn **to trust the headers sent by that proxy** telling it that the application is running behind HTTPS, etc.

### Tests

Added basic performance test report and unit tests. They are basic given the time constraint.

To install testing requirements: `pip install tests/testing_requirements.txt`

To run a performance test: `locust -f tests/performance_tests/performance_test.py`

Pushed the service running on a single uvicorn worker to above 90% util by ramping to 10000 users at a rate of 10000 per second by another single locust worker. A worker runs on Ubuntu 20.04 on [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) on a single thread of AMD Ryzen 5950X. Performance report is in `tests/performance_tests/locust_report_*.html`.

Response time measurements can be considered a bit inconsistent as the CPU throttled to 90% util, and throughtput until achieving 90% CPU util. Stress test was on total 800K+ requests, 1000 RPS, and **the median response time was 2600 ms under stress**.

To run unit tests;

```bash
coverage run --omit="tests/*" -m pytest tests --disable-pytest-warnings
coverage report -m
```

### Missing components

Some of the things I would do if I had more time:

- **Model improvement**: A PyTorch NN classifier with text embeddings provided by typically one of embedding models in `SentenceTransformers`. Also, adding a text cleaning functionality to increase quality of training & predictions.
- **Refactoring**: Proper OOP implementation of the service code. Besides, implementing endpoints as the routers (`APIRouter`).
- **Docstrings** Type hints used, but still proper code documentation needed, including mini examples.
- **Proper exception handling**: For the demo, I catched all exceptions with a wide try/except Exception in the endpoint. But the best practice is to explicitly catch the exceptions and deal with them individually, not only in the endpoint, but in all underlying objects.
- **Packaging the code** as a Python package using provided template.
- **CI/CD**: GitHub Action templates to be triggered by Pull Requests, running .yaml workflows for checking code quality and unit tests.
- **Kubernetes testing**: a basic gateway and a list of manifests (i.e. `configmap.yaml`, `deployment.yaml`, `service.yaml` `ingress.yaml`) on my local minikube / or development cluster.
- **Extensive Unit & Performance tests**: The most important test in the suite needs a quick fixing, but left due to time-constraint.  
- **Code Profiling**: using `pyinstrument` and `FastAPI.middleware` decorator to register a profiling middleware to our service. A good write-up on the topic [here](https://blog.balthazar-rouberol.com/how-to-profile-a-fastapi-asynchronous-request).
- **Metrics API**: assuming only for applications API metrics (i.e. Request Per Minute, Average and Max Latency, Errors per Minute), but some of these can also be considered 
  - Infrastructure API Metrics: (i.e. uptime, CPU util, memory util)
  - API Product metrics: (i.e. usage growth, unique API consumers, top customers by API usage, API retention, TTFHW, API calls per business transaction) can also be considered.
  - A good starting point: [PyTorch Metrics API](https://pytorch.org/serve/metrics_api.html).
