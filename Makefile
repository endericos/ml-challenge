NUM_WORKERS ?= 1

# deploy locally
.PHONY: run
run:
	uvicorn src.main:app --host localhost --port 8080 --workers $(NUM_WORKERS) --reload

# build docker image
.PHONY: build
build:
	docker build -t text-classifier .

# run docker container
.PHONY: run-container
run-container:
	docker run -d -p 8080:8080 --name prediction-api text-classifier

# make an example curl request
.PHONY: example-request
example-request:
	curl -X POST http://localhost:8080/prediction -H "Content-Type: application/json" -d "@./src/example_request.json"

# open API documentation
.PHONY: open-docs
open-docs:
	python -m webbrowser http://localhost:8080/docs