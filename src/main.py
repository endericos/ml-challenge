from fastapi import FastAPI, Body
import os
from contextlib import asynccontextmanager

from src.classifier import predict_label_async, load_model
from src.schemas import ApiRequest, ApiResponse

# Ensure model path is found regardless execution directory
model_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "text_classifier.onnx"
)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the model before the requests are handled,
    but only right before the application starts receiving requests,
    not while the code is being loaded (not in every request).
    """
    ml_models["text_classifier"] = load_model(model_path)
    yield
    ml_models.clear()


# Prediction API
app = FastAPI(
    title="Prediction API",
    description="This is a sample prediciton API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post(
    "/prediction",
    response_model=ApiResponse,
    response_description="Successful prediction",
    responses={405: {"description": "Validation exception"}},
    tags=["prediction"],
)
async def prediction(
    body: ApiRequest = Body(..., description="The document to perform a prediction on")
) -> ApiResponse:
    """
    Prediction endpoint to makes predictions on loaded model upon
    prediction requests received.
    """
    try:
        label = await predict_label_async(
            model=ml_models["text_classifier"],
            text=body.text,
        )
        return ApiResponse(label=label)
    except Exception as err:
        raise err
