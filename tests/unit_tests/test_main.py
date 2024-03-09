import sys

sys.path.append("src/")

import asyncio
import pytest
import httpx
import json
from src.main import app, ml_models
from unittest.mock import MagicMock, patch
import pytest_asyncio


@pytest_asyncio.fixture(scope="session")
async def async_client():
    async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def mock_ml_model():
    mock_model = MagicMock()
    return mock_model


@pytest.fixture
def mock_load_model():
    with patch("main.load_model") as mock_load_model:
        mock_load_model.return_value = MagicMock()
        yield


@pytest.fixture
def mock_ml_models(mock_ml_model):
    ml_models.clear()
    ml_models["text_classifier"] = mock_ml_model
    yield ml_models


# TO-DO: a tiny mistake in test input, fix it ASAP.

# @pytest.mark.asyncio
# async def test_prediction_endpoint(async_client, mock_ml_models):
#     mock_ml_models["text_classifier"].predict_label_async.return_value = "space"
#     response = await async_client.post(
#         "/prediction",
#         headers={"Content-type": "application/json"},
#         data={"text": "text in space"},
#     )
#     assert response.status_code == 200
#     assert response.json() == {"label": "space"}


# 422 Unprocessable Entity
@pytest.mark.asyncio
async def test_prediction_endpoint_invalid_request(async_client):
    response = await async_client.post("/prediction", json={})
    assert response.status_code == 422
