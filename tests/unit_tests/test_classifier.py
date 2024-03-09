import sys

sys.path.append("src/")

import pytest
from unittest.mock import MagicMock, patch
from src.classifier import load_model, predict_label, predict_label_async


@pytest.fixture
def mock_inference_session():
    mock_session = MagicMock()
    return mock_session


@pytest.fixture
def sample_model_path():
    return "src/text_classifier.onnx"


@pytest.fixture
def sample_text():
    return "sample text written by some species living in space"


@patch("src.classifier.InferenceSession")
def test_load_model(mock_inference_session, sample_model_path):
    mock_inference_session.return_value = mock_inference_session
    loaded_model = load_model(sample_model_path)
    assert loaded_model == mock_inference_session


def test_predict_label(mock_inference_session, sample_text):
    mock_result = ["space"]
    mock_inference_session.run.return_value = [mock_result]
    label = predict_label(mock_inference_session, sample_text)
    assert label == "space"


@pytest.mark.asyncio
async def test_predict_label_async(mock_inference_session, sample_text):
    mock_result = ["space"]
    mock_inference_session.run.return_value = [mock_result]
    label = await predict_label_async(mock_inference_session, sample_text)
    assert label == "space"


@pytest.mark.asyncio
async def test_predict_label_async_error(mock_inference_session, sample_text):
    mock_inference_session.run.side_effect = Exception("Some error")
    with pytest.raises(Exception):
        await predict_label_async(mock_inference_session, sample_text)
