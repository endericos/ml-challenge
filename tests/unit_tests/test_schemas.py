import sys

sys.path.append("src/")

from src.schemas import ApiRequest, ApiResponse
from pydantic import ValidationError
import pytest


# Test valid input for ApiRequest
def test_api_request_valid():
    request_data = {"text": "sample text written in space"}
    api_request = ApiRequest(**request_data)
    assert api_request.text == "sample text written in space"


# Test invalid input for ApiRequest by missing required field 'text'
def test_api_request_invalid():
    with pytest.raises(ValidationError):
        ApiRequest()


# Test valid input for ApiResponse
def test_api_response_valid():
    response_data = {"label": "space"}
    api_response = ApiResponse(**response_data)
    assert api_response.label == "space"


# Test invalid input for ApiResponse using missing required field 'label'
def test_api_response_invalid():
    with pytest.raises(ValidationError):
        ApiResponse()
