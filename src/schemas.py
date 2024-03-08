from pydantic import BaseModel, Field


# Pydantic data models for data validation
class ApiRequest(BaseModel):
    text: str = Field(..., description="The text to perform the prediction on")


class ApiResponse(BaseModel):
    label: str = Field(..., description="The prediction label")
