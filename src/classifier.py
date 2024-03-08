import asyncio
from onnxruntime import InferenceSession


def load_model(model_path: str) -> InferenceSession:
    """
    Returns an initialised an ONNX runtime loaded with trained model.
    """
    onnx_session = InferenceSession(model_path)
    return onnx_session


def predict_label(model: InferenceSession, text: str) -> str:
    """
    Makes prediction in ONNX runtime and returns the result.

    ONNX expects an array of dim: (batch_size, num_feats),
    In our data, this means a reshaped (-1, 1) array, hence input wrapped with [[]].
    """
    result = model.run(None, {"input": [[text]]})
    return result[0][0]


async def predict_label_async(model: InferenceSession, text: str) -> str:
    """
    Asynchronous implementation of `predict_label` function to
    allow for non-blocking execution and concurrent processing.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, predict_label, model, text)
