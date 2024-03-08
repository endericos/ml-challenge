import gzip
import json
from typing import List
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn, get_latest_tested_opset_version
from skl2onnx.common.data_types import StringTensorType


def define_pipe() -> Pipeline:
    """
    Creates a composite sklearn Pipeline object by chaining
    preprocessing and learner steps.
    """
    pipe = Pipeline(
        steps=[
            ("count_vectorizer", CountVectorizer()),
            ("tfidf_transformer", TfidfTransformer()),
            ("classifier", MultinomialNB()),
        ]
    )
    return pipe


def persist_model(pipeline: Pipeline, path: str):
    """
    Converts a composite sklearn pipeline into ONNX format
    and persists it into disk.
    """
    model_onnx = convert_sklearn(
        model=pipeline,
        name="text_classifier",
        initial_types=[("input", StringTensorType([None, 1]))],  # batch_size, num_feats
        target_opset=get_latest_tested_opset_version(),
    )
    with open(path, "wb") as f:
        f.write(model_onnx.SerializeToString())


def evaluate_model(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, dict]:
    """
    Evaluates micro, macro, weighted averages of precision, recall and f1-scores
    of model predictions and returns them in a nested dictionary.
    """
    evaluation = {}
    for avg in ["micro", "macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average=avg, zero_division=0
        )
        evaluation[avg] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1-score": round(f, 3),
        }
    return evaluation


def store_evaluation(evaluation: dict, path: str):
    """
    Stores evaluation dictionary as a json file into disk.
    """
    with open(path, "w") as f:
        json.dump(evaluation, f)


def load_dataset(path: str) -> List[dict]:
    data = []
    with gzip.open(path, "rb") as f_in:
        for line in f_in:
            data.append(json.loads(line))
    return data
