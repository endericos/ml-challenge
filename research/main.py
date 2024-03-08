import argparse
from typing import Iterable
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        default="data/train.jsonl.gz",
        help="path to training data (.jsonl.gz file)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/test.jsonl.gz",
        help="path to test data (.jsonl.gz file)",
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        default="evaluation.json",
        help="path to store evaluation (.json file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text_classifier.onnx",
        help="path to persist the trained model (.onnx file)",
    )
    return parser.parse_args()


def main(args):
    train_data = utils.load_dataset(args.train)
    pipe = utils.define_pipe()
    pipe.fit(*zip(*[(x["text"], x["label"]) for x in train_data]))

    utils.persist_model(pipe, args.model)

    test_data = utils.load_dataset(args.test)
    y_pred = pipe.predict([x["text"] for x in test_data])

    evaluation = utils.evaluate_model([x["label"] for x in test_data], y_pred)
    utils.store_evaluation(evaluation, args.evaluation)


if __name__ == "__main__":
    main(parse_args())
