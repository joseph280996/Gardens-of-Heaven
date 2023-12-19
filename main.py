import argparse
from models import OnehotEncoder, Iris_ANN, TrainingTerminationConfig

from utils import Database, train_validate_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    default="data_files/ANN - Iris data",
    help="The path to the test prediction file",
)
parser.add_argument(
    "--terminationType",
    default="converge",
    help="Termination type of the training process [converges, interval]",
)
parser.add_argument(
    "--interval",
    type=int,
    default=1000,
    help="The number of interval to train the model for",
)

if __name__ == "__main__":
    args = parser.parse_args()
    database_file = getattr(args, "input")
    termination_type = getattr(args, "terminationType")
    epochs = getattr(args, "interval")

    database = Database(database_file)
    raw_data = database.get_data()

    train, validate, test = train_validate_test_split(raw_data)

    feature_set, labels, train_label_encoded_map = OnehotEncoder(train).run()
    validate_feature_set, validate_labels, validate_label_encoded_map = OnehotEncoder(validate).run()
    test_feature_set, test_labels, label_encoded_map = OnehotEncoder(test).run()

    output_count = 1
    prev_label = labels[0]
    for label in labels:
        if label != prev_label:
            output_count += 1
            prev_label = label

    model = Iris_ANN(
        feature_set,
        labels,
        validate_feature_set,
        validate_labels,
        [len(feature_set[0]), len(feature_set[0]), output_count],
        termination_type,
        TrainingTerminationConfig(1000, 0.05, epochs),
    )
    model.train()

    predictions = model.predict(test_feature_set)
    incorrect = 0

    count = 0
    for prediction, expected in zip(predictions, test_labels):
        print("\n\nPrediction #", count + 1)
        print("Expecting :", label_encoded_map[tuple(expected)])
        print("Predicted :", label_encoded_map[tuple(prediction)])
        if prediction != expected:
            incorrect += 1
        count += 1

    print(
        "Prediction Accuracy:",
        (len(test_feature_set) - incorrect) / len(test_feature_set) * 100,
        "%",
    )
