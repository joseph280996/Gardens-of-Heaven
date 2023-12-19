from collections import defaultdict


def train_validate_test_split(
    data: list,
    config: dict[str, float] = {"train": 0.5, "validate": 0.25, "test": 0.25},
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    This will handle splitting the data into train, validation, and test set.
    It will handle selection in accordance with the type of label for each data.

    Arguments:
        data: The list of data that was parsed from database.
        config: The dictionary listing the splitting ratio.
    Returns:
        The trainning, validating, and testing set as tuple of 3
    """
    categories: dict[str, list[float]] = defaultdict(list)
    for instance in data:
        categories[instance[-1]].append(instance)

    training_set, validate_set, test_set = [], [], []
    for _, value in categories.items():
        trainning_index = round(len(value) * config["train"])
        validate_index = round(len(value) * (config["train"] + config["validate"]))

        training_set = training_set + value[:trainning_index]
        validate_set = validate_set + value[trainning_index:validate_index]
        test_set = test_set + value[validate_index:]

    return training_set, validate_set, test_set
