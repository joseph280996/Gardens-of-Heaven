from collections import defaultdict


class OnehotEncoder:
    """
    This class represent the One hot encoder that is used for preprocessing the input data.
    Property:
        raw_data: the raw data from the database
    """

    def __init__(self, raw_data):
        self.raw_data = raw_data

    def run(self):
        """
        This function will handle recognizing the amount of categories there are in the label and create appropriate numerical labeling for each label.
        It will also handle splitting the labeling from the input data.

        Returns:
            Feature data set and labeling set as tuple.
        """
        categories: dict[str, list[float]] = defaultdict(list)
        for data in self.raw_data:
            categories[data[-1]].append(data[:-1])

        labels = []
        label = [0] * len(categories)
        change_idx = len(categories) - 1
        label[change_idx] = 1

        prev_label = list(categories.keys())[0]
        label_encoded_map = {tuple(label.copy()): list(categories.keys())[0]}

        data = []
        for key, values in categories.items():
            if prev_label != key:
                prev_label = key
                label[change_idx] = 0
                change_idx -= 1
                label[change_idx] = 1
                label_encoded_map[tuple(label.copy())] = key

            for value in values:
                data.append(value)
                labels.append(label.copy())

        return data, labels, label_encoded_map
