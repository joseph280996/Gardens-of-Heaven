import numpy as np

from .Neuron import Neuron
from .TrainingTerminationConfig import TrainingTerminationConfig
from utils.generate_random_list import generate_random_list


class Iris_ANN:
    hidden_layer_output = None

    def __init__(
        self,
        training_feature_set: list[list[float]],
        training_labels: list[float],
        validation_set: list[list[float]],
        validation_label: list[list[float]],
        layers_shape: list[int],
        training_termination_type: str,
        training_termination_config: TrainingTerminationConfig,
    ) -> None:
        self.training_feature_set = np.asarray(training_feature_set) / 10
        self.training_labels = np.asarray(training_labels)
        self.validation_set = np.asarray(validation_set) / 10
        self.validation_label = np.asarray(validation_label)

        self.validation_mse = 1
        self.training_mse = 1
        self.overfit_count = 0
        self.target_mse = 0.05
        self.overfit_tolerance = 1000
        self.cycle_count = 0

        self.training_termination_type = training_termination_type
        self.termination_config = training_termination_config

        self.layers = self.__constructing_layers(layers_shape)

    def train(self) -> None:
        """
        Train the model using learning XOR algorithm with forward and backward propagation.
        We'll also perform validation for every training cycle to record for training termination.
        """
        while self.__should_continue_running():
            print(f"\n\nTraining Cycle: {self.cycle_count}")
            training_squared_error, correct_count = 0, 0
            for instance, label in zip(self.training_feature_set, self.training_labels):
                self.__forward_propagation(instance)
                forward_prop_output = self.__get_forward_prop_output()

                if label[np.argmax(forward_prop_output)] == 1:
                    correct_count += 1

                training_squared_error += np.sum((label - forward_prop_output) ** 2)
                self.__backward_propagation(instance, label)

            training_mse = training_squared_error / len(self.training_feature_set)
            print(f"Training MSE = {training_mse}", end="   ")
            print(
                "Training Accuracy: ",
                np.around(
                    correct_count / len(self.training_feature_set) * 100, decimals=4
                ),
                "%",
            )

            validation_mse = self.__validation()
            if (
                validation_mse >= self.validation_mse
                and training_mse < self.training_mse
            ):
                self.overfit_count += 1

            self.validation_mse = validation_mse
            self.training_mse = training_mse
            self.cycle_count += 1

    def predict(self, input: list[list[float]]) -> list[list[int]]:
        """
        This function will use the trained network to predict a given inputs and classify them accrodingly.
        It will use forward propagation to determine the output layer and record for each instance of the input as result.

        Arguments:
            input: the list of input for classification.
        Returns:
            The classification result.
        """
        reformat_input = np.asarray(input) / 10
        result = []
        for instance in reformat_input:
            self.__forward_propagation(instance)
            output_layer = self.layers[len(self.layers) - 1]
            forward_prop_output = [x.output for x in output_layer]

            instance_result = [0] * len(output_layer)
            instance_result[np.argmax(forward_prop_output)] = 1

            result.append(instance_result)
        return result

    def __should_continue_running(self):
        if self.training_termination_type in ["converge", "cycle"]:
            if self.training_termination_type == "converge":
                return (
                    self.overfit_count < self.termination_config.overfit_tolerance
                    and self.validation_mse > self.termination_config.target_mse
                )
            if self.training_termination_type == "cycle":
                return self.cycle_count < self.termination_config.cycle

            return False
        raise ValueError("Training termination type was not provided")

    def __get_forward_prop_output(self) -> list[float]:
        output_layer = self.layers[len(self.layers) - 1]
        return [x.output for x in output_layer]

    def __validation(self) -> float:
        validate_squared_error, correct_count = 0, 0
        for instance, label in zip(self.validation_set, self.validation_label):
            self.__forward_propagation(instance)
            forward_prop_output = self.__get_forward_prop_output()

            if label[np.argmax(forward_prop_output)] == 1:
                correct_count += 1

            validate_squared_error += np.sum((label - forward_prop_output) ** 2)

        validation_mse = validate_squared_error / len(self.validation_set)
        print(f"Validation MSE = {validation_mse}", end="   ")
        print(f"Validation Accuracy = {correct_count/len(self.validation_set)*100}")

        return validation_mse

    def __constructing_layers(self, layer_shape: list[int]) -> list[list[Neuron]]:
        result: list[list[Neuron]] = [[] for _ in range(len(layer_shape) - 1)]
        current_layer = 1

        while current_layer < len(layer_shape):
            neuron_count = 0

            while neuron_count < layer_shape[current_layer]:
                neuron_weights = generate_random_list(layer_shape[current_layer - 1])
                result[current_layer - 1].append(
                    Neuron(neuron_weights, np.random.rand())
                )
                neuron_count += 1
            current_layer += 1

        return result

    def __forward_propagation(self, instance: list[float]) -> None:
        for neuron in self.layers[0]:
            neuron.learn(instance)

        i = 1
        while i < len(self.layers):
            for neuron in self.layers[i]:
                neuron.learn([x.output for x in self.layers[i - 1]])
            i += 1

    def __calculate_output_layer_signal_error(self, label: list[float]):
        for output_neuron, expected_outcomes in zip(
            self.layers[len(self.layers) - 1], label
        ):
            backward_propagation_output = expected_outcomes - output_neuron.output
            output_neuron.calculate_signal_error(backward_propagation_output)

    def __calculate_backward_propagated_signal_error(
        self, edge_idx: int, layer_neurons: list[Neuron]
    ):
        return sum(
            [
                upper_layer_neuron.weights[edge_idx] * upper_layer_neuron.signal_error
                for upper_layer_neuron in layer_neurons
            ]
        )

    def __calculate_hidden_layer_signal_error(self):
        hidden_layer_idx = len(self.layers) - 2
        while hidden_layer_idx >= 0:
            hidden_layer_neuron_idx = 0
            while hidden_layer_neuron_idx < len(self.layers[hidden_layer_idx]):
                hidden_layer_neuron = self.layers[hidden_layer_idx][
                    hidden_layer_neuron_idx
                ]

                # Calculate the summation of all the upper layers neurons' signal error backward propagated to the current neuron
                backward_propagation_output = (
                    self.__calculate_backward_propagated_signal_error(
                        hidden_layer_neuron_idx, self.layers[hidden_layer_idx + 1]
                    )
                )

                hidden_layer_neuron.calculate_signal_error(backward_propagation_output)

                hidden_layer_neuron_idx += 1

            hidden_layer_idx -= 1

    def __update_weights_of_output_layer_neurons(self, label: list[float]):
        for output_neuron, _ in zip(self.layers[len(self.layers) - 1], label):
            output_neuron.recalculate_lower_level_weight(
                [
                    lower_layer.output
                    for lower_layer in self.layers[len(self.layers) - 2]
                ]
            )

    def __update_weights_of_hidden_layer_neurons(self, instance: list[float]):
        hidden_layer_idx = len(self.layers) - 2
        while hidden_layer_idx >= 0:
            hidden_layer_neuron_idx = 0

            while hidden_layer_neuron_idx < len(self.layers[hidden_layer_idx]):
                hidden_layer_neuron = self.layers[hidden_layer_idx][
                    hidden_layer_neuron_idx
                ]
                hidden_layer_neuron.recalculate_lower_level_weight(instance)
                hidden_layer_neuron_idx += 1
            hidden_layer_idx -= 1

    def __backward_propagation(self, instance: list[float], label: list[float]) -> None:
        # Calculating the signal error of all neurons
        self.__calculate_output_layer_signal_error(label)
        self.__calculate_hidden_layer_signal_error()

        # Update weights of all edges connected to all neuron_weights
        self.__update_weights_of_output_layer_neurons(label)
        self.__update_weights_of_hidden_layer_neurons(instance)
