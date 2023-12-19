import numpy as np


class Neuron:
    """
    This class represent a Neuron in a network
    Properties:
        weights: List of all weights of edges connecting from lower layer
        bias: The bias value
        bias_w: The weight of edge from this neuron to bias
        output: The activated value of the neuron
        learning_rate: The learning rate used when recalculating the weights
        signal_error: The signal error calculated for each new weights
    """

    def __init__(self, weights: list[float], bias_w: float):
        self.weights = weights
        self.bias = 1
        self.bias_w = bias_w
        self.output = 0
        self.learning_rate = 0.1
        self.signal_error = 0

    def learn(self, input: list[float]):
        """
        The learning trigger of a neuron.
        This will first calculate the potentials and call activation function to determine the output and record it.
        Arguments:
            input: The instance of the input that we're calculating on.
        Returns:

        """
        potential = self.__potentials(input)
        return self.__activate(potential)

    def calculate_signal_error(self, delta_error: float):
        """
        This function will calculate the signal error at the given neuron which will be used to recalculate the weights for all connected neurons.
        Arguments:
            delta_error: The second part of the equation where on the output layer will be the subtraction between the expected values and the hidden
                         layers are the summation of all the connected upper layers' signal error times the edge weight.
        """
        self.signal_error = self.__derivative_activation() * delta_error

    def recalculate_lower_level_weight(self, lower_layer_outputs: list[float]) -> None:
        """
        This function will recalculate all the weights of the lower level connected neurons.

        Arguments:
            lower_layer_outputs: The outputs of all the lower layers in the network.
        """
        self.bias_w += self.learning_rate * self.bias * self.signal_error
        for i in range(len(self.weights)):
            self.weights[i] += (
                self.learning_rate * self.signal_error * lower_layer_outputs[i]
            )

    def __potentials(self, input: list[float]) -> float:
        return sum([x * y for x, y in zip(self.weights, input)]) + (
            self.bias * self.bias_w
        )

    def __activate(self, potential: float):
        self.output = 1 / (1 + np.exp(-potential))
        return self.output

    def __derivative_activation(self) -> float:
        return self.output * (1 - self.output)
