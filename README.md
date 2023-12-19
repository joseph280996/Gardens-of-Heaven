# Gardens of Heavens - Artificial Neural Network

I'll be using my first late token on this assignment for a 4-day extension.

## Assumptions

- I'll assume that the version of python being used is 3.12.
- I'm going to use NumPy for some data manipulation and mathematical functions, the version of NumPy can be found in requirements.txt which you can use pip to install.
- The network only have a single hidden layer and the number of nodes is the number of features.
- It will have 3 neurons for the output and the hidden layer has 4 neurons for each input feature.
- Bias will have the value of 1.
- All edges weight will be randomly assigned at first.
- The activation function is using sigmoid formula.
- The training termination is determine through either number of cycle or values converges.
- For number of cycle termination approach, user needs to provide a termination cycle.
- For cycle or values converges, this is currently hardcoded to have the mean squared error of 0.05 and overfit tolerance of 1000 times.
- The valuation is performed through measuring of mean squared error and accuracy level.
- The training ratio that I'll be using is 50% training, 25% for validation, and 25% for testing.
- The feature input data when accepted into the model will be normalized to be less than 1. Since most of the values are less than 10, I'll simply divide each input data by 10.
- The input is processed first using OneHotEncoder.
- The learning rate used is 0.1

## Usage

```
usage: main.py [-h] [-i INPUT] --terminationType TERMINATIONTYPE [--interval INTERVAL]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The path to the test prediction file
  --terminationType TERMINATIONTYPE
                        Termination type of the training process [converges, interval]
  --interval INTERVAL   The number of interval to train the model for
```

## Usage example:
Run with default settings:
```
python3 main.py
```

Run with converge settings, this is the same as default settings:
```
python3 main.py --terminationType converge
```

Run with cycle settings:
```
python3 main.py --terminationType cycle --interval 1000
```
