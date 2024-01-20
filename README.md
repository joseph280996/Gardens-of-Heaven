# Gardens of Heavens - Artificial Neural Network

## Prerequisite
```
python 3.12.0
```

## Design and Assumptions

Assumptions made on variables:
- The learning rate used is 0.1
- Bias will have the value of 1.
- For cycle or values convergion detection, this is currently hardcoded to have the mean squared error of 0.05 and overfit tolerance of 1000 times.
- The training ratio is 50% training, 25% for validation, and 25% for testing.

Design:
- The feature input data when accepted into the model will be normalized to be less than 1. Since most of the values are less than 10, I'll simply divide each input data by 10.
- The network only have a single hidden layer and the number of nodes is the number of features.
- It will have 3 neurons for the output and the hidden layer has 4 neurons for each input feature.
- All edges weight will be randomly assigned at first.
- The activation function is using sigmoid formula.
- The training termination is determine through either number of cycle or values converges.
- For number of cycle termination approach, user needs to provide a termination cycle.
- The valuation is performed through measuring of mean squared error and accuracy level.
- The input is processed first using OneHotEncoder.

## Installation
```
git clone git@github.com:joseph280996/Gardens-of-Heaven.git
cd Gardens-of-Heaven
pip install -r requirements.txt
python3 main.py
```

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
