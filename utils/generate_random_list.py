import random

def generate_random_list(size: int) -> list[float]:
    """
    This function is used to generate a list of random number.

    Argument: 
        size: the desired size of the array.
    Return:
        An array of random number with the given size.
    """
    result = []
    for _ in range(size):
        result.append(random.random())

    return result
