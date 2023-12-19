class TrainingTerminationConfig:
    """
    This class represent the termination config for all termination type.
    Properties:
        overfit_tolerance: The overfit tolerance
        target_mse: The mean squared error that we're targetting.
        cycle: The number of cycles to terminate the training
    """

    def __init__(self, overfit_tolerance: int, target_mse: float, cycle: int):
        self.overfit_tolerance = overfit_tolerance
        self.target_mse = target_mse
        self.cycle = cycle
