from enum import Enum


class WorkerStatus(Enum):
    CONVERGED = 0
    FAILED = 1
    REACHED_N_STEPS = 2
    WORKING = 3
    NEW_STEP = 4
    STUCK = 5

    def done(self):
        return (
            (self == WorkerStatus.REACHED_N_STEPS)
            or (self == WorkerStatus.CONVERGED)
            or (self == WorkerStatus.FAILED)
        )
