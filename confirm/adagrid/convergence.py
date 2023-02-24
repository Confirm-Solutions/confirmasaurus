from enum import Enum


class WorkerStatus(Enum):
    # Statuses which terminate the worker.
    CONVERGED = 0
    REACHED_N_STEPS = 1
    # Statuses which end the self-help stage.
    EMPTY_STEP = 2
    NO_NEW_TILES = 3
    # Normal solo work statuses.
    NEW_STEP = 4
    WORKING = 5
    ALREADY_EXISTS = 6
    EMPTY_PACKET = 7
    # Coordination
    COORDINATED = 8

    def need_coordination(self):
        return (self == WorkerStatus.CONVERGED) or (self == WorkerStatus.EMPTY_STEP)

    def done(self):
        return (
            (self == WorkerStatus.REACHED_N_STEPS)
            or (self == WorkerStatus.CONVERGED)
            or (self == WorkerStatus.EMPTY_STEP)
        )
