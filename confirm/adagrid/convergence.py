from enum import Enum


class WorkerStatus(Enum):
    # Statuses which terminate the worker.
    CONVERGED = 0
    REACHED_N_STEPS = 1
    # Statuses which end the self-help stage.
    EMPTY_STEP = 2
    NO_NEW_TILES = 3
    COORDINATION = 4
    # Normal solo work statuses.
    NEW_STEP = 5
    WORKING = 6
    WORK_DONE = 7
    SKIPPED = 8  # The packet had already been processed by another worker.
    # Coordination
    COORDINATED = 9

    def done(self):
        return (
            (self == WorkerStatus.REACHED_N_STEPS)
            or (self == WorkerStatus.CONVERGED)
            or (self == WorkerStatus.EMPTY_STEP)
        )
