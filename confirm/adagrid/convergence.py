from enum import Enum


class WorkerStatus(Enum):
    INCOMPLETE = 0
    CONVERGED = 1
    FAILED = 2
    STUCK = 3
    NEW_PACKET = 4
