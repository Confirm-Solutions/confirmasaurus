import contextvars
import logging

worker_id = contextvars.ContextVar("worker_id", default=None)


class Adapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[worker_id={worker_id.get()}] {msg}", kwargs


def getLogger(name):
    """
    A replacement for logging.getLogger that adds the worker_id to the log message.

    Args:
        name: the name of the logger

    Returns:
        A LoggerAdapter that adds the worker_id to the log message.
    """
    return Adapter(logging.getLogger(name), {})
