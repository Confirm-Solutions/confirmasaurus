import os

from jax.config import config

dir_path = os.path.dirname(os.path.realpath(__file__))
# This avoids errors that occur when imprint is in a subtree.
if dir_path == os.getcwd():
    pytest_plugins = ["imprint.testing"]

config.update("jax_enable_x64", True)
