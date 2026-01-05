"""
Enumeration definitions for the publication extraction system.

This module defines status enumerations used throughout the application
to track the processing state of datasets and publications.
"""

from enum import Enum


class Status(Enum):
    """
    Enumeration representing the processing status of a dataset or task.
    
    Attributes:
        WAITING: Initial state, task is queued but not yet started.
        PROCESSING: Task is currently being processed by workers.
        DONE: Task has completed successfully.
    """
    WAITING = 1
    PROCESSING = 2
    DONE = 3