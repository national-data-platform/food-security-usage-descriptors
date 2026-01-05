"""
Source package for the academic publications metadata extraction system.

This package provides the core infrastructure and business logic for extracting,
processing, and storing scholarly publication metadata from various academic APIs.

Modules:
    infra: Infrastructure components including configuration, queues, and repositories.
    use_cases: Business logic implementations for different processing workflows.
"""

from .infra import *
from .infra.queue import *
from .infra.repository import *
from .infra.configuration import *

