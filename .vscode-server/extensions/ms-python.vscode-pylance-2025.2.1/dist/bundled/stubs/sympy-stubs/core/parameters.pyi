from contextlib import contextmanager
from threading import local
from typing import Any, Generator

class _global_parameters(local):
    def __init__(self, **kwargs) -> None: ...
    def __setattr__(self, name, value) -> None: ...

global_parameters = ...

@contextmanager
def evaluate(x) -> Generator[None, Any, None]: ...
@contextmanager
def distribute(x) -> Generator[None, Any, None]: ...
