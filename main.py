from __future__ import annotations
"""(MSC) Morphological Source Code Framework – V0.0.12
================================================================================
<https://github.com/Phovos/msc> • MSC: Morphological Source Code © 2025 by Phovos
Standard Library Imports - 3.13 std libs **ONLY**"""
import re
import os
import io
import dis
import sys
import ast
import time
import site
import math
import enum
import mmap
import json
import uuid
import cmath
import shlex
import errno
import random
import socket
import struct
import shutil
import pickle
import pstats
import ctypes
import signal
import logging
import decimal
import tomllib
import weakref
import pathlib
import asyncio
import inspect
import hashlib
import tempfile
import cProfile
import argparse
import platform
import datetime
import traceback
import functools
import linecache
import importlib
import threading
import subprocess
import tracemalloc
import http.server
import collections
import multiprocessing
from logging import config
from io import StringIO
from decimal import Decimal, getcontext
from array import array
from pathlib import Path
from enum import Enum, auto, StrEnum, IntEnum
from queue import Queue, Empty
from abc import ABC, abstractmethod
from threading import Thread, RLock
from dataclasses import dataclass, field
from logging import Formatter, StreamHandler
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import reduce, lru_cache, partial, wraps
from contextlib import contextmanager, asynccontextmanager, AbstractContextManager
from importlib.util import spec_from_file_location, module_from_spec
from types import SimpleNamespace, ModuleType,  MethodType, FunctionType, CodeType, TracebackType, FrameType
from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Set, Iterator, OrderedDict,
    Coroutine, Type, NamedTuple, ClassVar, Protocol, runtime_checkable, AsyncIterator,
)
try:
    if platform.system() == "Windows":
        from ctypes import windll, byref, wintypes
        # from ctypes.wintypes import HANDLE, DWORD, LPWSTR, LPVOID, BOOL
        from pathlib import PureWindowsPath
        class WindowsConsole:
            """Enable ANSI escape sequences on Windows consoles."""
            @staticmethod
            def enable_ansi():
                STD_OUTPUT_HANDLE = -11
                ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                kernel32 = windll.kernel32
                handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
                mode = wintypes.DWORD()
                if kernel32.GetConsoleMode(handle, byref(mode)):
                    new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                    kernel32.SetConsoleMode(handle, new_mode)
                else:
                    raise RuntimeError("Failed to get console mode for enabling ANSI.")
        try:
            WindowsConsole.enable_ansi()
        except Exception as e:
            print(f"Failed to enable ANSI escape codes on Windows console: {e}", file=sys.stderr)
            sys.exit(1)
except (ImportError, OSError, RuntimeError) as e:
    # If enabling ANSI fails (e.g., not a real console), print a warning
    # but don't exit. The program can run, just without colors.
    print(f"Warning: Failed to enable ANSI escape codes on Windows: {e}", file=sys.stderr)
@dataclass
class LogAdapter:
    """
    Sets up console + file + broadcast + (optional) queue handlers, and exposes a correlation-aware logger instance. You can customize handler levels and output filenames at instantiation time.
    ---
    | Token             | Meaning                                      |
    | ----------------- | -------------------------------------------- |
    | `%(asctime)s`     | Timestamp                                    |
    | `%(name)s`        | Logger name (`__name__`)                     |
    | `%(levelname)s`   | Log level name                               |
    | `%(message)s`     | The actual message                           |
    | `%(filename)s`    | File name where the log is emitted           |
    | `%(lineno)d`      | Line number of the log statement             |
    | `%(funcName)s`    | Function name                                |
    | `%(threadName)s`  | Thread name (super useful w/ `threading`)    |
    | `%(processName)s` | Process name (helpful for `multiprocessing`) |
    ---

    """
    console_level: str = "INFO"
    file_filename: str = "app.log"
    file_level: str = "INFO"
    broadcast_filename: str = "broadcast.log"
    broadcast_level: str = "INFO"
    queue_size: int = -1  # -1 = infinite
    correlation_id: str = "SYSTEM"
    LOGGING_CONFIG: dict = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.LOGGING_CONFIG = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '[%(levelname)s] %(asctime)s | [%(filename)s:%(lineno)d]: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'color': {
                    '()': self.ColorFormatter,
                    'format': '[%(levelname)s] %(asctime)s | [%(filename)s:%(lineno)d]: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.console_level,
                    'formatter': 'color',
                    'stream': 'ext://sys.stdout',
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.file_level,
                    'formatter': 'default',
                    'filename': self.file_filename,
                    'maxBytes': 10 * 1024 * 1024,
                    'backupCount': 5,
                    'encoding': 'utf-8'
                },
                'broadcast': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.broadcast_level,
                    'formatter': 'default',
                    'filename': self.broadcast_filename,
                    'maxBytes': 10 * 1024 * 1024,
                    'backupCount': 5,
                    'encoding': 'utf-8'
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file', 'broadcast']
            }
        }

        if self.queue_size is not None:
            self.LOGGING_CONFIG['handlers']['queue'] = {
                'class': 'logging.handlers.QueueHandler',
                'level': 'INFO',
                'formatter': 'default',
                'queue': multiprocessing.Queue(self.queue_size)
            }
            self.LOGGING_CONFIG['root']['handlers'].append('queue')

        logging.config.dictConfig(self.LOGGING_CONFIG)
        base_logger = logging.getLogger(__name__)
        self.logger = self.CorrelationLogger(base_logger, {"cid": self.correlation_id})

        self.logger.info("Logger initialized with [console, file, broadcast] handlers.")

    class CorrelationLogger(logging.LoggerAdapter):
        def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
            cid = self.extra.get("cid", "SYSTEM")
            return f"[{cid}] {msg}", kwargs

    class ColorFormatter(logging.Formatter):
        _COLORS = {
            logging.DEBUG: "\033[34m",    # Blue
            logging.INFO: "\033[32m",     # Green
            logging.WARNING: "\033[33m",  # Yellow
            logging.ERROR: "\033[31m",    # Red
            logging.CRITICAL: "\033[41m"  # Red background
        }
        _RESET = "\033[0m"

        def format(self, record: logging.LogRecord) -> str:
            base = super().format(record)
            color = self._COLORS.get(record.levelno, self._COLORS[logging.DEBUG])
            return f"{color}{base}{self._RESET}"
T = TypeVar('T')
V = TypeVar('V')
C = TypeVar('C')

# Global Registry for Morphological Classes and Functions
MSC_REGISTRY: Dict[str, Set[str]] = {'classes': set(), 'functions': set()}

# Exception for Morphodynamic Collapse
class MorphodynamicCollapse(Exception):
    """Raised when a morph object destabilizes under thermal pressure."""
    pass

# MorphSpec Blueprint for Morphological Classes
@dataclass
class MorphSpec:
    """Blueprint for morphological classes."""
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str

# Morphology Decorator for Class Registration and Validation
def morphology(source_model: Type) -> Callable[[Type], Type]:
    """Decorator: register & validate a class against a MorphSpec."""
    def decorator(target: Type) -> Type:
        target.__msc_source__ = source_model
        # Ensure target has all annotated fields from source_model
        for field_name in getattr(source_model, '__annotations__', {}):
            if field_name not in getattr(target, '__annotations__', {}):
                raise TypeError(f"{target.__name__} missing field '{field_name}'")
        MSC_REGISTRY['classes'].add(target.__name__)
        return target
    return decorator

# MorphicComplex: Complex Numbers with Morphic Properties
@dataclass
class MorphicComplex:
    """Represents a complex number with morphic properties."""
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def conjugate(self) -> 'MorphicComplex':
        """Return the complex conjugate."""
        return MorphicComplex(self.real, -self.imag)

    def __add__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        """Add two MorphicComplex numbers."""
        return MorphicComplex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        """Multiply two MorphicComplex numbers."""
        return MorphicComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )

    def __eq__(self, other: 'MorphicComplex') -> bool:
        """Check equality between two MorphicComplex numbers."""
        return isinstance(other, MorphicComplex) and self.real == other.real and self.imag == other.imag

    def __repr__(self) -> str:
        """String representation of the MorphicComplex number."""
        if self.imag == 0:
            return f"{self.real}"
        sign = "+" if self.imag >= 0 else ""
        return f"{self.real}{sign}{self.imag}j"

# Morphological Rule for Symbolic Rewriting
class MorphologicalRule:
    def __init__(self, symmetry: str, conservation: str, lhs: str, rhs: List[str]):
        self.symmetry = symmetry
        self.conservation = conservation
        self.lhs = lhs
        self.rhs = rhs

    def apply(self, seq: List[str]) -> List[str]:
        """Apply the rule to a sequence."""
        if self.lhs in seq:
            idx = seq.index(self.lhs)
            return seq[:idx] + self.rhs + seq[idx+1:]
        return seq

# Enumerations for Quantum States and Entanglement Types
class QState(enum.Enum):
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    DECOHERENT = "decoherent"
    EIGENSTATE = "eigenstate"

class EntanglementType(enum.Enum):
    CODE_LINEAGE = "code_lineage"
    TEMPORAL_SYNC = "temporal_sync"
    SEMANTIC_BRIDGE = "semantic_bridge"
    PROBABILITY_FIELD = "probability_field"

# Kronecker Field Function for Quantum Coherence
def kronecker_field(q1: Any, q2: Any, temperature: float) -> float:
    dot = sum(a * b for a, b in zip(q1.state.vector, q2.state.vector))
    if temperature > 0.5:
        return math.cos(dot)
    return 1.0 if dot > 0.99 else 0.0

# Elevate Function for Morphological Class Instantiation
def elevate(data: Any, cls: Type) -> object:
    """Raise a dict or object to a registered morphological class."""
    if not hasattr(cls, '__msc_source__'):
        raise TypeError(f"{cls.__name__} is not a morphological class.")
    source = cls.__msc_source__
    kwargs = {k: getattr(data, k, data.get(k)) for k in source.__annotations__}
    return cls(**kwargs)

# Functional Programming Patterns - Transducers
class Reduced:
    """Sentinel class to signal early termination during reduction."""
    def __init__(self, val: Any):
        self.val = val

def ensure_reduced(x: Any) -> Union[Any, Reduced]:
    """Ensure the value is wrapped in a Reduced sentinel."""
    return x if isinstance(x, Reduced) else Reduced(x)

def unreduced(x: Any) -> Any:
    """Unwrap a Reduced value or return the value itself."""
    return x.val if isinstance(x, Reduced) else x

def reduce(function: Callable[[Any, T], Any], iterable: Iterable[T], initializer: Any = None) -> Any:
    """A custom reduce implementation that supports early termination with Reduced."""
    accum_value = initializer if initializer is not None else function()
    for x in iterable:
        accum_value = function(accum_value, x)
        if isinstance(accum_value, Reduced):
            return accum_value.val
    return accum_value

# Base Transducer Class
class Transducer(ABC):
    """Base class for defining transducers."""
    @abstractmethod
    def __call__(self, step: Callable[[Any, T], Any]) -> Callable[[Any, T], Any]:
        """The transducer's __call__ method allows it to be used as a decorator."""
        pass

class Map(Transducer):
    """Transducer for mapping elements with a function."""
    def __init__(self, f: Callable[[T], Any]):
        self.f = f

    def __call__(self, step: Callable[[Any, T], Any]) -> Callable[[Any, T], Any]:
        def new_step(r, x):
            return step(r, self.f(x))
        return new_step

class Filter(Transducer):
    """Transducer for filtering elements based on a predicate."""
    def __init__(self, pred: Callable[[T], bool]):
        self.pred = pred

    def __call__(self, step: Callable[[Any, T], Any]) -> Callable[[Any, T], Any]:
        def new_step(r, x):
            return step(r, x) if self.pred(x) else r
        return new_step

class Cat(Transducer):
    """Transducer for flattening nested collections."""
    def __call__(self, step: Callable[[Any, T], Any]) -> Callable[[Any, T], Any]:
        def new_step(r, x):
            if not hasattr(x, '__iter__'):
                raise TypeError(f"Expected iterable, got {type(x)} with value {x}")
            result = r
            for item in x:
                result = step(result, item)
                if isinstance(result, Reduced):
                    return result
            return result
        return new_step

# Utility Functions for Transducers
def compose(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose functions in reverse order."""
    return reduce(lambda f, g: lambda x: f(g(x)), reversed(fns))

def transduce(xform: Transducer, f: Callable[[Any, T], Any], start: Any, coll: Iterable[T]) -> Any:
    """Apply a transducer to a collection with an initial value."""
    reducer = xform(f)
    return reduce(reducer, coll, start)

def mapcat(f: Callable[[T], Iterable[Any]]) -> Transducer:
    """Map then flatten results into one collection."""
    return compose(Map(f), Cat())

def into(target: Union[list, set], xducer: Transducer, coll: Iterable[T]) -> Any:
    """Apply transducer and collect results into a target container."""
    def append(r, x):
        if hasattr(r, 'append'):
            r.append(x)
        elif hasattr(r, 'add'):
            r.add(x)
        return r
    return transduce(xducer, append, target, coll)

# Mapper Function for Transforming Input Data
def mapper(mapping_description: Dict[str, Any], input_data: Dict[str, Any]):
    def transform(xform, value):
        if callable(xform):
            return xform(value)
        elif isinstance(xform, dict):
            return {k: transform(v, value) for k, v in xform.items()}
        else:
            raise ValueError(f"Invalid transformation type: {type(xform)}. Expected callable or Mapping.")

    def get_value(key):
        if isinstance(key, str) and key.startswith(":"):
            return input_data.get(key[1:])
        return input_data.get(key)

    def process_mapping(mapping_description):
        result = {}
        for key, xform in mapping_description.items():
            if isinstance(xform, str):
                value = get_value(xform)
                result[key] = value
            elif isinstance(xform, dict):
                if "key" in xform:
                    value = get_value(xform["key"])
                    if "xform" in xform:
                        result[key] = transform(xform["xform"], value)
                    elif "xf" in xform:
                        if isinstance(value, list):
                            transformed = [xform["xf"](v) for v in value]
                            if "f" in xform:
                                result[key] = xform["f"](transformed)
                            else:
                                result[key] = transformed
                        else:
                            result[key] = xform["xf"](value)
                    else:
                        result[key] = value
                else:
                    result[key] = process_mapping(xform)
            else:
                result[key] = xform
        return result

    return process_mapping(mapping_description)

# Helper Function for Formatting Complex Matrices
def format_complex_matrix(matrix: List[List[complex]], precision: int = 3) -> str:
    """Helper function to format complex matrices for printing."""
    result = []
    for row in matrix:
        formatted_row = []
        for elem in row:
            if not isinstance(elem, complex):
                raise ValueError(f"Expected complex number, got {type(elem)}.")
            real = round(elem.real, precision)
            imag = round(elem.imag, precision)
            if abs(imag) < 1e-10:
                formatted_row.append(f"{real:6.3f}")
            else:
                formatted_row.append(f"{real:6.3f}{'+' if imag >= 0 else ''}{imag:6.3f}j")
        result.append("[" + ", ".join(formatted_row) + "]")
    return "[\n " + "\n ".join(result) + "\n]"

if __name__ == "__main__":
    log = LogAdapter()
    log.logger.debug("Debug level test; \"Hello world!\"")
    log.logger.warning("Warning with CID and colors")
    log = LogAdapter(correlation_id="INIT")
    log.logger.warning("Warning with CID and colors")
    log.logger.error("Error occurred in something")
    log = LogAdapter(correlation_id="RUNTIME")
    log.logger.critical("Critical issue reported")
    print(f'Find logs @ {log.broadcast_filename}')