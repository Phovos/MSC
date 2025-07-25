#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
serverTest.py: Pedagogical test suite for a reflexive, quinean SDK.

Integrates with server.py to demonstrate:
- Agency and DynamicSystem for state transformations
- Quantum-inspired CPythonFrame and HilbertSpace
- PyWord for low-level memory management
- JSONRPCServer interactions (HTTP)
- Quinean self-replication via QuineOracle

Dependencies: Standard library only (socket, ctypes, json, math, etc.)
"""

import sys
import json
import time
import asyncio
import ctypes
import logging
import random
import math
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from pathlib import Path
from enum import Enum, IntEnum
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Explicit imports from server.py
from server import (
    JSONRPCServer,
    AppConfig,
    CodeRequest,
    SecurityContext,
    AccessPolicy,
    AccessLevel,
    performance_metrics,
    AppError,
)

# REPO_ROOT for paths
REPO_ROOT = Path(__file__).parent.parent

# ==========================================================================
# ENUMS AND CONSTANTS
# ==========================================================================


class Morphology(Enum):
    """Represents morphic state of a computational entity."""

    MORPHIC = 0  # Stable, low-energy
    DYNAMIC = 1  # High-energy, transformative
    MARKOVIAN = -1  # Forward-evolving
    NON_MARKOVIAN = math.e  # Reversible, memoryful


class QuantumState(Enum):
    """Tracks quantum-like properties of objects."""

    SUPERPOSITION = 1
    ENTANGLED = 2
    COLLAPSED = 4
    DECOHERENT = 8


class WordAlignment(IntEnum):
    """Standardized computational word sizes."""

    UNALIGNED = 1
    WORD = 2
    DWORD = 4
    QWORD = 8
    CACHE_LINE = 64
    PAGE = 4096


# ==========================================================================
# QUANTUM PRIMITIVES
# ==========================================================================


class MorphicComplex:
    """Complex number with morphic properties."""

    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def conjugate(self) -> 'MorphicComplex':
        return MorphicComplex(self.real, -self.imag)

    def __add__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: Union['MorphicComplex', float, int]) -> 'MorphicComplex':
        if isinstance(other, (int, float)):
            return MorphicComplex(self.real * other, self.imag * other)
        return MorphicComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __rmul__(self, other: Union[float, int]) -> 'MorphicComplex':
        return self.__mul__(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MorphicComplex):
            return False
        return (
            abs(self.real - other.real) < 1e-10 and abs(self.imag - other.imag) < 1e-10
        )

    def __hash__(self) -> int:
        return hash((self.real, self.imag))

    def __repr__(self) -> str:
        sign = "+" if self.imag >= 0 else ""
        return f"{self.real}{sign}{self.imag}j"


class HilbertSpace:
    """Hilbert space with MorphicComplex coordinates."""

    def __init__(self, dimension: int = 3):
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        self.dimension = dimension
        self.basis_vectors = [
            [MorphicComplex(1 if i == j else 0, 0) for j in range(dimension)]
            for i in range(dimension)
        ]

    def inner_product(
        self, vec1: List[MorphicComplex], vec2: List[MorphicComplex]
    ) -> MorphicComplex:
        if len(vec1) != len(vec2) or len(vec1) != self.dimension:
            raise ValueError("Vectors must match Hilbert space dimension")
        result = MorphicComplex(0, 0)
        for u, v in zip(vec1, vec2, strict=False):
            result = result + (u.conjugate() * v)
        return result

    def norm(self, vector: List[MorphicComplex]) -> float:
        inner = self.inner_product(vector, vector)
        return math.sqrt(inner.real)

    def normalize(self, vector: List[MorphicComplex]) -> List[MorphicComplex]:
        norm_val = self.norm(vector)
        if abs(norm_val) < 1e-10:
            raise ValueError("Cannot normalize zero vector")
        return [MorphicComplex(c.real / norm_val, c.imag / norm_val) for c in vector]


class QuantumOperator:
    """Quantum operator as a matrix in a Hilbert space."""

    def __init__(self, hilbert_space: HilbertSpace, matrix: List[List[MorphicComplex]]):
        if len(matrix) != hilbert_space.dimension or any(
            len(row) != hilbert_space.dimension for row in matrix
        ):
            raise ValueError("Matrix must match Hilbert space dimension")
        self.hilbert_space = hilbert_space
        self.matrix = matrix

    def apply(self, state_vector: List[MorphicComplex]) -> List[MorphicComplex]:
        if len(state_vector) != self.hilbert_space.dimension:
            raise ValueError("Vector dimension mismatch")
        result = [MorphicComplex(0, 0) for _ in range(self.hilbert_space.dimension)]
        for i in range(self.hilbert_space.dimension):
            for j in range(self.hilbert_space.dimension):
                result[i] = result[i] + (self.matrix[i][j] * state_vector[j])
        return result


# ==========================================================================
# CPYTHON INTEGRATION
# ==========================================================================


class BYTE_WORD:
    """Basic 8-bit word representation."""

    def __init__(self, value: int = 0):
        if not 0 <= value <= 255:
            raise ValueError("BYTE_WORD value must be between 0 and 255")
        self.value = value

    def get_bit(self, pos: int) -> int:
        return (self.value >> pos) & 1

    def flip_bit(self, pos: int) -> None:
        self.value ^= 1 << pos

    def __repr__(self) -> str:
        return f"BYTE_WORD(value={self.value:08b})"


class PyWord:
    """Word-sized value optimized for CPython integration."""

    __slots__ = ('_value', '_alignment')

    def __init__(
        self, value: Union[int, bytes], alignment: WordAlignment = WordAlignment.WORD
    ):
        self._alignment = alignment
        aligned_size = self._calculate_aligned_size()
        self._value = self._allocate_aligned(aligned_size)
        self._store_value(value)

    def _calculate_aligned_size(self) -> int:
        base_size = max(8, ctypes.sizeof(ctypes.c_size_t))
        return (base_size + self._alignment - 1) & ~(self._alignment - 1)

    def _allocate_aligned(self, size: int) -> ctypes.Array:
        class AlignedArray(ctypes.Structure):
            _pack_ = self._alignment
            _fields_ = [("data", ctypes.c_char * size)]

        return AlignedArray()

    def _store_value(self, value: Union[int, bytes]) -> None:
        if isinstance(value, int):
            c_val = ctypes.c_uint64(value)
            ctypes.memmove(
                ctypes.addressof(self._value),
                ctypes.addressof(c_val),
                ctypes.sizeof(c_val),
            )
        else:
            value_bytes = memoryview(value).tobytes()
            ctypes.memmove(
                ctypes.addressof(self._value),
                value_bytes,
                min(len(value_bytes), self._calculate_aligned_size()),
            )

    def get_raw_pointer(self) -> int:
        return ctypes.addressof(self._value)

    def as_bytes(self) -> bytes:
        return bytes(self._value.data)

    def __int__(self) -> int:
        return int.from_bytes(self._value.data, sys.byteorder)

    def __repr__(self) -> str:
        return f"PyWord(value={self.as_bytes()!r}, alignment={self._alignment})"


@dataclass
class CPythonFrame:
    """Quantum-informed representation mapping to CPython's PyObject."""

    type_ptr: int
    value: Any
    type: type
    refcount: int = 1
    ttl: Optional[int] = None
    state: QuantumState = QuantumState.SUPERPOSITION
    quantum_byte: Optional['BYTE_WORD'] = None

    def __post_init__(self):
        self._birth_timestamp = time.time()
        self._state = self.state
        self._value = self.value
        if self.quantum_byte is None:
            value_hash = (
                hash(self.value) if hasattr(self.value, '__hash__') else id(self.value)
            )
            self.quantum_byte = BYTE_WORD(value_hash & 0xFF)
        if self.ttl is not None:
            self._ttl_expiration = self._birth_timestamp + self.ttl
        else:
            self._ttl_expiration = None
        if self.state == QuantumState.SUPERPOSITION:
            states = self.quantum_byte.value ^ 0b1111
            self._superposition = [self.value, states]
        else:
            self._superposition = None

    @property
    def refcount(self) -> int:
        return self._refcount

    @refcount.setter
    def refcount(self, value: int) -> None:
        self._refcount = value

    def collapse(self) -> Any:
        if self._state != QuantumState.COLLAPSED:
            if self._state == QuantumState.SUPERPOSITION and self._superposition:
                self._value = random.choice(self._superposition)
            self._state = QuantumState.COLLAPSED
        return self._value

    def observe(self) -> Any:
        if self._ttl_expiration and time.time() >= self._ttl_expiration:
            self.collapse()
        if self.state == QuantumState.SUPERPOSITION:
            entropy = math.log2(self.quantum_byte.value + 1) / 8
            if random.random() <= entropy:
                self.collapse()
        return self._value


# ==========================================================================
# AGENCY FRAMEWORK
# ==========================================================================


class Agency(ABC):
    """Abstract base class for agencies catalyzing state transformations."""

    def __init__(self, name: str):
        self.name = name
        self.agency_state: Dict[str, Any] = {}

    @abstractmethod
    def act(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def update_state(self, new_state: Dict[str, Any]):
        self.agency_state.update(new_state)


class Action:
    """Elementary process for conditional state transformation."""

    def __init__(
        self, input_conditions: Dict[str, Any], output_conditions: Dict[str, Any]
    ):
        self.input_conditions = input_conditions
        self.output_conditions = output_conditions

    def execute(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        if all(conditions.get(k) == v for k, v in self.input_conditions.items()):
            updated_conditions = conditions.copy()
            updated_conditions.update(self.output_conditions)
            return updated_conditions
        return conditions


class RelationalAgency(Agency):
    """Agency that catalyzes multiple actions dynamically."""

    def __init__(self, name: str):
        super().__init__(name)
        self.actions: List[Action] = []
        self.reaction_history: List[str] = []

    def add_action(self, action: Action):
        self.actions.append(action)

    def act(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        for action in self.actions:
            conditions = action.execute(conditions)
        self.reaction_history.append(f"Reactions performed: {conditions}")
        return conditions

    def get_reaction_history(self) -> List[str]:
        return self.reaction_history


class DynamicSystem:
    """System composed of agencies evolving state."""

    def __init__(self):
        self.agencies: List[RelationalAgency] = []

    def add_agency(self, agency: RelationalAgency):
        self.agencies.append(agency)

    def simulate(
        self, initial_conditions: Dict[str, Any], steps: int = 1
    ) -> Dict[str, Any]:
        state = initial_conditions.copy()
        for _ in range(steps):
            for agency in self.agencies:
                state = agency.act(state)
        return state


# ==========================================================================
# QUINEAN FRAMEWORK
# ==========================================================================

T = TypeVar('T')
V = TypeVar('V')
C = TypeVar('C')


class Oracle(Generic[T, V, C], ABC):
    """Abstract oracle for generating transformations."""

    def __init__(self):
        self.initialized = False
        self.first_input = None
        self.state = {}

    @abstractmethod
    def initialize_state(self, value: Any) -> Any:
        pass

    @abstractmethod
    def apply_morphism(self, value: Any) -> Any:
        pass

    def send(self, value: Any) -> Any:
        if not self.initialized:
            self.first_input = value
            self.initialized = True
            return self.initialize_state(value)
        return self.apply_morphism(value)


class QuineOracle(Oracle[T, V, C]):
    """Oracle that produces self-referential output."""

    def initialize_state(self, value: Any) -> Any:
        self.state['hash'] = hash(value) if hasattr(value, '__hash__') else id(value)
        return self.create_quine_output(value)

    def apply_morphism(self, value: Any) -> Any:
        return self.create_quine_output(value)

    def create_quine_output(self, value: Any) -> Any:
        return (value, f"QuineOracle(state={self.state['hash']})")


# ==========================================================================
# SERVER INTERACTION UTILITIES
# ==========================================================================


async def send_http_request(
    url: str, payload: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    """Send HTTP POST request to server."""
    try:
        import http.client

        logger.debug(f"Sending HTTP payload: {payload}")
        conn = http.client.HTTPConnection("localhost", 8000)
        headers = {"Content-Type": "application/json"}
        conn.request("POST", url, json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')
        logger.debug(f"Raw HTTP response: {data}")
        result = json.loads(data)
        logger.info(f"HTTP response: {result}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise
    except AppError as e:
        logger.error(f"Application error: {e.message} (code: {e.error_code})")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        raise
    finally:
        conn.close()


# ==========================================================================
# TEST SCENARIOS
# ==========================================================================


async def test_quinean_sdk():
    """Demonstrate the quinean SDK with server interactions."""
    # Initialize logger
    logger = logging.getLogger("quinean_sdk")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    # Initialize server
    config = AppConfig(
        root_dir=REPO_ROOT,
        log_level=logging.DEBUG,
        allowed_extensions={'.py', '.txt'},
        admin_users={'test_user'},
        enable_security=True,
    )
    server = JSONRPCServer(config)
    logger.info("Starting JSONRPCServer...")
    server_task = asyncio.create_task(server.run_forever())
    await asyncio.sleep(1)

    try:
        # Test 1: Agency-driven state transformation
        logger.info("Testing Agency framework")
        agency = RelationalAgency("file_manager")
        agency.add_action(Action({"file_exists": True}, {"action": "read_file"}))
        system = DynamicSystem()
        system.add_agency(agency)
        initial_state = {"file_exists": True, "path": "server/test.txt"}
        final_state = system.simulate(initial_state)
        logger.info(f"Agency simulation: {final_state}")

        # Test 2: Quantum manipulation of file content
        logger.info("Testing QuantumOperator with file content")
        test_file = REPO_ROOT / "server" / "test.txt"
        test_file.write_text("Quantum SDK")
        content = server.content_manager.get_content(test_file)
        pyword = PyWord(content.encode('utf-8'), alignment=WordAlignment.QWORD)
        logger.info(
            f"PyWord for test.txt: {pyword}, pointer: {pyword.get_raw_pointer()}"
        )

        # Create CPythonFrame for content
        frame = CPythonFrame(
            type_ptr=id(type(content)), value=content, type=type(content)
        )
        logger.info(f"CPythonFrame state: {frame.state}, value: {frame.observe()}")

        # Apply quantum operator
        hilbert = HilbertSpace(dimension=2)
        hadamard = [
            [MorphicComplex(1 / math.sqrt(2), 0), MorphicComplex(1 / math.sqrt(2), 0)],
            [MorphicComplex(1 / math.sqrt(2), 0), MorphicComplex(-1 / math.sqrt(2), 0)],
        ]
        operator = QuantumOperator(hilbert, hadamard)
        state_vector = [MorphicComplex(ord(c) / 255, 0) for c in content[:2]]
        transformed = operator.apply(state_vector)
        logger.info(f"Quantum-transformed content: {transformed}")

        # Test 3: Quinean self-replication
        logger.info("Testing QuineOracle")
        oracle = QuineOracle()
        quine_output = oracle.send(content)
        logger.info(f"Quine output: {quine_output}")

        # Test 4: Server interaction
        logger.info("Testing HTTP /generate endpoint")
        code_request = CodeRequest(instruct="print('Quinean SDK')", user_id="test_user")
        http_payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "execute_code",
            "params": code_request.to_dict(),
        }
        http_response = await send_http_request("/generate", http_payload, logger)
        logger.info(f"HTTP response: {http_response}")

        # Test 5: Security context
        logger.info("Testing SecurityContext")
        policy = AccessPolicy(
            level=AccessLevel.ADMIN,
            namespace_patterns=["*"],
            allowed_operations={"read", "write", "execute"},
        )
        security = SecurityContext("test_user", policy, logger)
        can_access = security.check_access("execute_code", "execute")
        logger.info(f"Can test_user execute_code? {can_access}")

        # Test 6: Performance metrics
        logger.info("Testing PerformanceMetrics")
        metrics = performance_metrics.get_report()
        logger.info(f"Performance metrics: {metrics}")

    except AppError as e:
        logger.error(
            f"Application error: {e.message} (code: {e.error_code}, status: {e.status_code})"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
    finally:
        logger.info("Stopping JSONRPCServer...")
        await server.stop()


# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    logger = logging.getLogger("quinean_sdk")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    try:
        asyncio.run(test_quinean_sdk())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
        asyncio.get_event_loop().run_until_complete(
            asyncio.get_event_loop().create_task(test_quinean_sdk()).result().stop()
        )
