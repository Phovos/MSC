#!/usr/bin/env -S uv run
# /* script
# requires-python = ">=3.12"
# dependencies = [
#     "uv==*.*",
# ]
# */
# <a href="https://github.com/Phovos/MSC">Morphological Source Code</a> © 2023 by PHOVOS:PHOVOS@outlook.com CC BY
from __future__ import annotations
# Optional dependency handling (also add to '/* script..' comment, just above)
try:
    import flask
    USE_FLASK = True
    # if we omit "flask==*.*", or any non-std lib from the '/* script..' comment, then this should always fail
    pass
except ImportError:
    USE_FLASK = False
    coreLSP = False
# Import standard library components
import os
import re
import sys
import math
import enum
import ctypes
import decimal
import platform
import subprocess
from array import array
from enum import Enum, IntEnum, IntFlag, auto
from typing import (
    Any, List, Union, Callable, TypeVar,
    Generic
)
from dataclasses import dataclass

# Check if we are in a managed environment
IN_UV_ENV = os.getenv("UV_VIRTUAL_ENV") is not None

# '--bootstrap' flag
def bootstrap():
    """Attempt to install 'uv' and rerun the script in a managed environment."""
    print("Bootstrapping: Checking for 'uv' package manager...")
    try:
        subprocess.run(["uv", "--version"], check=True,
                       stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: 'uv' is not installed. Please install it manually.")
        sys.exit(1)

    print("Re-executing script with 'uv run'...")
    os.execvp("uv", ["uv", "run", sys.executable] + sys.argv)

if "--bootstrap" in sys.argv:  # Handle manual opt-in for bootstrapping
    bootstrap()
# system and platform code
class PlatformFactory:  # Platform abstraction
    """Detect and return the current platform."""
    @staticmethod
    def get_platform():
        if os.name == 'nt':
            return "windows"
        elif os.name == 'posix':
            return "posix"
        raise NotImplementedError("Unsupported platform")

    @staticmethod
    def create_platform_instance():
        plat = PlatformFactory.get_platform()
        return WindowsPlatform() if plat == "windows" else LinuxPlatform()

class PlatformInterface:
    """Abstract base for platform-specific implementations."""

    def load_c_library(self):
        raise NotImplementedError()

class WindowsPlatform(PlatformInterface):
    def load_c_library(self):
        try:
            return ctypes.CDLL("msvcrt.dll")
        except OSError:
            return None

class LinuxPlatform(PlatformInterface):
    def load_c_library(self):
        try:
            return ctypes.CDLL("libc.so.6")
        except OSError:
            return None

class ProcessorFeatures(IntFlag):
    BASIC = auto()
    SSE = auto()
    AVX = auto()
    AVX2 = auto()
    AVX512 = auto()
    NEON = auto()
    SVE = auto()
    RVV = auto()  # RISC-V Vector Extensions
    AMX = auto()  # Advanced Matrix Extensions

    @classmethod
    def detect_features(cls) -> 'ProcessorFeatures':
        features = cls.BASIC
        try:
            if platform.machine().lower() in ('x86_64', 'amd64', 'x86', 'i386'):
                if sys.platform == 'win32':
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                         r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
                    identifier = winreg.QueryValueEx(
                        key, 'ProcessorNameString')[0]
                else:
                    with open('/proc/cpuinfo') as f:
                        identifier = next(line.split(
                            ':')[1] for line in f if 'model name' in line)
                identifier = identifier.lower()
                if 'avx512' in identifier:
                    features |= cls.AVX512
                if 'avx2' in identifier:
                    features |= cls.AVX2
                if 'avx' in identifier:
                    features |= cls.AVX
                if 'sse' in identifier:
                    features |= cls.SSE
            elif platform.machine().lower().startswith('arm'):
                if sys.platform == 'darwin':  # Apple Silicon
                    features |= cls.NEON
                else:
                    with open('/proc/cpuinfo') as f:
                        content = f.read().lower()
                        if 'neon' in content:
                            features |= cls.NEON
                        if 'sve' in content:
                            features |= cls.SVE
        except Exception:
            pass
        return features

@dataclass
class RegisterSet:
    gp_registers: int
    vector_registers: int
    register_width: int
    vector_width: int

    @classmethod
    def detect_current(cls) -> 'RegisterSet':
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64'):
            return cls(gp_registers=16, vector_registers=32, register_width=64, vector_width=512)
        elif machine.startswith('arm64'):
            return cls(gp_registers=31, vector_registers=32, register_width=64, vector_width=128)
        else:
            return cls(gp_registers=8, vector_registers=8, register_width=32, vector_width=128)

class ProcessorArchitecture(IntEnum):
    X86 = auto()
    X86_64 = auto()
    ARM32 = auto()
    ARM64 = auto()
    RISCV32 = auto()
    RISCV64 = auto()

    @classmethod
    def current(cls) -> 'ProcessorArchitecture':
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64'):
            return cls.X86_64
        elif machine in ('x86', 'i386', 'i686'):
            return cls.X86
        elif machine.startswith('arm'):
            return cls.ARM64 if sys.maxsize > 2**32 else cls.ARM32
        elif machine.startswith('riscv'):
            return cls.RISCV64 if sys.maxsize > 2**32 else cls.RISCV32
        raise ValueError(f"Unsupported architecture: {machine}")
@dataclass
class MemoryModel:
    """Maps linear-virtual address space per the OS to Frames+Lifetimes+Arenas (linear allocator).."""
    ptr_size: int = ctypes.sizeof(ctypes.c_void_p)
    word_size: int = ctypes.sizeof(ctypes.c_size_t)
    cache_line_size: int = 64
    page_size: int = 4096

    @classmethod
    def get_system_info(cls) -> 'MemoryModel':
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size') as f:
                cache_line_size = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            cache_line_size = 64
        return cls(
            ptr_size=ctypes.sizeof(ctypes.c_void_p),
            word_size=ctypes.sizeof(ctypes.c_size_t),
            cache_line_size=cache_line_size,
            page_size=cls.page_size
        )
class WordAlignment(IntEnum):
    UNALIGNED = 1
    WORD = 2
    DWORD = 4
    QWORD = 8
    CACHE_LINE = 64
    PAGE = 4096
class WordSize(enum.IntEnum):
    # Utilization of anisotropy about (0) and the inflation of state space makes WordSize a core-scalar
    BYTE = 1     # 8-bit 'consumer hardware' = (1); Arbitrarily scaled: ryzen5 & NVIDIA RTX
    SHORT = 2    # 16-bit
    INT = 4      # 32-bit
    LONG = 8     # 64-bit; does not refer to the x86 x64 register(s)!
# # pymachine.py 
"""
CanonTM: Tuple(Q,T,B,ε,𝛿.q0,F)
Q: finite set of states
T: tape alphabet (symbols)
B: blank symbol (all cells are B, except input alphabet, initially)
ε: the input alphabet (symbols)
𝛿: transition function which maps 'Q x T -> Q x T x {L,R}'
q0: the initial state
F: the set of final states; if any state of F is reached: input string accepted
---
MSC(CanonTM):
T: 'Type'/structure
V: 'Value'
C: 'Compute'/control
R: 'Result(s)'
"""
# Advanced static typing
decimal.getcontext().prec = 28  # Set decimal precision at runtime
T = TypeVar('T')  # Type structure
V = TypeVar('V')  # Value space
C = TypeVar('C')  # 'Computation'/control type ['Captaincy']
R = TypeVar('R')  # Result type
BYTE = TypeVar("BYTE", bound="ByteWord")
T_co = TypeVar('T_co', covariant=True)  # Covariant Type structure
V_co = TypeVar('V_co', covariant=True)  # Covariant Value space
C_co = TypeVar('C_co', bound=Callable[..., Any], covariant=True)  # Covariant Control space
T_anti = TypeVar('T_anti', contravariant=True)  # Contravariant Type structure
V_anti = TypeVar('V_anti', contravariant=True)  # Contravariant Value space
C_anti = TypeVar('C_anti', bound=Callable[..., Any], contravariant=True)  # Contravariant Computation space
# Operator phenomenology, etc.
"""Core Operators:

Composition (@): Sequential application of operations
Tensor Product (*): Parallel combination of operations
Direct Sum (+): Alternative pathways of computation
Adjoint (†): Reversal/dual of operations

Algebraic Properties:

Associativity: (A @ B) @ C = A @ (B @ C)
Distributivity: A * (B + C) = (A * B) + (A * C)
Adjoint rules: (A @ B)† = B† @ A†"""
class OperatorType(Enum):
    """Fundamental operation types in our computational 'universe', referring explicitly to the universal-set [], and given the null set (a 00000000 ByteWord) as 'glue' (insofar as sheafification, groups, topos etc). The 'universe' of runtime, the applied set, is strictly-bounded and inertia-local, no relativistic effects outside of the 'relativistic effects' of morphological derivation (or time-like integration)* with respect to the cross-product of two cartesian coordinates in super position; a 'Born Rule'-type ontological scaffolding."""
    COMPOSITION = auto()   # Function composition (f >> g)
    TENSOR      = auto()   # Tensor product (⊗)
    DIRECT_SUM  = auto()   # Direct sum (⊕)
    OUTER       = auto()   # Outer product (|ψ⟩⟨φ|)
    ADJOINT     = auto()   # Hermitian adjoint (†)
    MEASUREMENT = auto()   # Quantum measurement (⟨M|ψ⟩)

class QuantumState(enum.Enum):
    """
    Quantum states for chiral quines, inspired by Wigner's Friend and Barandes' stochastic mechanics, amongst others.
    Maps to Morphology (MARKOVIAN, NON_MARKOVIAN) for non-Markovian tape evolution.
    Each state represents a ByteWord's epistemic role in the T/V/C toople:
    - Type: Tape (poset/frozenset) evolves via chiral tx (-1, 0, 1).
    - Value: Semantic vector (posit) tracks position with chiral updates.
    - Code: QOperator evolves ByteWords as quantum-like states.
    """
    SUPERPOSITION = 1  # Handle-only state, like a MARKOVIAN (-1) ByteWord with chiral tx (-1), history-dependent.
    ENTANGLED = 2      # Referenced but not materialized, like NON_MARKOVIAN (math.e), reversible with energy cost.
    COLLAPSED = 4      # Materialized state, like a stable quine (SmallTalk object), executable after measurement.
    DECOHERENT = 8     # Garbage-collected state, reversible only by re-running with new chiral tape (thermodynamic cost).

    def transition(self, operator: 'OperatorType') -> 'QuantumState':
        """
        Transition between quantum states based on OperatorType.
        - MEASUREMENT collapses SUPERPOSITION/ENTANGLED to COLLAPSED.
        - ADJOINT reverses COLLAPSED to ENTANGLED with energy cost.
        - DECOHERENT stays unless reset (re-run).
        """
        if operator == OperatorType.MEASUREMENT:
            if self in (QuantumState.SUPERPOSITION, QuantumState.ENTANGLED):
                return QuantumState.COLLAPSED
        elif operator == OperatorType.ADJOINT and self == QuantumState.COLLAPSED:
            return QuantumState.ENTANGLED
        elif self == QuantumState.DECOHERENT and operator == OperatorType.COMPOSITION:
            return QuantumState.SUPERPOSITION
        return self

# Semantic classes
_ANSI_RE = re.compile(
    r"""(
        [A-Za-z_][A-Za-z0-9_]* |     # ident
        \d+\.\d+ |                   # float-like (still treated as bytes; no FP math)
        \d+ |                        # int-like
        \s+ |                        # whitespace
        .                            # single char fallback
    )""",
    re.VERBOSE,
)

def _to_latin1_bytes(s: str) -> bytes:
    """Strict Latin-1 to guarantee 0..255 domain. Raises on non-ANSI."""
    return s.encode("latin-1", errors="strict")

def tokenize_ansi(s: str) -> List[bytes]:
    """Split into byte-tokens while preserving whitespace and punctuation."""
    tokens: List[bytes] = []
    for m in _ANSI_RE.finditer(s):
        tok = m.group(0)
        tokens.append(_to_latin1_bytes(tok))
    return tokens

class PyWord(Generic[T]):
    """
    [[PyWord]] represents a word-sized value optimized for CPython.
    It manages alignment according to the system's memory model and
    provides conversion between Python and C types.
    """
    __slots__ = ('_value', '_alignment', '_arch', '_mem_model')

    def __init__(self,
                 value: Union[int, bytes, bytearray, array.array],
                 alignment: WordAlignment = WordAlignment.WORD):
        self._mem_model = MemoryModel.get_system_info()
        self._arch = ProcessorArchitecture.current()
        self._alignment = alignment
        aligned_size = self._calculate_aligned_size()
        self._value = self._allocate_aligned(aligned_size)
        self._store_value(value)

    def _calculate_aligned_size(self) -> int:
        base_size = max(self._mem_model.word_size,
                        ctypes.sizeof(ctypes.c_size_t))
        return (base_size + self._alignment - 1) & ~(self._alignment - 1)

    def _allocate_aligned(self, size: int) -> ctypes.Array:
        class AlignedArray(ctypes.Structure):
            _pack_ = self._alignment
            _fields_ = [("data", ctypes.c_char * size)]
        return AlignedArray()

    def _store_value(self, value: Union[int, bytes, bytearray, array.array]) -> None:
        if isinstance(value, int):
            if self._arch in (ProcessorArchitecture.X86_64, ProcessorArchitecture.ARM64, ProcessorArchitecture.RISCV64):
                c_val = ctypes.c_uint64(value)
            else:
                c_val = ctypes.c_uint32(value)
            ctypes.memmove(ctypes.addressof(self._value),
                           ctypes.addressof(c_val), ctypes.sizeof(c_val))
        else:
            value_bytes = memoryview(value).tobytes()
            ctypes.memmove(ctypes.addressof(self._value),
                           value_bytes, len(value_bytes))

    def get_raw_pointer(self) -> int:
        return ctypes.addressof(self._value)

    def as_memoryview(self) -> memoryview:
        return memoryview(self._value)

    def as_buffer(self) -> ctypes.Array:
        return (ctypes.c_char * self._calculate_aligned_size()).from_buffer(self._value)

    @property
    def alignment(self) -> int:
        return self._alignment

    @property
    def architecture(self) -> ProcessorArchitecture:
        return self._arch

    def __int__(self) -> int:
        if isinstance(self._value, ctypes.Array):
            return int.from_bytes(self._value.data, sys.byteorder)
        return int.from_bytes(self._value.tobytes(), sys.byteorder)

    def __bytes__(self) -> bytes:
        if isinstance(self._value, ctypes.Array):
            return bytes(self._value.data)
        return self._value.tobytes()

class PyWordCache:
    """LRU Cache for [[PyWord]] objects to minimize allocations."""

# Ontology-types
class Morphology(enum.Enum):
    """
    Represents the (thermo) dynamism and floor morphic state of a ByteWord
    
    C = 0: Floor morphic state (stable, low-energy)
    C = 1: Dynamic or high-energy state

    - DYNAMIC (1): Other icons CAN point to this icon
    - MORPHIC (0): Other icons CANNOT point to this icon
    
    This ontology maps to intensive & extensive thermodynamic character. The 'location' of this character is about the boundary (integral and non-relativistic), with observables within the bulk (quantized, with uncertainty, requiring an Einsteinian observer).

    - MARKOVIAN (-1): History-dependant
    - NON_MARKOVIAN (math.e): "Fully-quantized" null-vector
    
    Implementation-not: (-1) & (math.e) are synonyms of (0) & (1), respectivly, in certain contexts such as during the creation of homogenous coordinate-'tooples', appearing as 0, 1, or a power of 2 (that needs to then divide the whole-column by it's total, as-many times as-necessary, until the new-homogenous row is only (0) and/or (1)). This mirrors the 'duputization cascade' and QuineicSaddle historisis function/Kronecker-Dirac delta (象 in the sense of phenomenological identity).
    """
    MORPHIC = 0      # Stable, low-energy state
    DYNAMIC = 1      # High-energy, potentially transformative state
    # Time-like but not relativistic Noetherian/Machian bulk-orchestration
    MARKOVIAN = -1    # Forward-evolving, irreversible
    NON_MARKOVIAN = math.e  # Reversible, with memory

class WindingMode(enum.Enum):
    BINARY = "binary"
    TERNARY = "ternary"

GLOBAL_WINDING_MODE = WindingMode.TERNARY


@dataclass(frozen=True)
class WindingPair:
    w1: int
    w2: int
    mode: WindingMode = GLOBAL_WINDING_MODE

    def __post_init__(self):
        if self.mode == WindingMode.BINARY:
            if self.w1 not in (0, 1) or self.w2 not in (0, 1):
                raise ValueError("Binary winding must be 0 or 1")
        else:
            if self.w1 not in (-1, 0, 1) or self.w2 not in (-1, 0, 1):
                raise ValueError("Ternary winding must be -1, 0, 1")

    def tx(self, a: int, b: int) -> int:
        if a == b: return 0
        if a == 0: return b
        if b == 0: return a
        return 0

    def apply_val(self, mask: "WindingPair") -> "WindingPair":
        if self.mode != mask.mode:
            raise ValueError("Mode mismatch")
        if self.mode == WindingMode.BINARY:
            return WindingPair(self.w1 ^ mask.w1, self.w2 ^ mask.w2, mode=self.mode)
        return WindingPair(
            self.w1 if mask.w1 == -1 else self.tx(self.w1, mask.w1),
            self.w2 if mask.w2 == -1 else self.tx(self.w2, mask.w2),
            mode=self.mode
        )

    def to_state_index(self) -> int:
        if self.mode == WindingMode.BINARY:
            return (self.w1 << 1) | self.w2
        idx_map = {-1: 0, 0: 1, 1: 2}
        return (idx_map[self.w1] * 3) + idx_map[self.w2]

@dataclass
class ByteWord:
    raw: int

    @classmethod
    def null(cls) -> "ByteWord":
        return cls(0)

    def __post_init__(self):
        if not (0 <= self.raw <= 0xFF):
            raise ValueError("raw must be 0..255")

    @property
    def captain(self) -> bool:
        return bool((self.raw >> 7) & 1)

    @property
    def value_field(self) -> int:
        return (self.raw >> 4) & 0x07

    @property
    def type_field(self) -> int:
        return self.raw & 0x0F

    @property
    def is_null(self) -> bool:
        return (not self.captain) and (self.type_field == 0)

    @property
    def winding(self) -> WindingPair:
        if GLOBAL_WINDING_MODE == WindingMode.BINARY:
            w1 = (self.type_field >> 1) & 0x01
            w2 = (self.type_field >> 0) & 0x01
        else:
            tbl = [-1, 0, 1, 0]
            w1 = tbl[(self.type_field >> 2) & 0x03]
            w2 = tbl[self.type_field & 0x03]
        return WindingPair(w1, w2, mode=GLOBAL_WINDING_MODE)

    def apply_unitary(self, operator: "ByteWord") -> "ByteWord":
        new_w = self.winding.apply_val(operator.winding)
        if GLOBAL_WINDING_MODE == WindingMode.BINARY:
            new_type = (new_w.w1 << 1) | new_w.w2
        else:
            inv = {-1: 0, 0: 1, 1: 2}
            new_type = ((inv[new_w.w1] & 0x03) << 2) | (inv[new_w.w2] & 0x03)
        new_raw = (self.raw & 0xF0) | (new_type & 0x0F)
        return ByteWord(new_raw)
