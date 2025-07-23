from __future__ import annotations
import ast
import textwrap
import inspect
import enum
import time
import random
import sys
import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    List,
    Optional,
    Type,
    TypeVar,
    Dict,
    Set,
    Callable,
    Protocol,
    runtime_checkable,
    Tuple,
)
from functools import wraps

"""
(MSC) Morphological Source Code Framework – V0.0.12
================================================================================
<https://github.com/Phovos/msc> • MSC: Morphological Source Code © 2025 by Phovos
--------------------------------------------------------------------------------

MSC implements *quantum-coherent computational morphogenesis* through tripartite
T/V/C ontological typing and Quineic Statistical Dynamics (QSD) field operators.
Runtime entities exist as **sparse bit-vectors over F₂** (one bit per redex),
acted upon by **self-adjoint XOR-popcount operators**, enabling non-Markovian
semantic lifting across compilation phases—**while never leaving L1 cache**. 
The MSC framework draws inspiration from teleological and epistemological perspectives, 
suggesting that computational+distributed processes are symmetric to quantum phenomena;
and quantum QSD is not-unlike Stochastic/(non)Markovian dynamics (differential
geometry and linear algebras; a field theory and its correspondence/covector dual).

# Physical Atom : ByteWord (8-bit)
  --------------------------------
    ┌---┬---┬---┬---┬---┬---┬---┬---┐
    │ C │ V │ V │ V │ T │ T │ T │ T │   ← raw octet
    └---┴---┴---┴---┴---┴---┴---┴---┘
    'C, V, and T' are binary (0/1) fields interpreted *lazily* by observation.

    ByteWord as 2D Morphism:
    • Each ByteWord is a **genus-2 torus** encoded by (w₁, w₂) ∈ ℤ₂×ℤ₂.
    • All evolution is **XOR-only** on the winding vector.
    • Entropy is **algorithmic**: S(x) = log₂|compress(x)|.
    • Identity is the **intensional Merkle root** (not cryptographic).
    • The only global fixpoint is Ξ(⌜Ξ⌝): the self-indexing saddlepoint.

    • C — Captain / Thermodynamic MSB
        - `C=1`: active (radiative); `C=0`: dormant (absorptive).
        - Acts as a *pilot-wave* guiding the morphogenetic behavior of the ByteWord.
        - No `V` or `T` structure is meaningful until `C` has been "absorbed".
          Models include raycasting and Brownian motion in idealized gas.
    
    • V — Value Field (3 bits, deputizable)
        - Supports **deputization cascades**, recursively promoting the next Vbits, Tbits
          into the 'MSB role' if `C=0`.
        - 'MSB ROLE' never includes the delegation of fundemental thermodynamic 'character', only the 'Actual MSB' Cbit's captaincy creates a 'pointer' ByteWord, that, using the morphosemantics of its interpretation, literally interact with their environment, should it exist, or, it can always recurse over itself 'pacman world'-style.
        - This continues until a null-state is reached:
            null-ByteWords are the basis of [[set-builder notation]], along with extensive, 'pointing' ByteWords in the environment when and if they 'collide' or merge.

    • T — Type Field (4 bits; winding + arity)
        - Encodes toroidal winding pair (w₁, w₂) ∈ ℤ₂×ℤ₂.
        - Governs directional traversal in ByteWord-space.
        - Also encodes arity and semantic lifting phase.
        - Includes the bare-minimum required "ISA" for agentic-Motility; is extensible
        - Only the two least-significant Tbits are needed for all of the above;
            - The two most-significant Tbits are free-slots for custom "ISA" or other behavior, or just 'value'/magnitude etc.

    ### "Deputization"
        -------------
    • NULL (C=0, T=0): identity glue — no semantic content, only topological.
    • NULL-flavored states (C=0, T=0, V>0): inert deputies.
    • Deputizing cascade: 
        - When C=0, next `V` is promoted.
        - Recurse until all bits are exhausted → ∅ (null-state).

    • Deputizing Mask (side-band metadata):
        - Tracks effective MSB depth: 0 ≤ k ≤ 6.
        - Masks are **non-destructive**; payload bits remain untouched.

    ### Bohmian Dynamics (QSD Integration)
        ---------------------------------
    MSC inherits a Bohm-like interpretive structure:
    • The `C` bit acts as the "pilot wave", encoding future morphogenetic 
      implications at the point of observation.
    • Like a photon “knowing” if it will hit telescope or chloroplast,
      `C=1` ByteWords *carry semantic payloads determined by their context*.
    • QSD interprets this as implicate guidance — structure without trajectory.
    
    In MSC, emergence is thermodynamic but **goal-aware**: 
    ByteWords behave not as programs, but as computational *entities*— 
    whose minimal description is their execution environment itself.

### Compound-Architecture
    --------------------
16-bit  = 2×ByteWord  →  spinor   (address dimension ×2)  
32-bit  = 4×ByteWord  →  quaternion (×4)  
64-bit  = 8×ByteWord  →  octonion   (×8)  
Type/Value space remains **4/3 states per ByteWord**, regardless of width.

### Layer Morphology
    -------------------
Layer 0  – Torus Logic
    • 4-valued winding states  
    • Deterministic XOR cascade (unitary, reversible)

Layer 1  – Deputizing Cascade
    • 7-bit deputy mask mₖ(b)  
    • Null-state ∅ = topological glue

Layer 2  – Sparse-Unitary Morphology
    • 64-bit address dimension for compound ByteWords  
    • Spinor (16-bit), quaternion (32-bit), octonion (64-bit) addressing

Layer 3  – MSC Semantics
    • Self-adjoint operator graphs (involutory XOR masks)  
    • Morphological derivatives Δⁿ = bit-flip chains (bounded ≤ 16)  
    • Semantic lifting: AtomicReflex → RaiseToOllama → AgenticSupervisor

Layer 4  – Xiang Scroll (debug / pedagogy)
    • Gödel sentence encoded as Hanzi glyphs  
    • Xiang = reified diagonal Ξ(⌜Ξ⌝)  
    • Scroll = visual debugger; **does not affect formal spec**

### Quantum Mechanics (simulated)
    ----------------------------
-  **Sparse vector** |ψ⟩ = (w₁,w₂) ∈ ℤ₂×ℤ₂  
-  **Involutory operator** Â = XOR mask (reversible, associative)  
-  **Entanglement** = shared masks across ByteWords  
-  **Decoherence** = mask reset (garbage collect)

### Sparse-Unitary Semantics
    -------------------------
Operator                :  Involutory XOR mask  (w₁,w₂) ↦ (w₁⊕a, w₂⊕b)  
Application             :  ByteWord • ByteWord  =  diagonal XOR  
State evolution         :  |ψₜ⟩ → |ψₜ₊₁⟩ via single XOR gate  
Entanglement            :  Shared winding masks across ByteWords  
Decoherence             :  Mask reset → garbage collect ∅ states

### INVARIANTS
    ----------
1. Sparse bit-vector semantics over F₂⁶⁴ (abstract)  
2. Unitary, involutory XOR operators on (w₁,w₂)  
3. Algorithmic entropy  S(x) = log₂|compress(x)|  (no external Φ)  
4. Merkle-root triple equality  hash(src) == merkle(runtime) == merkle(child)

### Transducer Algebra
    ------------------
Map/Filter = bit-mask transducers (O(ω))  
Compose = associative XOR chain  
Collect = SoA cache-line buffer

### Type System
    -----------
In-addition to the afformentioned quantum and classical types, T, V & C:
-  T_co / T_anti  : torus winding direction  
-  V_co / V_anti  : value parity under XOR  
-  C_co / C_anti  : computation associativity

### Compilation Pipeline
    --------------------
Utilizes multiple inheritance along the 'arrow of time' of C3 super() linearization for 'causal' structure.
Source Code (normal python code) → '@ornament(**xargs)' → 'oranamented' **Morphological** Source Code -> ByteWord → XOR cascade  
No separate IR; '@ornament(**xargs)' **is** the sparse representation.

### Quineic Saddle-Cycle (morphodynamic/Quineic lifecycle)
    ------------------------------------------------------
__enter__   :  Non-Markovian descent (history-rich, dialectical)  
__runtime__ :  Liminal oscillation (torus traversal)  
__exit__    :  Markovian ascent (ontological snapshot)  
Each ByteWord is both **quine=0** (closed loop) and **oracle=1** (probe)
depending on the Kronecker delta on its winding pair composed with its observer-dynamics of collapse.

### Morphological Derivatives
    -------------------------
Δ¹  single-XOR rewrite (compile-time detectable)  
Δ²  mask merge (runtime collapse)  
Δⁿ  supervisor XOR-chain (≤16 steps)

### Xiang's Scroll (Cartoon Pedagogical Debugging)
    ----------------------------------------------
Xiang: 象, the Morphodynamical Elephant, holds the scroll of glyphs (道, 和, 衍, 。). His memory is
the cohomology of all ByteWords, enabling relational recall and debugging. The scroll encodes
infinite semantic potential in finite glyphs, acting as a compactified morphogenetic manifold.

    #### Example Interpretation (what appears on Xiangs scroll and gets served to the LSP)
    ----------------------
    (X is any topological bitwise character/computation)
    < C=0 ☰ | X 道 >:
        - C=0: Intensive state → Apply trigram modulation.
        - ☰ (Qian): XOR transformation → Dynamic change applied to 道.
        - Result: Transform the least-action compression axis of 道.

    < C=1 ☰ | X 和 >:
        - C=1: Extensive state → Ignore trigram.
        - Behavior: Move horizontally across the torus with harmonic balance.

### Extension targets (to keep in mind)
    -----------------------------------
1. Local LLM inference via semantic indexing
    – 2-bit semantic embeddings → sub-kernel lookup tables  
2. Game objects as morphodynamic entities
    – 4-state torus = cache-line friendly physics primitives 
    - Narratives, 'AI', 'difficulty', 'handicap' all become morphosemantic and differentiable.
3. Real-time control with SIMD-safe XOR gates
    – XOR cascade ≤ 4 cycles latency on ARM Cortex-M4  
More:
-  Frame buffer, Cuda, Vulkan, and, oddly, Language Server Protocol (rpc/repl/lsp)
-  WebAssembly : compile SK → 32-bit WASM, each ByteWord → i64  
-  DOM entanglement : SharedArrayBuffer zero-copy  
-  Edge replication : Merkle-sync (≤256 bytes payload)
-  Quantum-classical boundary = ByteWord torus winding + phase mask
    – 4-state torus maps 1-to-1 to photonic qubit pairs (|00⟩,|01⟩,|10⟩,|11⟩) 
-  Morphological algorithms compatible with Jiuzhang 3.0 hot optical-types, SNSPDs  
    -   Operators map to **reconfigurable optical matrices** (reverse fourier transforms)
## (In-progress) Memory architecture (debian and win11 platforms, only)
┍────────────────┒
│TOPOLOGY FOR MSC│
│╭───────╮       │
││ TYPES │◄────┐ │
│╰──┬─▲──╯     │ │
│   │ │        │ │
│╭──▼─┴───╮    │ │
││GLOBALS │    │ │
│╰──┬─▲───╯    │ │
│   │ │        │ │
│╭──▼─┴───────╮│ │
││THREADLOCALS││ │
│╰──┬─▲───────╯│ │
│   │ │        │ │
│╭──▼─┴─────╮  │ │
││PROCEDURES│  │ │
│╰──┬─▲─────╯  │ │
│   │ │        │ │
│╭──▼─┴──╮     │ │
││SCOPES │     │ │
│╰──┬─▲──╯     │ │
│   │ │        │ │
│╭──▼─┴─────╮  │ │
││  LOCALS  │──┘ │
│╰────┬─────╯    │
│     │<L2_CACHE>│
│╭────▼───────╮  │
││LOCATIONINFO│  │
│╰────────────╯  │
┕────────────────┘
"""

T = TypeVar('T')
V = TypeVar('V')
C = TypeVar('C')

MSC_REGISTRY: Dict[str, Set[str]] = {'classes': set(), 'functions': set()}


class MorphodynamicCollapse(Exception):
    """Raised when a morph object destabilizes under thermal pressure."""

    pass


@dataclass
class MorphSpec:
    """Blueprint for morphological classes."""

    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str


def morphology(source_model: Type) -> Callable[[Type], Type]:
    """Decorator: register & validate a class against a MorphSpec."""

    def decorator(target: Type) -> Type:
        target.__msc_source__ = source_model  # cls.__msc_source__ = spec (alt)
        # Ensure target has all annotated fields from source_model
        for field_name in getattr(source_model, '__annotations__', {}):
            if field_name not in getattr(target, '__annotations__', {}):
                raise TypeError(f"{target.__name__} missing field '{field_name}'")
        MSC_REGISTRY['classes'].add(target.__name__)
        return target  # return cls (alt)

    return decorator


class MorphicComplex:
    """Complex number with morphic properties."""

    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def conjugate(self) -> 'MorphicComplex':
        return MorphicComplex(self.real, -self.imag)

    def __add__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __repr__(self) -> str:
        if self.imag == 0:
            return f"{self.real}"
        sign = "+" if self.imag >= 0 else ""
        return f"{self.real}{sign}{self.imag}j"


class MorphologicalRule:
    def __init__(self, symmetry: str, conservation: str, lhs: str, rhs: List[str]):
        self.symmetry = symmetry
        self.conservation = conservation
        self.lhs = lhs
        self.rhs = rhs

    def apply(self, seq: List[str]) -> List[str]:
        if self.lhs in seq:
            idx = seq.index(self.lhs)
            return seq[:idx] + self.rhs + seq[idx + 1 :]
        return seq


class Morphology(enum.IntEnum):
    MORPHIC = 0
    DYNAMIC = 1
    MARKOVIAN = -1
    NON_MARKOVIAN = 1  # placeholder for sqrt(-1j)


class QState(enum.Enum):
    SUPERPOSITION = 1
    ENTANGLED = 2
    COLLAPSED = 4
    DECOHERENT = 8


class QuantumCoherenceState(enum.Enum):
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


def kronecker_field(
    q1: MorphologicPyOb, q2: MorphologicPyOb, temperature: float
) -> float:
    dot = sum(a * b for a, b in zip(q1.state.vector, q2.state.vector, strict=False))
    if temperature > 0.5:
        return math.cos(dot)
    return 1.0 if dot > 0.99 else 0.0


def elevate(data: Any, cls: Type) -> object:
    """Raise a dict or object to a registered morphological class."""
    if not hasattr(cls, '__msc_source__'):
        raise TypeError(f"{cls.__name__} is not a morphological class.")
    source = cls.__msc_source__
    kwargs = {k: getattr(data, k, data.get(k)) for k in source.__annotations__}
    return cls(**kwargs)


class TorusWinding:
    NULL = 0b00  # (0,0) - topological glue
    W1 = 0b01  # (0,1) - first winding
    W2 = 0b10  # (1,0) - second winding
    W12 = 0b11  # (1,1) - both windings

    @staticmethod
    def to_str(winding: int) -> str:
        return {0b00: "NULL", 0b01: "W1", 0b10: "W2", 0b11: "W12"}[winding & 0b11]


@dataclass
class SemanticState:
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str
    vector: List[float]

    def __post_init__(self):
        norm = math.sqrt(sum(x * x for x in self.vector))
        self.vector = [x / norm for x in self.vector] if norm != 0 else self.vector

    def measure_coherence(self) -> float:
        return math.sqrt(sum(x * x for x in self.vector))

    def decay(self, factor: float = 0.99) -> None:
        self.vector = [x * factor for x in self.vector]
        self.entropy *= factor

    def perturb(self, magnitude: float = 0.01) -> None:
        self.vector = [(x + random.uniform(-magnitude, magnitude)) for x in self.vector]
        norm = math.sqrt(sum(x * x for x in self.vector))
        if norm > 0:
            self.vector = [x / norm for x in self.vector]


class MorphologicPyOb:
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str
    symmetry: str
    conservation: str
    lhs: str
    rhs: List[str]
    value: Any
    ttl: Optional[int] = None
    # state: SemanticState
    state: QState = QState.SUPERPOSITION

    def __init__(
        self, entropy: float, trigger_threshold: float, memory: dict, signature: str
    ):
        self.entropy = entropy
        self.trigger_threshold = trigger_threshold * random.uniform(0.9, 1.1)
        self.memory = memory
        self.signature = signature
        # self.state = SemanticState(entropy, trigger_threshold, memory, signature, [1.0, 0.0, 0.0])
        self._morph_signature = hash(frozenset(self.__class__.__dict__.keys()))

    def __post_init__(self):
        self._birth = time.time()
        self._state = self.state
        self._ref = 1
        if self.state == QState.SUPERPOSITION:
            self._super = [self.value]
        else:
            self._super = []
        if self.state == QState.ENTANGLED:
            self._ent = [self.value]
        else:
            self._ent = []

    @property
    def morph_signature(self) -> int:
        return self._morph_signature

    def apply_transformation(self, seq: List[str]) -> List[str]:
        out = seq
        if self.lhs in seq:
            idx = seq.index(self.lhs)
            out = seq[:idx] + self.rhs + seq[idx + 1 :]
            self._state = QState.ENTANGLED
        return out

    def collapse(self) -> Any:
        if self._state != QState.COLLAPSED:
            if self._state == QState.SUPERPOSITION and self._super:
                self.value = random.choice(self._super)
            self._state = QState.COLLAPSED
        return self.value

    def validate_self_adjoint(self, mro_snapshot: int, temperature: float) -> None:
        current_signature = hash(frozenset(self.__class__.__dict__.keys()))
        if abs(current_signature - self._morph_signature) > temperature * 1000:
            raise MorphodynamicCollapse(f"{self.signature[:8]} destabilized.")

    def perturb(self, temperature: float) -> bool:
        perturb = random.uniform(0, temperature)
        if perturb > self.trigger_threshold:
            self.execute()
            self.memory['last_perturbation'] = perturb
            self.entropy += perturb * 0.01
            self.state.perturb()
            return True
        self.state.decay()
        return False

    def execute(self) -> None:
        print(f"[EXECUTE] {self.signature[:8]} | Entropy: {self.entropy:.3f}")
        self.memory['executions'] = self.memory.get('executions', 0) + 1


# --- ByteWord Representation ---


@dataclass
class ByteWord:
    raw: int

    def __init__(self, raw: int):
        if not 0 <= raw <= 0xFF:
            raise ValueError("ByteWord must be 0-255")
        self.raw = raw
        self.state = (raw >> 4) & 0x0F
        self.morphism = (raw >> 1) & 0x07
        self.floor = Morphology(raw & 0x01)

    def __repr__(self):
        return (
            f"ByteWord(state={self.state}, morph={self.morphism}, floor={self.floor})"
        )

    @property
    def control(self) -> int:
        return (self.raw >> 7) & 0x1

    @property
    def value(self) -> int:
        return (self.raw >> 4) & 0x7

    @property
    def topology(self) -> int:
        return self.raw & 0xF

    @property
    def winding(self) -> int:
        return self.topology & 0x3

    @property
    def captain(self) -> int:
        for i in range(7, -1, -1):
            if self.raw & (1 << i):
                return i
        return -1

    @property
    def is_null(self) -> bool:
        return self.captain == -1

    def __str__(self) -> str:
        return (
            f"ByteWord[C:{self.control}, V:{self.value}, "
            f"T:0x{self.topology:01X}, W:{TorusWinding.to_str(self.winding)}, "
            f"Captain:{self.captain}, Raw:0x{self.raw:02X}]"
        )

    def deputize(self) -> 'ByteWord':
        if self.control == 1:
            return self
        value = self.value
        new_raw = ((value & 0x4) << 4) | ((value & 0x3) << 5) | self.topology
        return ByteWord(new_raw)

    def xor_cascade(self, other: 'ByteWord') -> 'ByteWord':
        new_w = self.winding ^ other.winding
        new_torus = (self.topology & 0xC) | new_w
        new_raw = (self.control << 7) | (self.value << 4) | new_torus
        return ByteWord(new_raw)

    def apply_unitary(self, mask: int) -> 'ByteWord':
        new_w = self.winding ^ (mask & 0x3)
        new_torus = (self.topology & 0xC) | new_w
        new_raw = (self.control << 7) | (self.value << 4) | new_torus
        return ByteWord(new_raw)

    def hash(self) -> bytes:
        return hashlib.sha256(bytes([self.raw])).digest()


# --- CPythonFrame for Runtime Tracking ---


@dataclass
class CPythonFrame:
    type_ptr: int
    obj_type: Type[Any]
    value: Any
    refcount: int = field(default=1)
    ttl: Optional[int] = None
    state: QState = field(init=False, default=QState.SUPERPOSITION)

    def __post_init__(self):
        self._birth = time.time()
        if self.ttl:
            self._expiry = self._birth + self.ttl
        else:
            self._expiry = None

    @classmethod
    def from_object(cls, obj: Any) -> 'CPythonFrame':
        return cls(
            type_ptr=id(type(obj)),
            obj_type=type(obj),
            value=obj,
            refcount=sys.getrefcount(obj) - 1,
        )

    def collapse(self) -> Any:
        self.state = QState.COLLAPSED
        return self.value


class ALU:
    def __init__(self, size: int = 8):
        self.registers = [ByteWord(0) for _ in range(size)]
        self.history = []
        self.state = SemanticState(1.0, 0.5, {}, "alu", [1.0, 0.0, 0.0])

    def add(self, reg1: int, reg2: int, dest: int) -> ByteWord:
        a, b = self.registers[reg1], self.registers[reg2]
        result = a
        if a.is_null or b.is_null:
            result = ByteWord(a.raw if b.is_null else b.raw)
        else:
            result = ByteWord(
                (a.raw & 0xF8) | ((a.value + b.value) & 0x7) << 4 | a.topology
            )
        self.registers[dest] = result
        self.history.append(result)
        self.state.perturb()
        return result

    def set_builder(self, ptr_reg: int, null_reg: int) -> None:
        if not self.registers[null_reg].is_null:
            return
        ptr = self.registers[ptr_reg]
        self.registers[ptr_reg] = ByteWord((1 << 7) | (ptr.value << 4) | ptr.topology)
        self.history.append(self.registers[ptr_reg])
        self.state.perturb()

    def apply_unitary(self, reg: int, dest: int, mask: int) -> ByteWord:
        result = self.registers[reg].apply_unitary(mask)
        self.registers[dest] = result
        self.history.append(result)
        self.state.perturb()
        return result

    def xor_cascade(self, reg1: int, reg2: int, dest: int) -> ByteWord:
        result = self.registers[reg1].xor_cascade(self.registers[reg2])
        self.registers[dest] = result
        self.history.append(result)
        self.state.perturb()
        return result

    def deputize(self, reg: int, dest: int) -> ByteWord:
        result = self.registers[reg].deputize()
        self.registers[dest] = result
        self.history.append(result)
        self.state.perturb()
        return result

    def morphic_field(self) -> List[int]:
        return [reg.value for reg in self.registers]

    def toroidal_laplacian(self) -> List[int]:
        field = self.morphic_field()
        n = len(field)
        return [
            field[(i - 1) % n] + field[(i + 1) % n] - 2 * field[i] for i in range(n)
        ]

    def heat_morph_step(self) -> None:
        lap = self.toroidal_laplacian()
        new_regs = []
        for bw, delta in zip(self.registers, lap, strict=False):
            if bw.captain == 7:
                mask = 1 << (0 if delta % 2 else 1)
                new_regs.append(bw.apply_unitary(mask))
            else:
                new_regs.append(bw)
        self.registers = new_regs
        self.history.extend(new_regs)
        self.state.perturb()

    def derive_operator_from_history(self) -> callable:
        if not self.history:
            return lambda x: x
        entropy = sum(reg.raw for reg in self.history) % 4
        mask = [(entropy >> 1) & 1, entropy & 1]
        return lambda bw: bw.apply_unitary(mask[0] << 1 | mask[1])

    def quineic_runtime_step(
        self, new_word: ByteWord
    ) -> Tuple[ByteWord, List[ByteWord]]:
        op = self.derive_operator_from_history()
        out = op(new_word)
        self.history.append(out)
        self.state.perturb()
        return out, self.history

    def is_quine(self, state: Tuple[ByteWord, List[ByteWord]]) -> bool:
        out, hist = state
        return len(hist) >= 2 and out.winding == hist[0].winding

    def is_oracle(self, state: Tuple[ByteWord, List[ByteWord]]) -> bool:
        return TorusWinding.to_str(state[0].winding) != "NULL"


def next_traversal_index(alu: ALU, current: int) -> int:
    """Compute next index based on winding."""
    winding = alu.registers[current].winding
    step = {
        TorusWinding.NULL: 1,
        TorusWinding.W1: 1,
        TorusWinding.W2: 2,
        TorusWinding.W12: 3,
    }[winding]
    return (current + step) % len(alu.registers)


def torus_trajectory(alu: ALU, steps: int) -> List[int]:
    """Generate traversal trajectory."""
    trajectory = [0]
    for _ in range(steps):
        trajectory.append(next_traversal_index(alu, trajectory[-1]))
    return trajectory


def detect_cycle(trajectory: List[int]) -> int:
    """Detect cycle length in trajectory."""
    seen = {}
    for i, idx in enumerate(trajectory):
        if idx in seen:
            return i - seen[idx]
        seen[idx] = i
    return 0  # No cycle found


def detect_cycle(trajectory: List[int]) -> int:
    """Detect cycle length in trajectory."""
    seen = {}
    for i, idx in enumerate(trajectory):
        if idx in seen:
            return i - seen[idx]
        seen[idx] = i
    return 0  # No cycle found


class MSCDSL:
    def __init__(self, alu: ALU):
        self.alu = alu
        self.commands = {
            "LOAD": self._load,
            "DEPUTIZE": self._deputize,
            "XOR_CASCADE": self._xor_cascade,
            "UNITARY": self._unitary,
            "ADD": self._add,
            "SET_BUILDER": self._set_builder,
            "HEAT_MORPH": self._heat_morph,
            "TRAVERSE": self._traverse,
            "PRINT": self._print,
            "QUINE_STEP": self._quine_step,
        }

    def _parse_reg(self, reg: str) -> int:
        if reg.startswith("R"):
            return int(reg[1:])
        raise ValueError(f"Invalid register: {reg}")

    def _load(self, args: List[str]) -> None:
        reg, value = args[0].split("=")
        reg = self._parse_reg(reg.strip())
        # Dangerous, replace with safer parsing in production
        value = eval(value.strip())
        self.alu.registers[reg] = value

    def _deputize(self, args: List[str]) -> None:
        src, dest = args[0].split("->")
        src = self._parse_reg(src.strip())
        dest = self._parse_reg(dest.strip())
        self.alu.deputize(src, dest)

    def _xor_cascade(self, args: List[str]) -> None:
        regs, dest = args[0].split("->")
        reg1, reg2 = [self._parse_reg(r.strip()) for r in regs.split(",")]
        dest = self._parse_reg(dest.strip())
        self.alu.xor_cascade(reg1, reg2, dest)

    def _unitary(self, args: List[str]) -> None:
        reg_mask, dest = args[0].split("->")
        reg, mask = reg_mask.split(",")
        reg = self._parse_reg(reg.strip())
        mask = int(mask.replace("MASK=", "").strip(), 2)
        dest = self._parse_reg(dest.strip())
        self.alu.apply_unitary(reg, dest, mask)

    def _add(self, args: List[str]) -> None:
        regs, dest = args[0].split("->")
        reg1, reg2 = [self._parse_reg(r.strip()) for r in regs.split(",")]
        dest = self._parse_reg(dest.strip())
        self.alu.add(reg1, reg2, dest)

    def _set_builder(self, args: List[str]) -> None:
        src, dest = args[0].split("->")
        src = self._parse_reg(src.strip())
        dest = self._parse_reg(dest.strip())
        self.alu.set_builder(src, dest)

    def _heat_morph(self, args: List[str]) -> None:
        self.alu.heat_morph_step()

    def _traverse(self, args: List[str]) -> None:
        steps, dest = args[0].split("->")
        steps = int(steps.strip().split()[0])
        dest = dest.strip()
        traj = torus_trajectory(self.alu, steps)
        self.alu.state.memory[dest] = traj

    def _print(self, args: List[str]) -> None:
        var = args[0].strip()
        if var in self.alu.state.memory:
            print(f"{var}: {self.alu.state.memory[var]}")
        else:
            print("Registers:")
            for i, reg in enumerate(self.alu.registers):
                print(f"R{i}: {reg}")

    def _quine_step(self, args: List[str]) -> None:
        reg, dest = args[0].split("->")
        reg = self._parse_reg(reg.strip())
        dest = dest.strip()
        state = self.alu.quineic_runtime_step(self.alu.registers[reg])
        self.alu.state.memory[dest] = state

    def execute(self, code: str) -> None:
        for line in code.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cmd, *args = line.split(maxsplit=1)
            if cmd in self.commands:
                self.commands[cmd](args)
            else:
                raise ValueError(f"Unknown command: {cmd}")


# QuineTransducer for self-modifying DSL


class QuineTransducer:
    def __init__(self, dsl: MSCDSL):
        self.dsl = dsl
        self.transformation_history = []
        self._current_source = textwrap.dedent("""
            def msc_program(alu):
                pass
        """)

    def transform(self, code: str) -> None:
        self.dsl.execute(code)
        new_source = self._generate_new_source(code)
        self.transformation_history.append(new_source)
        namespace = {}
        exec(new_source, namespace)
        self.dsl.commands["PROGRAM"] = lambda _: namespace["msc_program"](self.dsl.alu)
        self._current_source = new_source

    def _generate_new_source(self, code: str) -> str:
        # brittle, string-version
        lines = []
        for line in code.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(f"    alu.state.memory['last_cmd'] = {repr(line)}")
        body = "\n".join(lines)
        return f"def msc_program(alu):\n{body}\n"

    @property
    def source(self) -> str:
        return self._current_source


"""
# Glossary for fast integration + 'squaring' of smooth cognitive functions in Morphological Source Code

| Concept | Definition |
|---------|------------|
| **Fourier Transform** | Maps functions between time/spatial and frequency domains.  The specific form of the transform (including normalization constants) depends on the convention used. |
| **Square-Integrable Function (L²)** | A function whose squared magnitude integrates to a finite value. |
| **Hilbert Space** | A complete inner product space. L² spaces are Hilbert spaces. |
| **Inner Product** | ⟨f, g⟩ = ∫ f*(x) g(x) dx |
| **Hermitian Operator** | Ô satisfies ⟨f | Ôg⟩ = ⟨Ôf | g⟩. |
| **Momentum Operator (p̂)** | p̂ = -i ħ d/dx (in the spatial domain).  In the frequency domain, it becomes multiplication by ħk. |
| **Unitary Transformation** | A transformation that preserves inner products (up to a normalization factor). The Fourier transform is unitary. |

# SEMANTIC SUGAR (not syntax sugar; this is not your Mama's cupboard!)
Semantic sugar is about tagging stdlib-only bytecode for re-interpretation by downstream compilers, etc.

| Symbol  | AST Hint                        | MorphTag                 | Downstream Implication       |
| ------- | ------------------------------- | ------------------------ | ---------------------------- |
| `⟨f,g⟩` | `Operator(name='inner', ...)`   | `hilbert.L2.inner`       | Requires L² machinery        |
| `F{f}`  | `Operator(name='fourier', ...)` | `hilbert.L2.fourier`     | Optional FFT backend         |
| `δ`     | `Generator(name='delta', ...)`  | `hilbert.impulse.dirac`  | May suggest bandlimit        |
| `ψ*ψ`   | `Operator(name='density', ...)` | `quantum.wavefn.density` | Suggests Born interpretation |

Born interpretation: `ψ*ψ ⟹ morph.quantum.wavefn.density[normalized=True, probabilistic=True]`
---

## Fourier Transform

The Fourier transform maps a function f(x) into its frequency-domain representation F(k). This is expressed as:

    F[f(x)] = F(k) = ∫ from -∞ to ∞ f(x) e^(-2πi k x) dx
    F(k) = ∫₋∞^∞ f(x) e^(-ikx) dx

This integral operates on f(x), meaning the exponential term alone is not the Fourier transform but rather part of the kernel function.

The Fourier transform applies a feedback loop in frequency space, where functions transformed under `e^(-2πi k x)` can exhibit self-similar or dual properties, particularly in the case of Gaussians.

## Square-Integrable functions

A function f(x) is square-integrable over an interval [a,b] if:

    ∫ₐᵇ |f(x)|² dx < ∞

Or, in the case of functions over the entire real line (common in Fourier analysis):

    ∫₋∞^∞ |f(x)|² dx < ∞

This means that f(x) belongs to the space L²(a,b) or L²(ℝ), respectively.  L² represents the set of all such square-integrable functions.

Breaking It Down:

    |f(x)|² ensures we're dealing with the magnitude squared, avoiding issues with negative values.
    The integral ∫ |f(x)|² dx represents the total "energy" of the function.
    If this integral is finite, then f(x) belongs to the space of square-integrable functions, denoted as L²(a,b) or L²(ℝ).

Formal Definition:

A function f(x) belongs to the Hilbert space L²(a,b) if:

    f ∈ L²(a,b) ⟺ ∫ₐᵇ |f(x)|² dx < ∞

## Hilbert Spaces & Inner Products

L² spaces are Hilbert spaces.  A Hilbert space is a complete inner product space.

The space `L²(a,b)` (or more commonly `L²(ℝ)` for the whole real line) is the set of square-integrable functions over an interval `(a,b)`, defined as:

    L²(a,b) = {f : ∫ₐᵇ |f(x)|² dx < ∞}

L² as a Hilbert Space: L² is a complete inner product space with the inner product:

    ⟨f, g⟩ = ∫ₐᵇ f*(x) g(x) dx

    where f*(x) is the complex conjugate of f(x).

This inner product allows us to define orthogonality:

    ⟨f, g⟩ = 0 ⇒ f ⊥ g

    The norm associated with this inner product is:

    ‖f‖ = √⟨f, f⟩ = (∫ₐᵇ |f(x)|² dx)^(1/2)

## Hermitian Operators

In a Hilbert space, an operator Ô is Hermitian if:

    ⟨f | Ôg⟩ = ⟨Ôf | g⟩, for all functions f, g in the space.

The Fourier transform itself isn't Hermitian, but the momentum operator in quantum mechanics is:

    p̂ = -iħ d/dx

which satisfies:

    ⟨f | p̂g⟩ = ⟨p̂f | g⟩

The Fourier transform *is* a unitary operator.  However, it does not diagonalize the momentum operator directly.  Instead, when the momentum operator is transformed to the frequency domain using the Fourier transform, it becomes a multiplicative operator:

    p̂f(x) = -iħ d/dx f(x)  ⟶  pF(k) = ħkF(k)

    meaning in Fourier space; momentum simply acts as multiplication by k.

The Fourier transform is unitary, meaning it preserves inner products (up to a normalization constant, depending on the specific definition of the Fourier transform used):

    ⟨F^f, F^g⟩ = ⟨f, g⟩

where F^ is the Fourier transform operator. This ensures that Fourier transforms preserve energy (norms) in L².  The specific form of the normalization depends on the convention used for the Fourier transform.

    ⟨F, G⟩ = ⟨f, g⟩

which ensures that Fourier transforms preserve energy (norms) in L².
"""


def aluTest():
    # Test the DSL
    alu = ALU()
    dsl = MSCDSL(alu)
    transducer = QuineTransducer(dsl)

    program = """
        LOAD R0 = ByteWord(0b10110011)
        LOAD R1 = ByteWord(0b01010001)
        LOAD R2 = ByteWord(0b11100010)
        LOAD R3 = ByteWord(0b00000000)
        DEPUTIZE R1 -> R4
        XOR_CASCADE R0, R2 -> R5
        UNITARY R0, MASK=0b01 -> R6
        ADD R0, R3 -> R7
        SET_BUILDER R0 -> R3
        HEAT_MORPH
        TRAVERSE 10 STEPS -> trajectory
        PRINT trajectory
        QUINE_STEP R0 -> state
        PRINT state
    """

    print("Executing DSL program:")
    dsl.execute(program)
    print("\nTransducing program:")
    transducer.transform(program)
    print("Transduced source:")
    print(transducer.source)


if __name__ == "__main__":
    mc = MorphicComplex(1, 2)
    print(mc, mc.conjugate(), mc * mc)
    bw = ByteWord(0b10110010)
    print(bw)

    mp = MorphologicPyOb(
        entropy=1.0,
        trigger_threshold=0.5,
        memory={"init": True},
        signature="abc123",
        # state=state
    )

    cf = CPythonFrame.from_object([1, 2, 3])
    print(cf, cf.collapse())

aluTest()
