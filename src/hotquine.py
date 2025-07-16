from __future__ import annotations
import enum
import time
import random
import sys
import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type, TypeVar, Dict, Set, Callable, Protocol, runtime_checkable
from functools import wraps

T = TypeVar('T')
V = TypeVar('V')
C = TypeVar('C')

# --- MSC Registry and Spec ---

MSC_REGISTRY: Dict[str, Set[str]] = {'classes': set(), 'functions': set()}

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
        target.__msc_source__ = source_model
        # Ensure target has all annotated fields from source_model
        for field_name in getattr(source_model, '__annotations__', {}):
            if field_name not in getattr(target, '__annotations__', {}):
                raise TypeError(f"{target.__name__} missing field '{field_name}'")
        MSC_REGISTRY['classes'].add(target.__name__)
        return target
    return decorator

# --- Core Numeric & State Types ---

class MorphicComplex:
    """Complex number with morphic properties."""
    def __init__(self, real: float, imag: float):
        self.real = real; self.imag = imag
    def conjugate(self) -> 'MorphicComplex':
        return MorphicComplex(self.real, -self.imag)
    def __add__(self, o: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(self.real + o.real, self.imag + o.imag)
    def __mul__(self, o: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(
            self.real*o.real - self.imag*o.imag,
            self.real*o.imag + self.imag*o.real
        )
    def __repr__(self) -> str:
        if self.imag == 0: return f"{self.real}"
        sign = "+" if self.imag >= 0 else ""
        return f"{self.real}{sign}{self.imag}j"

class Morphology(enum.IntEnum):
    MORPHIC = 0
    DYNAMIC = 1
    MARKOVIAN = -1
    NON_MARKOVIAN = 1  # placeholder

class QState(enum.Enum):
    SUPERPOSITION = 1
    ENTANGLED    = 2
    COLLAPSED    = 4
    DECOHERENT   = 8

# --- Base Morphologic Object ---

@dataclass
class MorphologicPyOb:
    """Unified runtime & morphological object."""
    symmetry: str
    conservation: str
    lhs: str
    rhs: List[str]
    value: Any
    ttl: Optional[int] = None
    state: QState = QState.SUPERPOSITION
    
    def __post_init__(self):
        self._birth = time.time()
        self._state = self.state
        self._ref = 1
        self._super = [self.value] if self._state == QState.SUPERPOSITION else []
        self._ent   = [self.value] if self._state == QState.ENTANGLED else []
    
    def apply_transformation(self, seq: List[str]) -> List[str]:
        if self.lhs in seq:
            idx = seq.index(self.lhs)
            seq = seq[:idx] + self.rhs + seq[idx+1:]
            self._state = QState.ENTANGLED
        return seq
    
    def collapse(self) -> Any:
        if self._state != QState.COLLAPSED:
            if self._state == QState.SUPERPOSITION and self._super:
                self.value = random.choice(self._super)
            self._state = QState.COLLAPSED
        return self.value

# --- Thermal Quine (V2) ---

@morphology(MorphSpec)
class ThermalQuine(MorphologicPyOb):
    """MorphologicPyOb specialized as a thermal quine."""
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str
    semantic_vector: List[float]
    
    def __init__(
        self,
        entropy: float,
        trigger_threshold: float,
        memory: dict,
        signature: str,
        semantic_vector: List[float]
    ):
        super().__init__(
            symmetry="thermal",
            conservation="entropy",
            lhs="", rhs=[],
            value=None
        )
        self.entropy = entropy
        self.trigger_threshold = trigger_threshold * random.uniform(0.9, 1.1)
        self.memory = memory
        self.signature = signature
        norm = math.sqrt(sum(x*x for x in semantic_vector))
        self.semantic_vector = ([x/norm for x in semantic_vector] if norm else semantic_vector)
        self._morph_signature = hash(frozenset(self.__class__.__dict__.keys()))
    
    def validate_self_adjoint(self, mro_snapshot: int, temperature: float) -> None:
        current_sig = hash(frozenset(self.__class__.__dict__.keys()))
        if abs(current_sig - self._morph_signature) > temperature * 1000:
            raise RuntimeError(f"Structure destabilized: {self.signature[:8]}")
    
    def maybe_fire(self, temperature: float) -> bool:
        perturb = random.uniform(0, temperature)
        if perturb > self.trigger_threshold:
            self.execute()
            self.memory['last_perturb'] = perturb
            self.entropy += perturb * 0.01
            return True
        return False
    
    def execute(self) -> None:
        print(f"Quine {self.signature[:8]} fired; entropy={self.entropy:.3f}")
        self.memory['executions'] = self.memory.get('executions', 0) + 1

# --- System Orchestration ---

class MorphicSystem:
    """Manages a population of ThermalQuines under thermal dynamics."""
    def __init__(self, quines: List[ThermalQuine], temperature: float = 0.5):
        self.quines = quines
        self.temperature = temperature
    
    def tick(self) -> int:
        fired = 0
        for q in self.quines:
            mro_snap = hash(frozenset(q.__class__.__dict__.keys()))
            try:
                q.validate_self_adjoint(mro_snap, self.temperature)
                if q.maybe_fire(self.temperature):
                    fired += 1
            except Exception as e:
                print(f"Collapse: {e}")
        return fired
    
    def spawn_alpha(self) -> ThermalQuine:
        base = random.choice(self.quines)
        if base.entropy < 0.5:
            sig = hashlib.sha256(f"{base.signature}{random.random()}".encode()).hexdigest()
            alpha = ThermalQuine(
                entropy=base.entropy * 0.9,
                trigger_threshold=base.trigger_threshold,
                memory={'operators': base.memory.get('operators', [])},
                signature=sig,
                semantic_vector=base.semantic_vector
            )
            print(f"Alpha quine {sig[:8]} spawned!")
            return alpha
        return base

# --- Demo Execution ---

if __name__ == "__main__":
    quines = [
        ThermalQuine(0.5, 0.3, {}, hashlib.sha256(f"q{i}".encode()).hexdigest(), [1.0,0.0,0.0])
        for i in range(5)
    ]
    system = MorphicSystem(quines, temperature=0.4)
    for tick in range(3):
        print(f"\n-- Tick {tick+1} @ T={system.temperature:.2f} --")
        fired = system.tick()
        print(f"Fired: {fired}")
        new_q = system.spawn_alpha()
        if new_q not in system.quines:
            system.quines.append(new_q)
        system.temperature = min(system.temperature + 0.1, 1.0)
