from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Type, Callable, Dict, List
from abc import ABC, abstractmethod
import hashlib
import math
import random

# === Morphological Specification and Registry ===

MSC_REGISTRY: Dict[str, List[str]] = {
    "classes": [],
    "functions": [],
}


@dataclass
class MorphSpec:
    """Declarative schema for Morphological classes."""
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str


def morphology(spec: Type[MorphSpec]) -> Callable[[Type], Type]:
    """Decorator for registering and validating morphological classes."""
    def decorator(cls: Type) -> Type:
        cls.__msc_source__ = spec
        for field in spec.__annotations__:
            if field not in cls.__annotations__:
                raise TypeError(f"{cls.__name__} missing required field: {field}")
        MSC_REGISTRY["classes"].append(cls.__name__)
        return cls
    return decorator


# === Abstract Base Class ===

class MorphologicPyOb(ABC):
    """Abstract base for all morphological entities."""

    @abstractmethod
    def validate_self_adjoint(self, mro_snapshot: int, temperature: float) -> None:
        pass

    @property
    @abstractmethod
    def morph_signature(self) -> int:
        pass


# === Morphological Runtime Helpers ===

def elevate(data: Any, cls: Type) -> object:
    """Raise a dict or object to a registered morphological class."""
    if not hasattr(cls, '__msc_source__'):
        raise TypeError(f"{cls.__name__} is not a morphological class.")
    source = cls.__msc_source__
    kwargs = {k: getattr(data, k, data.get(k)) for k in source.__annotations__}
    return cls(**kwargs)


# === Simulation State ===

@dataclass
class SemanticState:
    """Encodes semantic structure and coherence vector."""
    vector: List[float]

    def normalize(self) -> None:
        norm = math.sqrt(sum(x * x for x in self.vector))
        if norm != 0:
            self.vector = [x / norm for x in self.vector]

    def dot(self, other: SemanticState) -> float:
        return sum(a * b for a, b in zip(self.vector, other.vector))

    def perturb(self, factor: float) -> None:
        self.vector = [x + random.uniform(-factor, factor) for x in self.vector]
        self.normalize()


# === Morphological Collapse Exception ===

class MorphodynamicCollapse(Exception):
    pass


# === Thermal Quine V2 ===

@morphology(MorphSpec)
class ThermalQuineV2(MorphologicPyOb):
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str
    state: SemanticState

    def __init__(self, entropy: float, trigger_threshold: float, memory: dict, signature: str):
        self.entropy = entropy
        self.trigger_threshold = trigger_threshold
        self.memory = memory
        self.signature = signature
        self.state = SemanticState([1.0, 0.0, 0.0])
        self._morph_signature = hash(frozenset(self.__class__.__dict__.keys()))

    def validate_self_adjoint(self, mro_snapshot: int, temperature: float) -> None:
        """Ensure class structure hasn't drifted beyond coherence."""
        current_sig = hash(frozenset(self.__class__.__dict__.keys()))
        if abs(current_sig - self._morph_signature) > temperature * 1000:
            raise MorphodynamicCollapse(f"{self.signature[:8]} destabilized at temp {temperature:.3f}")

    @property
    def morph_signature(self) -> int:
        return self._morph_signature

    def maybe_fire(self, temperature: float) -> bool:
        perturb = random.uniform(0, temperature)
        if perturb > self.trigger_threshold:
            self.execute()
            self.entropy += perturb * 0.01
            self.memory["last_perturbation"] = perturb
            self.state.perturb(factor=0.01)
            return True
        self.state.perturb(factor=0.001)
        return False

    def execute(self) -> None:
        print(f"🔥 Quine {self.signature[:8]} executed at entropy {self.entropy:.3f}")
        self.memory["executions"] = self.memory.get("executions", 0) + 1


# === Morphological Simulation System ===

class MorphicSystem:
    def __init__(self, temperature: float = 0.4):
        self.temperature = temperature
        self.quines: List[ThermalQuineV2] = []

    def seed(self, count: int) -> None:
        for i in range(count):
            signature = hashlib.sha256(f"seed{i}".encode()).hexdigest()
            q = ThermalQuineV2(
                entropy=0.5,
                trigger_threshold=0.3,
                memory={},
                signature=signature,
            )
            self.quines.append(q)

    def tick(self) -> int:
        fired = 0
        for q in self.quines:
            snapshot = hash(frozenset(q.__class__.__dict__.keys()))
            try:
                q.validate_self_adjoint(snapshot, self.temperature)
                if q.maybe_fire(self.temperature):
                    fired += 1
            except MorphodynamicCollapse as e:
                print(f"💥 Collapse: {e}")
        return fired

    def spawn_alpha_quine(self) -> None:
        q = random.choice(self.quines)
        if q.entropy < 0.4:
            new_sig = hashlib.sha256(f"{q.signature}{random.random()}".encode()).hexdigest()
            alpha = ThermalQuineV2(
                entropy=q.entropy * 0.9,
                trigger_threshold=q.trigger_threshold,
                memory=dict(q.memory),
                signature=new_sig
            )
            self.quines.append(alpha)
            print(f"🧬 Alpha Quine spawned: {new_sig[:8]}")

    def run(self, ticks: int = 3) -> None:
        for t in range(ticks):
            print(f"\n=== Tick {t + 1} (Temp: {self.temperature:.2f}) ===")
            fired = self.tick()
            print(f"{fired} quines fired.")
            self.spawn_alpha_quine()
            self.temperature = min(1.0, self.temperature + 0.1)


# === Entrypoint ===

def main() -> None:
    system = MorphicSystem()
    system.seed(count=5)
    system.run(ticks=5)


if __name__ == "__main__":
    main()
