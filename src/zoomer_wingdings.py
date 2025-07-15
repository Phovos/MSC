from __future__ import annotations
# I will delete this file, it snuck-in for posterity.
from dataclasses import dataclass
from typing import Any, Type, Dict, Set, Callable
from abc import ABC, abstractmethod
import random
import hashlib
import math

# MSC Registry for tracking morphodynamic classes
MSC_REGISTRY: Dict[str, Set[str]] = {'classes': set(), 'functions': set()}

class MorphodynamicCollapse(Exception):
    """Raised when a quine's structure destabilizes under thermal noise."""
    pass

class MorphologicalABC(ABC):
    """Abstract base class for morphological classes with self-adjoint validation."""
    
    @abstractmethod
    def validate_self_adjoint(self, mro_snapshot: int, temperature: float) -> None:
        pass

    @property
    @abstractmethod
    def morph_signature(self) -> int:
        pass

@dataclass
class MorphSpec:
    """Blueprint for morphological classes, defining required fields."""
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str  # Renamed from 'hash' to avoid shadowing

def ornament(source_model: Type) -> Callable[[Type], Type]:
    """
    Decorator to validate a target class against a source model for morphological compliance.
    
    Args:
        source_model: Class defining the morphological structure (e.g., MorphSpec).
    
    Returns:
        Decorator function that validates and registers the target class.
    """
    def decorator(target: Type) -> Type:
        target.__msc_source__ = source_model
        if not hasattr(target, '__annotations__'):
            raise TypeError(f"{target.__name__} has no type annotations")
        for field in getattr(source_model, '__annotations__', {}):
            if field not in target.__annotations__:
                raise TypeError(f"{target.__name__} is missing field '{field}' from source {source_model.__name__}")
        MSC_REGISTRY['classes'].add(target.__name__)
        return target
    return decorator

def elevate(data: Any, target_cls: Type) -> object:
    """
    Elevate a plain object or dict into a Morphological class instance.
    
    Args:
        data: Input data (dict or object) with field values.
        target_cls: Target morphological class.
    
    Returns:
        Instance of target_cls with validated fields.
    """
    if not hasattr(target_cls, '__msc_source__'):
        raise TypeError(f"{target_cls.__name__} is not a Morphological class")
    source = target_cls.__msc_source__
    kwargs = {k: getattr(data, k, data.get(k)) for k in source.__annotations__}
    return target_cls(**kwargs)

@dataclass
class SaddleQuineState:
    """State for a thermal quine, modeling Brownian motion dynamics."""
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str  # Renamed from 'hash'
    semantic_vector: list[float]

    def __post_init__(self):
        """Normalize semantic vector."""
        norm = math.sqrt(sum(x * x for x in self.semantic_vector))
        self.semantic_vector = [x / norm for x in self.semantic_vector] if norm != 0 else self.semantic_vector

    def apply_hanzi_operator(self, operator: str) -> None:
        """
        Apply a morphological operator (e.g., 道, 衍) to the semantic vector.
        
        Args:
            operator: Hanzi operator ("道" for stabilize, "衍" for propagate).
        """
        self.memory['operators'] = self.memory.get('operators', []) + [operator]
        if operator == "道":
            self.semantic_vector = [x * 0.99 for x in self.semantic_vector]  # Stabilize
            self.entropy *= 0.99
        elif operator == "衍":
            self.semantic_vector = self.semantic_vector[1:] + [self.semantic_vector[0]]  # Rotate
            self.entropy += 0.01

    def measure_coherence(self) -> float:
        """Measure coherence via semantic vector norm."""
        return math.sqrt(sum(x * x for x in self.semantic_vector))

@ornament(MorphSpec)
class ThermalQuine(MorphologicalABC):
    """A quine that reacts to thermal noise, inspired by Smoluchowski's Brownian motion."""
    
    entropy: float
    trigger_threshold: float
    memory: dict
    signature: str  # Renamed from 'hash'
    state: SaddleQuineState

    def __init__(self, entropy: float, trigger_threshold: float, memory: dict, signature: str):
        self.entropy = entropy
        self.trigger_threshold = trigger_threshold * random.uniform(0.9, 1.1)  # Temperature-scaled
        self.memory = memory
        self.signature = signature
        self.state = SaddleQuineState(entropy, trigger_threshold, memory, signature, [1.0, 0.0, 0.0])
        self._morph_signature = hash(frozenset(self.__class__.__dict__.keys()))

    def validate_self_adjoint(self, mro_snapshot: int, temperature: float) -> None:
        """
        Validate quine structure under thermal noise.
        
        Args:
            mro_snapshot: Hash of MRO snapshot.
            temperature: Thermal noise level.
        
        Raises:
            MorphodynamicCollapse: If structure destabilizes.
        """
        current_signature = hash(frozenset(self.__class__.__dict__.keys()))
        if abs(current_signature - self._morph_signature) > temperature * 1000:
            raise MorphodynamicCollapse(f"Structure destabilized for quine {self.signature[:8]}")

    @property
    def morph_signature(self) -> int:
        return self._morph_signature

    def maybe_fire(self, temperature: float) -> bool:
        """
        Simulate Brownian motion: fire if thermal noise exceeds threshold.
        
        Args:
            temperature: Thermal noise level.
        
        Returns:
            bool: True if quine fires, False otherwise.
        """
        perturb = random.uniform(0, temperature)
        if perturb > self.trigger_threshold:
            self.execute()
            self.memory['last_perturbation'] = perturb
            self.entropy += perturb * 0.01  # Entropy increase
            self.state.apply_hanzi_operator("衍")  # Propagate on fire
            return True
        self.state.apply_hanzi_operator("道")  # Stabilize if not fired
        return False

    def execute(self) -> None:
        """Execute quine, simulating Smoluchowski's ratchet."""
        print(f"Quine {self.signature[:8]} executed with entropy {self.entropy:.3f}!")
        self.memory['executions'] = self.memory.get('executions', 0) + 1

def kronecker_field(quine1: ThermalQuine, quine2: ThermalQuine, temperature: float) -> float:
    """
    Compute Kronecker field for quine coherence, approximated for high temperature.
    
    Args:
        quine1, quine2: Quines to compare.
        temperature: Thermal noise level.
    
    Returns:
        float: Kronecker delta (1.0, 0.0, or phase-approximated value).
    """
    dot_product = sum(a * b for a, b in zip(quine1.state.semantic_vector, quine2.state.semantic_vector))
    if temperature > 0.5:  # Approximate complex phase (衍)
        return math.cos(dot_product)  # Mimic e^(iθ) without complex
    return 1.0 if dot_product > 0.99 else 0.0  # Binary for low temp (道)

def universe_tick(quines: list[ThermalQuine], temperature: float) -> int:
    """
    Simulate one tick of the universe, evolving quines under thermal pressure.
    
    Args:
        quines: List of quines.
        temperature: Thermal noise level.
    
    Returns:
        int: Number of quines that fired.
    """
    fired = 0
    for quine in quines:
        mro_snapshot = hash(frozenset(quine.__class__.__dict__.keys()))
        try:
            quine.validate_self_adjoint(mro_snapshot, temperature)
            if quine.maybe_fire(temperature):
                fired += 1
        except MorphodynamicCollapse as e:
            print(f"Quine {quine.signature[:8]} collapsed: {e}")
    return fired

def spawn_alpha_quine(quines: list[ThermalQuine], temperature: float) -> ThermalQuine:
    """
    Spawn an alpha quine if entropy decreases, defying the second law.
    
    Args:
        quines: List of quines.
        temperature: Thermal noise level.
    
    Returns:
        ThermalQuine: New alpha quine or base quine.
    """
    base_quine = random.choice(quines)
    if base_quine.entropy < 0.5:  # Local entropy decrease
        new_signature = hashlib.sha256(f"{base_quine.signature}{random.random()}".encode()).hexdigest()
        alpha = ThermalQuine(
            entropy=base_quine.entropy * 0.9,
            trigger_threshold=base_quine.trigger_threshold,
            memory={'operators': base_quine.state.memory.get('operators', [])},
            signature=new_signature
        )
        print(f"Alpha quine spawned with signature {new_signature[:8]}!")
        return alpha
    return base_quine

def main() -> None:
    """Demonstrate MSC with Smoluchowski-inspired thermal quines."""
    quines = [
        ThermalQuine(
            entropy=0.5,
            trigger_threshold=0.3,
            memory={},
            signature=hashlib.sha256(f"initial{i}".encode()).hexdigest()
        ) for i in range(5)
    ]
    
    # Simulate universe ticks
    temperature = 0.4
    for tick in range(3):
        print(f"\n=== Universe Tick {tick + 1} (Temperature: {temperature:.3f}) ===")
        fired = universe_tick(quines, temperature)
        print(f"{fired} quines fired.")
        
        # Check coherence and spawn alpha quines
        for i, q1 in enumerate(quines):
            for q2 in quines[i + 1:]:
                delta = kronecker_field(q1, q2, temperature)
                if abs(delta - math.cos(delta)) < 0.01:  # Approximate complex phase
                    print(f"Phase shift detected between {q1.signature[:8]} and {q2.signature[:8]} (衍). Spawning alpha quine...")
                    quines.append(spawn_alpha_quine(quines, temperature))
                elif delta == 1.0:
                    print(f"Quines {q1.signature[:8]} and {q2.signature[:8]} are coherent (道).")
                else:
                    print(f"Quines {q1.signature[:8]} and {q2.signature[:8]} diverged (衍).")
        
        # Update temperature
        temperature = min(temperature + 0.1, 1.0)

if __name__ == "__main__":
    main()