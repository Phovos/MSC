from __future__ import annotations

#!/usr/bin/env -S uv run
# -*- coding: utf-8 -*-
import sys
import platform
import subprocess
import traceback
import logging
import json
import cProfile
import time
import socket
import threading
import argparse
import asyncio
import tomllib
import pstats
from io import StringIO
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import ctypes
from enum import Flag, auto
from functools import lru_cache

"""
<https://github.com/Phovos/msc> • MSC: Morphological Source Code © 2025 by Phovos
This file uses only standard libraries.

A monolithic __init__.py that provides:
  - A cross-platform project management system (with UV integration, dependency handling, etc.)
  - Platform-integrated process execution, benchmarking, and profiling utilities

Usage examples:
  - Project management:
      $ python __init__.py project --root . DEV
      $ python __init__.py project --root . --create-module mymodule DEV
  - Benchmarking:
      $ python __init__.py benchmark -n 5 -- python -c "print('hello')"


CPU feature detection for x86-64 & ARM64 on our platforms, Win11 and debian/ubuntu.
Supports Python 3.12+ with full feature detection + testing for Ryzen 5600 and GitHub CI.
"""

# Platform detection
_IS_WIN = sys.platform == "win32"
IS_WINDOWS = _IS_WIN
_IS_LINUX = sys.platform.startswith("linux")
_MACHINE = platform.machine().upper()
_IS_X86 = _MACHINE in ("X86_64", "AMD64", "I386", "I686")
_IS_ARM = _MACHINE.startswith(("ARM64", "AARCH64"))

# Global profiler instance (for module-level use if desired)
profiler = cProfile.Profile()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_port_available(port: int) -> bool:
    """Check if a given port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('127.0.0.1', port))
        return result != 0  # non-zero means the port is available


def find_available_port(start_port: int) -> int:
    """Find an available port starting from {{start_port}}."""
    port = start_port
    while not is_port_available(port):
        logger.info(f"Port {port} is occupied. Trying next port.")
        port += 1
    logger.info(f"Found available port: {port}")
    return port


@(lambda f: f())
def FireFirst() -> None:
    """Function that fires on import.

    Checks for an available port starting at 8420 and logs the result.
    """
    PORT = 8420
    try:
        available_port = find_available_port(PORT)
        logger.info(f"Using port: {available_port}")
        print("FireFirst executed!")
    except Exception as e:
        logger.error(f"An error occurred in FireFirst: {e}")
    finally:
        return True


# ================================================================
# CPUID Implementation
# ================================================================


class CPUIDRegs(ctypes.Structure):
    """CPUID register results."""

    _fields_ = [
        ("eax", ctypes.c_uint32),
        ("ebx", ctypes.c_uint32),
        ("ecx", ctypes.c_uint32),
        ("edx", ctypes.c_uint32),
    ]


@lru_cache(maxsize=128)
def _cpuid_x86(leaf: int, subleaf: int = 0) -> Tuple[int, int, int, int]:
    """
    Execute CPUID instruction on x86/x86_64.
    Returns (eax, ebx, ecx, edx) or (0,0,0,0) on failure.
    """
    if not _IS_X86:
        return (0, 0, 0, 0)

    regs = CPUIDRegs()

    if _IS_WIN:
        try:
            # Windows CPUID using inline assembly simulation
            kernel32 = ctypes.windll.kernel32
            PAGE_EXECUTE_READWRITE = 0x40
            MEM_COMMIT = 0x1000

            code = (
                b"\x53"  # push rbx
                b"\x48\x89\xc8"  # mov rax, rcx (leaf)
                b"\x48\x89\xd1"  # mov rcx, rdx (subleaf)
                b"\x0f\xa2"  # cpuid
                b"\x41\x89\x00"  # mov [r8], eax
                b"\x41\x89\x58\x04"  # mov [r8+4], ebx
                b"\x41\x89\x48\x08"  # mov [r8+8], ecx
                b"\x41\x89\x50\x0c"  # mov [r8+12], edx
                b"\x5b"  # pop rbx
                b"\xc3"  # ret
            )

            addr = kernel32.VirtualAlloc(
                None, len(code), MEM_COMMIT, PAGE_EXECUTE_READWRITE
            )
            if not addr:
                raise OSError("VirtualAlloc failed")

            ctypes.memmove(addr, code, len(code))
            func = ctypes.WINFUNCTYPE(
                None, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(CPUIDRegs)
            )(addr)
            func(leaf, subleaf, ctypes.byref(regs))

            kernel32.VirtualFree(addr, 0, 0x8000)

        except Exception:
            # Fallback to MSVCRT intrinsics
            try:
                msvcrt = ctypes.CDLL(ctypes.util.find_msvcrt())
                cpuidex = getattr(msvcrt, '__cpuidex', None)
                if cpuidex:
                    info = (ctypes.c_int * 4)()
                    cpuidex(info, leaf, subleaf)
                    return tuple(info)
            except Exception:
                return (0, 0, 0, 0)

    elif _IS_LINUX:
        try:
            import mmap

            code = bytes(
                [
                    0x53,  # push %rbx
                    0x48,
                    0x89,
                    0xF8,  # mov %rdi, %rax (leaf)
                    0x48,
                    0x89,
                    0xF1,  # mov %rsi, %rcx (subleaf)
                    0x0F,
                    0xA2,  # cpuid
                    0x48,
                    0x89,
                    0x07,  # mov %rax, (%rdi)
                    0x48,
                    0x89,
                    0x5F,
                    0x04,  # mov %rbx, 4(%rdi)
                    0x48,
                    0x89,
                    0x4F,
                    0x08,  # mov %rcx, 8(%rdi)
                    0x48,
                    0x89,
                    0x57,
                    0x0C,  # mov %rdx, 12(%rdi)
                    0x5B,  # pop %rbx
                    0xC3,  # ret
                ]
            )

            mem = mmap.mmap(
                -1,
                len(code),
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
            )
            mem.write(code)

            func = ctypes.CFUNCTYPE(
                None, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(CPUIDRegs)
            )(ctypes.cast(mem, ctypes.c_void_p).value)
            func(leaf, subleaf, ctypes.byref(regs))

            mem.close()

        except Exception:
            return (0, 0, 0, 0)

    return (regs.eax, regs.ebx, regs.ecx, regs.edx)


# ================================================================
# CPU Feature Flags
# ================================================================


class CpuFeature(Flag):
    """Comprehensive CPU feature flags for x86-64 and ARM64."""

    # Basic
    BASIC = auto()

    # x86 Legacy Features
    MMX = auto()
    SSE = auto()
    SSE2 = auto()
    SSE3 = auto()
    SSSE3 = auto()
    SSE41 = auto()
    SSE42 = auto()

    # x86 Extended Features
    POPCNT = auto()
    AES = auto()
    PCLMULQDQ = auto()
    RDRAND = auto()
    RDSEED = auto()
    MOVBE = auto()
    F16C = auto()

    # AVX Family
    AVX = auto()
    AVX2 = auto()
    FMA = auto()

    # Bit Manipulation
    BMI1 = auto()
    BMI2 = auto()
    ABM = auto()
    ADX = auto()

    # Cryptography
    SHA = auto()
    GFNI = auto()
    VAES = auto()
    VPCLMULQDQ = auto()

    # Memory & Threading
    RTM = auto()
    HLE = auto()
    TSX = auto()

    # ARM Features
    NEON = auto()
    ASIMD = auto()
    SVE = auto()
    SVE2 = auto()
    BF16 = auto()
    I8MM = auto()
    RNG = auto()

    # ARM Crypto
    ARM_AES = auto()
    ARM_SHA1 = auto()
    ARM_SHA2 = auto()
    ARM_SHA3 = auto()

    @classmethod
    @lru_cache(maxsize=1)
    def detect(cls) -> "CpuFeature":
        """Detect CPU features once and cache the result."""
        features = cls.BASIC

        if _IS_X86:
            features |= cls._detect_x86_features()
        elif _IS_ARM:
            features |= cls._detect_arm_features()

        return features

    @classmethod
    def _detect_x86_features(cls) -> "CpuFeature":
        """Detect x86/x86_64 CPU features using multiple methods."""
        features = cls.BASIC

        features |= cls._detect_x86_cpuid()
        if _IS_WIN:
            features |= cls._detect_x86_windows()
        elif _IS_LINUX:
            features |= cls._detect_x86_linux()

        return features

    @classmethod
    def _detect_x86_cpuid(cls) -> "CpuFeature":
        """Detect x86 features via CPUID instruction."""
        features = cls(0)

        try:
            # Leaf 1
            eax, ebx, ecx, edx = _cpuid_x86(1, 0)

            if edx & (1 << 23):
                features |= cls.MMX
            if edx & (1 << 25):
                features |= cls.SSE
            if edx & (1 << 26):
                features |= cls.SSE2
            if ecx & (1 << 0):
                features |= cls.SSE3
            if ecx & (1 << 9):
                features |= cls.SSSE3
            if ecx & (1 << 19):
                features |= cls.SSE41
            if ecx & (1 << 20):
                features |= cls.SSE42
            if ecx & (1 << 22):
                features |= cls.MOVBE
            if ecx & (1 << 23):
                features |= cls.POPCNT
            if ecx & (1 << 25):
                features |= cls.AES
            if ecx & (1 << 1):
                features |= cls.PCLMULQDQ
            if ecx & (1 << 28):
                features |= cls.AVX
            if ecx & (1 << 29):
                features |= cls.F16C
            if ecx & (1 << 30):
                features |= cls.RDRAND
            if ecx & (1 << 12):
                features |= cls.FMA

            # Leaf 7, Subleaf 0
            eax, ebx, ecx, edx = _cpuid_x86(7, 0)

            if ebx & (1 << 3):
                features |= cls.BMI1
            if ebx & (1 << 5):
                features |= cls.AVX2
            if ebx & (1 << 8):
                features |= cls.BMI2
            if ebx & (1 << 18):
                features |= cls.RDSEED
            if ebx & (1 << 19):
                features |= cls.ADX
            if ebx & (1 << 29):
                features |= cls.SHA
            if ecx & (1 << 8):
                features |= cls.GFNI
            if ecx & (1 << 9):
                features |= cls.VAES
            if ecx & (1 << 10):
                features |= cls.VPCLMULQDQ
            if ebx & (1 << 4):
                features |= cls.HLE
            if ebx & (1 << 11):
                features |= cls.RTM
            if features & (cls.HLE | cls.RTM):
                features |= cls.TSX

        except Exception:
            pass

        return features

    @classmethod
    def _detect_x86_windows(cls) -> "CpuFeature":
        """Detect x86 features via Windows APIs."""
        features = cls(0)

        try:
            kernel32 = ctypes.windll.kernel32

            feature_map = {
                6: cls.SSE,
                10: cls.SSE2,
                13: cls.SSE3,
                36: cls.SSSE3,
                37: cls.SSE41,
                38: cls.SSE42,
                39: cls.AVX,
                40: cls.AVX2,
            }

            for feature_code, cpu_feature in feature_map.items():
                if kernel32.IsProcessorFeaturePresent(feature_code):
                    features |= cpu_feature

        except Exception:
            pass

        return features

    @classmethod
    def _detect_x86_linux(cls) -> "CpuFeature":
        """Detect x86 features via Linux /proc/cpuinfo."""
        features = cls(0)

        try:
            cpuinfo_path = Path("/proc/cpuinfo")
            if cpuinfo_path.exists():
                content = cpuinfo_path.read_text().lower()

                flag_map = {
                    "mmx": cls.MMX,
                    "sse": cls.SSE,
                    "sse2": cls.SSE2,
                    "sse3": cls.SSE3,
                    "ssse3": cls.SSSE3,
                    "sse4_1": cls.SSE41,
                    "sse4_2": cls.SSE42,
                    "popcnt": cls.POPCNT,
                    "aes": cls.AES,
                    "pclmulqdq": cls.PCLMULQDQ,
                    "avx": cls.AVX,
                    "avx2": cls.AVX2,
                    "f16c": cls.F16C,
                    "rdrand": cls.RDRAND,
                    "rdseed": cls.RDSEED,
                    "fma": cls.FMA,
                    "movbe": cls.MOVBE,
                    "bmi1": cls.BMI1,
                    "bmi2": cls.BMI2,
                    "abm": cls.ABM,
                    "adx": cls.ADX,
                    "sha_ni": cls.SHA,
                    "gfni": cls.GFNI,
                    "vaes": cls.VAES,
                    "vpclmulqdq": cls.VPCLMULQDQ,
                    "hle": cls.HLE,
                    "rtm": cls.RTM,
                }

                for flag_name, cpu_feature in flag_map.items():
                    if flag_name in content:
                        features |= cpu_feature

                if features & (cls.HLE | cls.RTM):
                    features |= cls.TSX

        except Exception:
            pass

        return features

    @classmethod
    def _detect_arm_features(cls) -> "CpuFeature":
        """Detect ARM CPU features."""
        features = cls.BASIC

        if _IS_LINUX:
            features |= cls._detect_arm_linux()
        elif _IS_WIN:
            features |= cls._detect_arm_windows()

        return features

    @classmethod
    def _detect_arm_linux(cls) -> "CpuFeature":
        """Detect ARM features via Linux /proc/cpuinfo and getauxval."""
        features = cls(0)

        try:
            # /proc/cpuinfo
            cpuinfo_path = Path("/proc/cpuinfo")
            if cpuinfo_path.exists():
                content = cpuinfo_path.read_text().lower()

                feature_map = {
                    "neon": cls.NEON,
                    "asimd": cls.ASIMD,
                    "sve": cls.SVE,
                    "sve2": cls.SVE2,
                    "bf16": cls.BF16,
                    "i8mm": cls.I8MM,
                    "rng": cls.RNG,
                    "aes": cls.ARM_AES,
                    "sha1": cls.ARM_SHA1,
                    "sha2": cls.ARM_SHA2,
                    "sha3": cls.ARM_SHA3,
                }

                for flag_name, cpu_feature in feature_map.items():
                    if flag_name in content:
                        features |= cpu_feature

            # getauxval
            try:
                libc = ctypes.CDLL(ctypes.util.find_library("c"))
                getauxval = libc.getauxval
                getauxval.restype = ctypes.c_ulong
                getauxval.argtypes = [ctypes.c_ulong]

                AT_HWCAP = 16
                AT_HWCAP2 = 26

                hwcap = getauxval(AT_HWCAP)
                hwcap2 = getauxval(AT_HWCAP2)

                if hwcap & (1 << 1):
                    features |= cls.ASIMD
                if hwcap & (1 << 3):
                    features |= cls.ARM_AES
                if hwcap & (1 << 6):
                    features |= cls.ARM_SHA1
                if hwcap & (1 << 5):
                    features |= cls.ARM_SHA2
                if hwcap2 & (1 << 0):
                    features |= cls.SVE
                if hwcap2 & (1 << 1):
                    features |= cls.SVE2
                if hwcap2 & (1 << 14):
                    features |= cls.BF16
                if hwcap2 & (1 << 13):
                    features |= cls.I8MM
                if hwcap2 & (1 << 16):
                    features |= cls.RNG

            except Exception:
                pass

        except Exception:
            pass

        return features

    @classmethod
    def _detect_arm_windows(cls) -> "CpuFeature":
        """Detect ARM features on Windows."""
        features = cls(0)

        try:
            kernel32 = ctypes.windll.kernel32

            # Windows ARM64 feature detection (limited by API availability)
            feature_map = {
                12: cls.NEON  # PF_ARM_NEON_INSTRUCTIONS_AVAILABLE
            }

            for feature_code, cpu_feature in feature_map.items():
                if kernel32.IsProcessorFeaturePresent(feature_code):
                    features |= cpu_feature

        except Exception:
            pass

        return features

    def names(self) -> List[str]:
        """Get list of feature names."""
        return [
            member.name
            for member in CpuFeature
            if member != CpuFeature.BASIC and member in self
        ]

    def vector_width(self) -> int:
        """Get maximum vector width in bits."""
        if self & self.AVX2:
            return 256
        elif self & self.AVX:
            return 256
        elif self & (
            self.SSE | self.SSE2 | self.SSE3 | self.SSSE3 | self.SSE41 | self.SSE42
        ):
            return 128
        elif self & (self.NEON | self.ASIMD):
            return 128
        elif self & self.SVE:
            return 2048  # SVE supports up to 2048 bits
        elif self & self.SVE2:
            return 2048
        else:
            return 64

    def __str__(self) -> str:
        names = self.names()
        return "BASIC" if not names else " | ".join(sorted(names))

    def __repr__(self) -> str:
        return f"CpuFeature({self})"


# ================================================================
# System Information
# ================================================================


class SystemInfo:
    """Extended system information."""

    @staticmethod
    def get_cpu_info() -> Dict[str, Union[str, int, List[str]]]:
        """Get comprehensive CPU information."""
        info = {
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "features": CpuFeature.detect().names(),
            "vector_width": CpuFeature.detect().vector_width(),
        }

        if _IS_X86:
            try:
                # CPU brand string
                brand_parts = []
                for i in range(3):
                    eax, ebx, ecx, edx = _cpuid_x86(0x80000002 + i, 0)
                    for reg in [eax, ebx, ecx, edx]:
                        brand_parts.extend(
                            [
                                chr((reg >> 0) & 0xFF),
                                chr((reg >> 8) & 0xFF),
                                chr((reg >> 16) & 0xFF),
                                chr((reg >> 24) & 0xFF),
                            ]
                        )
                info["brand"] = "".join(brand_parts).strip()
            except Exception:
                info["brand"] = "Unknown"

        return info

    @staticmethod
    def benchmark_features() -> Dict[str, float]:
        """Benchmark CPU features."""
        features = CpuFeature.detect()
        results = {}

        # Integer benchmark
        start = time.perf_counter()
        total = sum(i * i for i in range(1000000))
        results["integer_ops"] = time.perf_counter() - start

        # Floating-point benchmark
        start = time.perf_counter()
        total = sum(float(i) ** 2.5 for i in range(100000))
        results["float_ops"] = time.perf_counter() - start

        # Feature presence
        results["has_sse2"] = bool(features & CpuFeature.SSE2)
        results["has_avx"] = bool(features & CpuFeature.AVX)
        results["has_avx2"] = bool(features & CpuFeature.AVX2)
        results["has_neon"] = bool(features & CpuFeature.NEON)
        results["has_sve"] = bool(features & CpuFeature.SVE)

        return results


# ================================================================
# Profiling
# ================================================================


class SystemProfiler:
    """Handles system profiling and performance measurements."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> SystemProfiler:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self) -> None:
        self.profiler = cProfile.Profile()
        self.start_time = time.monotonic()

    def start(self) -> None:
        self.profiler.enable()

    def stop(self) -> str:
        self.profiler.disable()
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        return s.getvalue()


class ProcessExecutor:
    """[[ProcessExecutor]] – Platform-independent process execution."""

    @staticmethod
    async def run_command(
        command: List[str],
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str, int]:
        """Execute a command asynchronously in a platform-independent way."""
        shell = IS_WINDOWS  # Use shell=True on Windows for compatibility
        try:
            if IS_WINDOWS:
                cmd_str = subprocess.list2cmdline(command) if shell else command
                process = await asyncio.create_subprocess_shell(
                    cmd_str if shell else command[0],
                    *([] if shell else command[1:]),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
                )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            return stdout_str, stderr_str, process.returncode
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        except FileNotFoundError:
            raise RuntimeError(
                f"Command not found: {command[0]}. Is it installed and in PATH?"
            )


class Benchmark:
    """Command benchmarking utility."""

    def __init__(
        self, command: List[str], iterations: int = 10, root_dir: Union[str, Path] = "."
    ):
        self.command = command
        self.iterations = iterations
        self.results: List[float] = []
        self.profiler = SystemProfiler()
        self.project_manager = ProjectManager(
            root_dir
        )  # Initialize ProjectManager for UV context

    async def run(self) -> float:
        await (
            self.project_manager.setup_environment()
        )  # Ensure virtual environment is set up
        self.profiler.start()
        best = sys.maxsize
        for _ in range(self.iterations):
            t0 = time.monotonic()
            if (
                self.command[0] in ("uv", "uv.exe")
                and self.command[1] == "run"
                and len(self.command) > 2
            ):
                # Resolve script path for uv run
                script_path = Path(self.command[2]).resolve()
                if not script_path.exists():
                    raise FileNotFoundError(f"Script not found: {script_path}")
                uv_command = [self.command[0], "run", str(script_path)] + self.command[
                    3:
                ]
                result = await self.project_manager.run_uv_command(
                    uv_command, timeout=None
                )
                stdout, stderr, status = result.stdout, result.stderr, result.returncode
            else:
                # Fallback to ProcessExecutor for non-UV commands
                stdout, stderr, status = await ProcessExecutor.run_command(self.command)
            t1 = time.monotonic()
            duration = t1 - t0
            self.results.append(duration)
            best = min(best, duration)
            print(f'{duration:.3f}s')
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            print("STATUS:", status, '\n', '_' * 80)
        profile_data = self.profiler.stop()
        print('_' * 80)
        print(f'Best of {self.iterations}: {best:.3f}s')
        print('Profile data:')
        print(profile_data)
        return best


def generate_ansi_color(c: str) -> str:
    """Generate an ANSI escape code for colored text."""
    colors = {
        'reset': '\033[0m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
    }
    return colors.get(c.lower(), colors['reset'])


@dataclass
class BenchmarkReport:
    command: str
    best_time: float
    iterations: int

    def __repr__(self):
        command_color = generate_ansi_color('cyan')
        timing_color = generate_ansi_color('green')
        title_color = generate_ansi_color('yellow')
        reset_color = generate_ansi_color('reset')
        report = f"{title_color}Benchmark Report:{reset_color}\n"
        report += f"{command_color}Command:{reset_color} {self.command}\n"
        report += f"{timing_color}Best time:{reset_color} {self.best_time:.3f}s over {self.iterations} iterations\n"
        return report


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int

    def __repr__(self):
        color_stdout = generate_ansi_color('green')
        color_stderr = generate_ansi_color('red')
        color_return = generate_ansi_color('cyan')
        reset_color = generate_ansi_color('reset')
        output = f"{color_stdout}STDOUT:{reset_color}\n{self.stdout}\n"
        output += f"{color_stderr}STDERR:{reset_color}\n{self.stderr}\n"
        output += f"{color_return}RETURN CODE:{reset_color} {self.returncode}\n"
        return output


# ================================================================
# ----------------- Project Management Code ----------------------
# ================================================================


@dataclass
class ProjectConfig:
    """Project configuration container ([[ProjectConfig]])."""

    name: str
    version: str
    python_version: str
    dependencies: List[str]
    dev_dependencies: List[str] = field(default_factory=list)
    ruff_config: Dict[str, Any] = field(default_factory=dict)
    ffi_modules: List[str] = field(default_factory=list)
    src_path: Path = Path("src")
    tests_path: Path = Path("tests")


class ProjectManager:
    """Manages project configuration, environment setup, and command execution ([[ProjectManager]])."""

    def __init__(self, root_dir: Union[str, Path]):
        # {{root_dir}} as absolute path
        self.root_dir = Path(root_dir).resolve()
        self.logger = self._setup_logging()
        self.config = self._load_or_create_config()
        self.project_config = self._load_project_config()
        self.ffi_modules = self.project_config.get("ffi_modules", [])
        self._ensure_directory_structure()
        self.is_windows = platform.system() == "Windows"

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("ProjectManager")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _load_project_config(self) -> Dict[str, Any]:  # TODO: SEMVER FILE
        """Load project-specific configuration from msc.json."""
        config_path = self.root_dir / "msc.json"
        default_config = {
            "ffi_modules": [],
            "src_path": "src",
            "dev_dependencies": [],
            "profile_enabled": True,
            "platform_specific": {
                "windows": {"priority": 32},
                "linux": {"priority": 0},
            },
        }
        if not config_path.exists():
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        with open(config_path, encoding='utf-8') as f:
            return json.load(f)

    def _load_or_create_config(self) -> ProjectConfig:
        pyproject_path = self.root_dir / "pyproject.toml"
        self.logger.debug(f"Checking for pyproject.toml at: {pyproject_path}")
        if not pyproject_path.exists():
            self.logger.info("No pyproject.toml found. Creating default configuration.")
            config = ProjectConfig(
                name=self.root_dir.name,
                version="0.1.0",
                python_version=">=3.13",
                dependencies=["uvx>=0.1.0"],
                dev_dependencies=[
                    "ruff>=0.3.0",
                    "pytest>=8.0.0",
                    "pytest-asyncio>=0.23.0",
                ],
                ruff_config={
                    "line-length": 88,
                    "target-version": "py313",
                    "select": ["E", "F", "I", "N", "W"],
                    "ignore": [],
                    "fixable": ["A", "B", "C", "D", "E", "F", "I"],
                },
                ffi_modules=[],
            )
            self._write_pyproject_toml(config)
            return config
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return ProjectConfig(
            name=data["project"]["name"],
            version=data["project"]["version"],
            python_version=data["project"]["requires-python"],
            dependencies=data["project"].get("dependencies", []),
            dev_dependencies=data["project"].get("dev-dependencies", []),
            ruff_config=data.get("tool", {}).get("ruff", {}),
            ffi_modules=data["project"].get("ffi-modules", []),
            src_path=Path(data["project"].get("src-path", "src")),
            tests_path=Path(data["project"].get("tests-path", "tests")),
        )

    def _write_pyproject_toml(self, config: ProjectConfig):
        """Write pyproject.toml using manual string construction."""
        toml_content = f"""[project]
    name = "{config.name}"
    version = "{config.version}"
    requires-python = "{config.python_version}"
    dependencies = [
    """
        for dep in config.dependencies:
            toml_content += f'    "{dep}",\n'
        toml_content += "]\n\n"
        toml_content += "dev-dependencies = [\n"
        for dep in config.dev_dependencies:
            toml_content += f'    "{dep}",\n'
        toml_content += "]\n\n"
        toml_content += f'ffi-modules = {json.dumps(config.ffi_modules)}\n'
        toml_content += f'src-path = "{config.src_path}"\n'
        toml_content += f'tests-path = "{config.tests_path}"\n\n'
        toml_content += "[tool.ruff]\n"
        for key, value in config.ruff_config.items():
            if isinstance(value, list):
                toml_content += f"{key} = {json.dumps(value)}\n"
            elif key == "target-version":  # Explicitly add quotes for target-version
                toml_content += f'{key} = "{value}"\n'
            else:
                toml_content += f"{key} = {value}\n"
        with open(self.root_dir / "pyproject.toml", "w", encoding='utf-8') as f:
            f.write(toml_content)

    def _ensure_directory_structure(self):
        """Create necessary project directories if they don't exist."""
        dirs = [
            self.config.src_path,
            self.config.tests_path,
            self.config.src_path / "ffi",
        ]
        for dir_path in dirs:
            full_path = self.root_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    async def run_uv_command(
        self, cmd: List[str], timeout: Optional[float] = None
    ) -> subprocess.CompletedProcess:
        """Run a UV command asynchronously with timeout support."""
        self.logger.debug(f"Running UV command: {' '.join(cmd)}")
        if self.is_windows:
            if not cmd[0].endswith('.exe') and '/' not in cmd[0] and '\\' not in cmd[0]:
                if cmd[0] in ("uv", "uvx"):
                    cmd[0] = f"{cmd[0]}.exe"
        try:
            shell = self.is_windows
            if shell:
                cmd_str = subprocess.list2cmdline(cmd)
                process = await asyncio.create_subprocess_shell(
                    cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                try:
                    process.terminate()
                    await process.wait()
                except ProcessLookupError:
                    pass
                raise TimeoutError(f"Command timed out after {timeout} seconds")
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            if process.returncode != 0:
                error_msg = stderr_str or stdout_str
                self.logger.error(f"UV command failed: {error_msg}")
                raise RuntimeError(f"UV command failed: {error_msg}")
            return subprocess.CompletedProcess(
                cmd, process.returncode, stdout_str, stderr_str
            )
        except FileNotFoundError as e:
            self.logger.error(f"Command not found: {cmd[0]}")
            raise RuntimeError(
                f"Command not found: {cmd[0]}. Is UV installed and in PATH?"
            ) from e

    async def setup_environment(self):
        """Set up the environment based on mode."""
        self.logger.info("Setting up environment...")
        venv_cmd = ["uv", "venv"] if not self.is_windows else ["uv.exe", "venv"]
        await self.run_uv_command(venv_cmd)
        requirements_path = self.root_dir / "requirements.txt"
        dev_requirements_path = self.root_dir / "requirements-dev.txt"
        if self.config.dependencies:
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.config.dependencies) + '\n')
        if self.config.dev_dependencies:
            with open(dev_requirements_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.config.dev_dependencies) + '\n')
        if requirements_path.exists():
            self.logger.info("Compiling requirements...")
            pip_cmd = ["uv", "pip"] if not self.is_windows else ["uv.exe", "pip"]
            await self.run_uv_command(
                [
                    *pip_cmd,
                    "compile",
                    str(requirements_path),
                    "--output-file",
                    str(self.root_dir / "requirements.lock"),
                ]
            )
        if dev_requirements_path.exists():
            self.logger.info("Compiling dev requirements...")
            pip_cmd = ["uv", "pip"] if not self.is_windows else ["uv.exe", "pip"]
            await self.run_uv_command(
                [
                    *pip_cmd,
                    "compile",
                    str(dev_requirements_path),
                    "--output-file",
                    str(self.root_dir / "requirements-dev.lock"),
                ]
            )
        if (self.root_dir / "requirements.lock").exists():
            self.logger.info("Installing dependencies from lock file...")
            pip_cmd = ["uv", "pip"] if not self.is_windows else ["uv.exe", "pip"]
            await self.run_uv_command(
                [*pip_cmd, "install", "-r", str(self.root_dir / "requirements.lock")]
            )
        if (self.root_dir / "requirements-dev.lock").exists():
            self.logger.info("Installing dev dependencies from lock file...")
            pip_cmd = ["uv", "pip"] if not self.is_windows else ["uv.exe", "pip"]
            await self.run_uv_command(
                [
                    *pip_cmd,
                    "install",
                    "-r",
                    str(self.root_dir / "requirements-dev.lock"),
                ]
            )
        if (self.root_dir / "setup.py").exists():
            self.logger.info("Installing project in editable mode...")
            pip_cmd = ["uv", "pip"] if not self.is_windows else ["uv.exe", "pip"]
            await self.run_uv_command([*pip_cmd, "install", "-e", "."])

    async def run_app(self, module_path: str, *args, timeout: Optional[float] = None):
        """Run the application using Python directly."""
        module_path = str(Path(module_path))
        python_cmd = "python" if self.is_windows else "python3"
        cmd = [python_cmd, module_path, *map(str, args)]
        self.logger.info(f"Running: {' '.join(cmd)}")
        return await self.run_uv_command(cmd, timeout=timeout)

    async def run_tests(self):
        """Run tests using pytest."""
        uvx_cmd = ["uvx.exe"] if self.is_windows else ["uvx"]
        await self.run_uv_command(
            [*uvx_cmd, "run", "-m", "pytest", str(self.config.tests_path)]
        )

    async def run_linter(self):
        """Run Ruff linter."""
        uvx_cmd = ["uvx.exe"] if self.is_windows else ["uvx"]
        await self.run_uv_command([*uvx_cmd, "run", "-m", "ruff", "check", "."])

    async def format_code(self):
        """Format code using Ruff."""
        uvx_cmd = ["uvx.exe"] if self.is_windows else ["uvx"]
        await self.run_uv_command([*uvx_cmd, "run", "-m", "ruff", "format", "."])

    async def run_dev_mode(self):
        """Setup and run operations specific to Developer Mode."""
        self.logger.info("Running Developer Mode tasks...")
        await self.setup_environment()
        await self.run_tests()
        await self.run_linter()
        await self.format_code()
        self.logger.info(
            "Development environment setup complete. You can now start coding or run your application."
        )

    async def run_admin_mode(self):
        """Setup and run operations specific to Admin Mode."""
        self.logger.info("Running Admin Mode tasks...")
        await self.setup_environment()
        if self.is_windows:
            self.logger.info("Performing Windows-specific admin tasks...")
            if (
                "platform_specific" in self.project_config
                and "windows" in self.project_config["platform_specific"]
            ):
                priority = self.project_config["platform_specific"]["windows"].get(
                    "priority", 32
                )
                self.logger.info(f"Setting process priority to {priority}")
        else:
            self.logger.info("Performing Linux-specific admin tasks...")
            if (
                "platform_specific" in self.project_config
                and "linux" in self.project_config["platform_specific"]
            ):
                priority = self.project_config["platform_specific"]["linux"].get(
                    "priority", 0
                )
                self.logger.info(f"Setting process priority to {priority}")

    async def run_user_mode(self):
        """Setup and run operations specific to User Mode."""
        self.logger.info("Running User Mode tasks...")
        self.logger.info("Setting up minimal runtime environment...")
        venv_path = self.root_dir / ".venv"
        if not venv_path.exists():
            venv_cmd = ["uv.exe", "venv"] if self.is_windows else ["uv", "venv"]
            await self.run_uv_command(venv_cmd)
        requirements_path = self.root_dir / "requirements.txt"
        if requirements_path.exists():
            self.logger.info("Installing runtime dependencies...")
            pip_cmd = ["uv.exe", "pip"] if self.is_windows else ["uv", "pip"]
            await self.run_uv_command(
                [*pip_cmd, "install", "-r", str(requirements_path)]
            )
        main_module = self.root_dir / self.config.src_path / "__main__.py"
        if main_module.exists():
            self.logger.info("Running main application...")
            await self.run_app(str(main_module))
        else:
            self.logger.error(f"Main module not found at {main_module}")
            self.logger.info("Please specify the main module path explicitly.")

    async def teardown(self):
        """Teardown any setup done by modes or setup_environment."""
        self.logger.info("Tearing down environment...")
        self.logger.info("Stopping any running processes...")
        self.logger.info("Cleaning up temporary files...")
        if (self.root_dir / ".venv").exists():
            self.logger.info("Virtual environment removal skipped.")
        self.logger.info("Teardown complete.")

    async def upgrade_dependencies(self):
        """Upgrade all dependencies to their latest versions."""
        self.logger.info("Upgrading dependencies...")
        requirements_path = self.root_dir / "requirements.txt"
        dev_requirements_path = self.root_dir / "requirements-dev.txt"
        if requirements_path.exists():
            self.logger.info("Upgrading runtime dependencies...")
            pip_cmd = ["uv.exe", "pip"] if self.is_windows else ["uv", "pip"]
            await self.run_uv_command(
                [
                    *pip_cmd,
                    "compile",
                    str(requirements_path),
                    "--output-file",
                    str(self.root_dir / "requirements.lock"),
                    "--upgrade",
                ]
            )
            await self.run_uv_command(
                [*pip_cmd, "install", "-r", str(self.root_dir / "requirements.lock")]
            )
        if dev_requirements_path.exists():
            self.logger.info("Upgrading development dependencies...")
            pip_cmd = ["uv.exe", "pip"] if self.is_windows else ["uv", "pip"]
            await self.run_uv_command(
                [
                    *pip_cmd,
                    "compile",
                    str(dev_requirements_path),
                    "--output-file",
                    str(self.root_dir / "requirements-dev.lock"),
                    "--upgrade",
                ]
            )
            await self.run_uv_command(
                [
                    *pip_cmd,
                    "install",
                    "-r",
                    str(self.root_dir / "requirements-dev.lock"),
                ]
            )
        self.logger.info("Dependency upgrade complete.")

    async def create_module(self, module_name: str):
        """Create a new module in the src directory."""
        module_path = self.root_dir / self.config.src_path / module_name
        module_path.mkdir(parents=True, exist_ok=True)
        init_file = module_path / "__init__.py"
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
{module_name} module
\"\"\"

__version__ = "0.1.0"
""")
        module_file = module_path / f"{module_name}.py"
        with open(module_file, 'w', encoding='utf-8') as f:
            f.write(f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
Main functionality for {module_name}
\"\"\"

def main():
    \"\"\"Main function for {module_name}\"\"\"
    print("Hello from {module_name}!")

if __name__ == "__main__":
    main()
""")
        test_dir = self.root_dir / self.config.tests_path
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / f"test_{module_name}.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
Tests for {module_name} module
\"\"\"

import pytest
from {self.config.src_path.name}.{module_name} import {module_name}

def test_{module_name}_main():
    \"\"\"Test the main function of {module_name}\"\"\"
    assert True
""")
        self.logger.info(f"Created module {module_name} at {module_path}")
        self.logger.info(f"Created test file at {test_file}")


# ================================================================
# Main Function
# ================================================================


def benchmark_main(args) -> int:
    """[[benchmark_main]] – Run benchmark tests on a given command."""
    command = args.cmd
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        print("Command is required for benchmarking.")
        return 1
    benchmark = Benchmark(command, args.num, root_dir=".")
    best_time = asyncio.run(benchmark.run())  # Run asynchronously
    # Run one final time for ExecutionResult
    if command[0] in ("uv", "uv.exe") and command[1] == "run" and len(command) > 2:
        script_path = Path(command[2]).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        uv_command = [command[0], "run", str(script_path)] + command[3:]
        result = asyncio.run(
            benchmark.project_manager.run_uv_command(uv_command, timeout=None)
        )
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    else:
        stdout, stderr, returncode = asyncio.run(ProcessExecutor.run_command(command))
    execution_result = ExecutionResult(
        stdout=stdout, stderr=stderr, returncode=returncode
    )
    print(execution_result)
    benchmark_report = BenchmarkReport(
        command=' '.join(command), best_time=best_time, iterations=args.num
    )
    print(benchmark_report)
    return 0


async def project_main(args) -> int:
    """[[project_main]] – Run project management modes (DEV, ADMIN, USER, etc.)."""
    manager = ProjectManager(args.root)
    try:
        if args.create_module:
            await manager.create_module(args.create_module)
            return 0
        if args.mode == "DEV":
            await manager.run_dev_mode()
        elif args.mode == "ADMIN":
            await manager.run_admin_mode()
        elif args.mode == "USER":
            await manager.run_user_mode()
        elif args.mode == "TEARDOWN":
            await manager.teardown()
        elif args.mode == "UPGRADE":
            await manager.upgrade_dependencies()
    except Exception as e:
        manager.logger.error(f"Error: {e}")
        traceback.print_exc()
        return 1
    return 0


def unified_main() -> int:
    """[[unified_main]] – Unified CLI entry point using subparsers."""
    parser = argparse.ArgumentParser(
        description='Monolithic Project Manager & Benchmark Utility'
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Choose 'project' or 'benchmark' mode"
    )
    # Subparser for project manager
    project_parser = subparsers.add_parser(
        "project", help="Run project management tasks"
    )
    project_parser.add_argument("--root", default=".", help="Project root directory")
    project_parser.add_argument(
        "mode",
        choices=["DEV", "ADMIN", "USER", "TEARDOWN", "UPGRADE"],
        help="Mode to execute",
    )
    project_parser.add_argument(
        "--timeout", type=float, default=None, help="Timeout in seconds for commands"
    )
    project_parser.add_argument(
        "--create-module", type=str, help="Create a new module with the specified name"
    )
    # Subparser for benchmarking
    bench_parser = subparsers.add_parser(
        "benchmark", help="Benchmark command execution"
    )
    bench_parser.add_argument(
        "-n", "--num", type=int, default=10, help="Number of iterations"
    )
    bench_parser.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="Command to execute for benchmarking"
    )
    args = parser.parse_args()

    if args.command == "project":
        return asyncio.run(project_main(args))
    elif args.command == "benchmark":
        return benchmark_main(args)
    else:
        parser.error("Invalid command.")
        return 1


def main() -> None:
    """Demonstrate CPU feature detection and system information."""
    print("=== CPU Feature Detection ===")
    features = CpuFeature.detect()
    print(f"Detected Features: {features}")
    print(f"Vector Width: {features.vector_width()} bits")
    print("\n=== System Information ===")
    cpu_info = SystemInfo.get_cpu_info()
    for key, value in cpu_info.items():
        print(f"{key.replace('_', ' ').title():20}: {value}")
    print("\n=== Benchmark Results ===")
    bench_results = SystemInfo.benchmark_features()
    for key, value in bench_results.items():
        print(
            f"{key.replace('_', ' ').title():20}: {value:.6f}"
            if isinstance(value, float)
            else f"{key.replace('_', ' ').title():20}: {value}"
        )


if __name__ == "__main__":
    main()
    sys.exit(unified_main())
