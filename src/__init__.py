#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import gc
import os
import dis
import sys
import ast
import time
import site
import mmap
import json
import uuid
import math
import cmath
import shlex
import socket
import struct
import shutil
import pickle
import ctypes
import pstats
import weakref
import logging
import tomllib
import pathlib
import asyncio
import inspect
import hashlib
import cProfile
import argparse
import tempfile
import platform
import traceback
import functools
import linecache
import importlib
import threading
import subprocess
import tracemalloc
import http.server
import collections
import http.client
import http.server
import socketserver
from array import array
from io import StringIO
from pathlib import Path
from math import sqrt, pi
from datetime import datetime
from queue import Queue, Empty
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto, IntEnum, StrEnum, Flag
from collections import defaultdict, deque, namedtuple
from functools import reduce, lru_cache, partial, wraps
from contextlib import contextmanager, asynccontextmanager
from importlib.util import spec_from_file_location, module_from_spec
from types import SimpleNamespace, ModuleType,  MethodType, FunctionType, CodeType, TracebackType, FrameType
from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Set, OrderedDict,
    Coroutine, Type, NamedTuple, ClassVar, Protocol, runtime_checkable, AsyncIterator, Iterator
)

class ProcessExecutor:
    """Platform-independent process execution"""
    @staticmethod
    def _windows_run_command(command, timeout, env):
        from ctypes import windll, wintypes
        
        # Optimize process priority - using Windows ABOVE_NORMAL_PRIORITY_CLASS
        def set_process_priority():
            windll.kernel32.SetPriorityClass(
                wintypes.HANDLE(-1), 
                0x00008000  # ABOVE_NORMAL_PRIORITY_CLASS
            )

        def wrun_command(command, timeout=None, env=None):
            # Increase buffer sizes for better performance
            BUFFER_SIZE = 65536  # 64KB buffer
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                shell=True,
                env=env,
                bufsize=BUFFER_SIZE  # Set larger buffer
            )
            
            # Set higher priority for the subprocess
            set_process_priority()
            
            # Use memoryview for zero-copy buffering
            def read_stream(stream):
                buffer = []
                while True:
                    chunk = stream.read1(BUFFER_SIZE)
                    if not chunk:
                        break
                    buffer.append(chunk)
                return b''.join(buffer).decode()
                
            stdout = read_stream(process.stdout)
            stderr = read_stream(process.stderr)
            
            return_code = process.wait(timeout=timeout)
            return stdout, stderr, return_code

        try:
            stdout, stderr, status = wrun_command(command, timeout=timeout, env=env)
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            print("STATUS:", status, '\n', '_' * 80)
            return stdout, stderr, status
        except TimeoutError as e:
            print(e)
            raise
        except Exception as e:
            print(e)
            raise
    @staticmethod
    def _posix_run_command(command, timeout, env):
        import resource
        
        # Set process priority using nice value (-20 to 19, lower is higher priority)
        def set_process_priority():
            try:
                os.nice(-10)  # Higher priority but not maximum
            except PermissionError:
                pass
                
        def run_command(command, timeout=None, env=None):
            BUFFER_SIZE = 65536  # 64KB buffer
            
            # Set resource limits for better performance
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                shell=True,
                env=env,
                bufsize=BUFFER_SIZE,
                preexec_fn=set_process_priority
            )
            
            # Use memoryview for efficient reading
            stdout, stderr = process.communicate(timeout=timeout)
            return stdout.decode(), stderr.decode(), process.returncode
        try:
            stdout, stderr, status = run_command(command, timeout=timeout, env=env)
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            print("STATUS:", status, '\n', '_' * 80)
            return stdout, stderr, status  # Add return statement
        except TimeoutError as e:
            print(e)
            raise
        except Exception as e:
            print(e)
            raise
    @staticmethod
    def run_command(command: List[str], timeout: Optional[float] = None, 
                   env: Optional[Dict[str, str]] = None) -> Tuple[str, str, int]:
        """Platform-independent command execution"""
        if IS_WINDOWS:
            return ProcessExecutor._windows_run_command(command, timeout, env)
        return ProcessExecutor._posix_run_command(command, timeout, env)

IS_WINDOWS = sys.platform == "win32"
_IS_LINUX = sys.platform.startswith("linux")
_MACHINE = platform.machine().upper()
_IS_X86 = _MACHINE in ("X86_64", "AMD64", "I386", "I686")
_IS_ARM = _MACHINE.startswith(("ARM64", "AARCH64"))
profiler = cProfile.Profile()
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

class SystemProfiler:
    """Handles system profiling and performance measurements"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'SystemProfiler':
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

class Benchmark:
    """Command benchmarking utility"""
    def __init__(self, command: List[str], iterations: int = 10):
        self.command = command
        self.iterations = iterations
        self.results: List[float] = []
        self.profiler = SystemProfiler()
    def run(self) -> float:
        self.profiler.start()
        best = sys.maxsize
        for _ in range(self.iterations):
            t0 = time.monotonic()
            ProcessExecutor.run_command(self.command)
            t1 = time.monotonic()
            duration = t1 - t0
            self.results.append(duration)
            best = min(best, duration)
            print(f'{duration:.3f}s')
        profile_data = self.profiler.stop()
        print('_' * 80)
        print(f'Best of {self.iterations}: {best:.3f}s')
        print('Profile data:')
        print(profile_data)
        return best

def generate_ansi_color(c: str) -> str:
    """Generate an ANSI escape code for colored text"""
    colors = {
        'reset': '\033[0m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m'
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

def main() -> int:
    parser = argparse.ArgumentParser(description='Benchmark command execution')
    parser.add_argument('-n', '--num', type=int, default=10,
                        help="Number of iterations")
    parser.add_argument('cmd', nargs=argparse.REMAINDER, help="Command to execute")
    args = parser.parse_args()
    
    if not args.cmd:
        parser.error("Command is required")

    # Remove the '--' separator if it exists
    if args.cmd[0] == '--':
        command = args.cmd[1:]
    else:
        command = args.cmd

    # Simulate benchmark execution
    # For demonstration, using fake time measurements; replace with actual benchmarking logic
    benchmark = Benchmark(command, args.num)
    best_time = benchmark.run()  # this should return the best time from the Benchmark class


    # Demonstrate execution result output
    stdout, stderr, returncode = ProcessExecutor.run_command(command)  # this should be a real execution
    execution_result = ExecutionResult(stdout=stdout, stderr=stderr, returncode=returncode)
    # print(execution_result.__dict__)

    benchmark_report = BenchmarkReport(command=' '.join(command), best_time=best_time, iterations=args.num)
    print(benchmark_report)

    return 0

if __name__ == "__main__":
  # Try:
  # python topinit.py -- python -c "print('hello')"
  # python topinit.py -- python src/__init__.py arg1  
  sys.exit(main())
