#!/usr/bin/env python3
"""
CPU feature detection for x86-64 & ARM64 on our platforms, Win11 and debian/ubuntu.
Supports Python 3.12+ with full feature detection + testing for Ryzen 5600 and GitHub CI.
"""

from __future__ import annotations

import ctypes
import platform
import sys
import time
from enum import Flag, auto
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Platform detection
_IS_WIN = sys.platform == "win32"
_IS_LINUX = sys.platform.startswith("linux")
_MACHINE = platform.machine().upper()
_IS_X86 = _MACHINE in ("X86_64", "AMD64", "I386", "I686")
_IS_ARM = _MACHINE.startswith(("ARM64", "AARCH64"))

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
# Main Function
# ================================================================


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
