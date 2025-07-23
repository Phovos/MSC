"""
(MSC) Morphological Source Code Framework – V0.0.12
================================================================================
<https://github.com/Phovos/msc> • MSC: Morphological Source Code © 2025 by Phovos
--------------------------------------------------------------------------------
"""

import sys
import os
import platform
import ctypes
from enum import IntFlag, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple


# Platform detection constants
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform.startswith('linux')
IS_64BIT = sys.maxsize > 2**32


class ProcessorFeatures(IntFlag):
    """Extensible processor feature detection."""

    BASIC = auto()
    SSE = auto()
    SSE2 = auto()
    SSE3 = auto()
    SSSE3 = auto()
    SSE41 = auto()
    SSE42 = auto()
    AVX = auto()
    AVX2 = auto()
    FMA = auto()
    NEON = auto()
    # ... add more as needed ...

    @classmethod
    def detect_features(cls) -> 'ProcessorFeatures':
        """Detect available processor features across Windows and Linux."""
        if hasattr(cls, '_cached_features') and cls._cached_features is not None:
            return cls._cached_features
        features = cls.BASIC
        machine = platform.machine().lower()
        try:
            if machine in ('x86_64', 'amd64', 'x86', 'i386', 'i686'):
                features |= cls._detect_x86()
            elif machine.startswith(('arm', 'aarch')):
                features |= cls._detect_arm()
        except Exception:
            pass
        cls._cached_features = features
        return features

    @classmethod
    def _detect_x86(cls) -> 'ProcessorFeatures':
        f = cls.BASIC
        if IS_WINDOWS:
            try:
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                checks = {
                    cls.SSE: 6,
                    cls.SSE2: 10,
                    cls.SSE3: 13,
                    cls.SSSE3: 36,
                    cls.SSE41: 37,
                    cls.SSE42: 38,
                    cls.AVX: 39,
                    cls.AVX2: 40,
                }
                for feat, code in checks.items():
                    if kernel32.IsProcessorFeaturePresent(code):
                        f |= feat
            except Exception:
                pass
        elif IS_LINUX:
            try:
                with open('/proc/cpuinfo') as fp:
                    flags = ' '.join(fp.read().splitlines())
                for flag, feat in [
                    ('sse', cls.SSE),
                    ('sse2', cls.SSE2),
                    ('sse3', cls.SSE3),
                    ('ssse3', cls.SSSE3),
                    ('sse4_1', cls.SSE41),
                    ('sse4_2', cls.SSE42),
                    ('avx', cls.AVX),
                    ('avx2', cls.AVX2),
                    ('fma', cls.FMA),
                ]:
                    if flag in flags:
                        f |= feat
            except Exception:
                pass
        return f

    @classmethod
    def _detect_arm(cls) -> 'ProcessorFeatures':
        f = cls.BASIC
        if IS_WINDOWS:
            f |= cls.NEON
        elif IS_LINUX:
            try:
                with open('/proc/cpuinfo') as fp:
                    info = fp.read().lower()
                if 'neon' in info or 'asimd' in info:
                    f |= cls.NEON
            except Exception:
                pass
        return f

    def names(self) -> List[str]:
        return [
            feat.name
            for feat in ProcessorFeatures
            if feat != ProcessorFeatures.BASIC and feat in self
        ]

    def __str__(self):
        names = self.names()
        return ' | '.join(names) if names else 'BASIC'


# Ensure class var exists
ProcessorFeatures._cached_features = None


@dataclass
class PlatformInfo:
    system: str
    release: str
    version: str
    architecture: str
    processor: str
    python_version: str
    is_64bit: bool
    processor_features: ProcessorFeatures
    extra: Dict[str, Any] = field(default_factory=dict)


class PlatformInterface:
    """Base class for Windows and Linux."""

    def __init__(self):
        self.info = self._gather_info()

    def _gather_info(self) -> PlatformInfo:
        base = PlatformInfo(
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            architecture=platform.machine(),
            processor=platform.processor(),
            python_version=platform.python_version(),
            is_64bit=IS_64BIT,
            processor_features=ProcessorFeatures.detect_features(),
        )
        self._add_extra_info(base.extra)
        return base

    def _add_extra_info(self, extra: Dict[str, Any]):
        """Add platform-specific details."""
        pass

    def load_c_library(self) -> Optional[ctypes.CDLL]:
        raise NotImplementedError

    def get_memory_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    def execute(self, cmd: List[str]) -> Tuple[int, str, str]:
        import subprocess

        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return proc.returncode, proc.stdout, proc.stderr


class WindowsPlatform(PlatformInterface):
    def load_c_library(self) -> Optional[ctypes.CDLL]:
        for lib in ("msvcrt.dll", "kernel32.dll"):
            try:
                return ctypes.CDLL(lib)
            except OSError:
                continue
        return None

    def _add_extra_info(self, extra: Dict[str, Any]):
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            )
            extra['product_name'] = winreg.QueryValueEx(key, "ProductName")[0]
        except Exception:
            pass

    def get_memory_info(self) -> Dict[str, Any]:
        try:

            class MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint),
                    ("memLoad", ctypes.c_uint),
                    ("totalPhys", ctypes.c_ulonglong),
                    ("availPhys", ctypes.c_ulonglong),
                    ("totalPageFile", ctypes.c_ulonglong),
                    ("availPageFile", ctypes.c_ulonglong),
                    ("totalVirtual", ctypes.c_ulonglong),
                    ("availVirtual", ctypes.c_ulonglong),
                    ("availExtendedVirtual", ctypes.c_ulonglong),
                ]

            ms = MEMSTAT()
            ms.length = ctypes.sizeof(ms)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return {
                "total_phys": ms.totalPhys,
                "avail_phys": ms.availPhys,
                "total_virt": ms.totalVirtual,
                "avail_virt": ms.availVirtual,
                "mem_load_pct": ms.memLoad,
            }
        except Exception:
            return {}


class LinuxPlatform(PlatformInterface):
    def load_c_library(self) -> Optional[ctypes.CDLL]:
        for lib in ("libc.so.6", "libc.so"):
            try:
                return ctypes.CDLL(lib)
            except OSError:
                continue
        return None

    def _add_extra_info(self, extra: Dict[str, Any]):
        release_path = "/etc/os-release"
        if os.path.exists(release_path):
            try:
                with open(release_path) as f:
                    for line in f:
                        if "=" in line:
                            k, v = line.rstrip().split("=", 1)
                            extra[k] = v.strip('"')
            except Exception:
                pass

    def get_memory_info(self) -> Dict[str, Any]:
        info = {}
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
        except Exception:
            pass
        return info


class PlatformFactory:
    @staticmethod
    def create() -> PlatformInterface:
        if IS_WINDOWS:
            return WindowsPlatform()
        elif IS_LINUX:
            return LinuxPlatform()
        else:
            return PlatformInterface()


# Example usage:
if __name__ == "__main__":
    plat = PlatformFactory.create()
    print(plat.info)
    print("C library loaded:", bool(plat.load_c_library()))
    print("Memory info sample:", plat.get_memory_info())
