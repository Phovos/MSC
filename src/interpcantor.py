#!/usr/bin/env -S uv run
# /* script
# requires-python = ">=3.14"
# dependencies = [
#     "uv==*.*",
# ]
# -*- coding: utf-8 -*-
#------------------------------
# 3.14 std libs **ONLY**      |
# Platform(s):                |
# Win11 (production)          |
# Ubuntu-22.04 (dev, staging) |
#------------------------------
# <a href="https://github.com/MOONLAPSED/Pleroma">Morphological Source Code</a> © 2025 by Moonlapsed:MOONLAPSED@GMAIL.COM CC BY; SEE LICENCE
# Engineering + Pedagogy script, not client code
#------------------------------------------------------------------------------
import os
import time
import ast
import secrets
import hmac
import sys
import re
import platform
import ctypes
from enum import IntEnum, IntFlag, auto
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor # For managing sub-interpreter threads
from concurrent.interpreters import create, RunFailedError
import mmap, ctypes, ast, tokenize, io, struct

# Configure main interpreter logging
logging.basicConfig(level=logging.INFO,
                    format='[Main-%(process)d] %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- SharedMemoryChannel definition (copied for self-containment in main) ---
# This is needed in the main interpreter to manage the shared memory directly
# and potentially to create the initial channel objects if not passing raw addresses.
# For this example, we'll use it to directly interact with the buffer for verification.
MESSAGE_HEADER_FORMAT = "!I32s"
MESSAGE_HEADER_SIZE = struct.calcsize(MESSAGE_HEADER_FORMAT)

class SharedMemoryChannel:
    def __init__(self, buffer: mmap.mmap, region_offset: int, region_size: int,
                 peer_offset: int, peer_size: int, buffer_address: int):
        self._buffer = buffer
        self._region_offset = region_offset
        self._region_size = region_size
        self._peer_offset = peer_offset
        self._peer_size = peer_size
        self._buffer_address = buffer_address

        self._shm = (ctypes.c_char * self._region_size).from_address(self._buffer_address + self._region_offset)
        self._peer_shm = (ctypes.c_char * self._peer_size).from_address(self._buffer_address + self._peer_offset)

        self._key = None

    def _wait_for_peer_signal(self, offset: int, expected_value: bytes, timeout: float = 5.0):
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            if self._peer_shm[offset:offset + len(expected_value)] == expected_value:
                return True
            time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for peer signal at offset {offset}")

    def establish_dh_key(self, P: int, G: int, timeout: float = 5.0):
        logger.info("Establishing DH key...")
        priv = secrets.randbelow(P - 2) + 1
        pub = pow(G, priv, P)
        pub_bytes = pub.to_bytes(32, 'big')

        self._shm[:32] = pub_bytes
        self._shm[32] = b'\x01'

        self._wait_for_peer_signal(32, b'\x01', timeout=timeout)
        their_pub = int.from_bytes(bytes(self._peer_shm[:32]), 'big')

        shared = pow(their_pub, priv, P)
        self._key = hashlib.sha256(shared.to_bytes((shared.bit_length() + 7) // 8 or 1, 'big')).digest()
        logger.info("DH key established.")
        return self._key

    def send_message(self, message: bytes, timeout: float = 5.0):
        if self._key is None:
            raise RuntimeError("Key not established. Call establish_dh_key first.")

        mac = hmac.new(self._key, message, hashlib.sha256).digest()
        header = struct.pack(MESSAGE_HEADER_FORMAT, len(message), mac)
        full_message = header + message

        if len(full_message) > self._region_size - 33:
            raise ValueError(f"Message too large for region size. Max payload: {self._region_size - MESSAGE_HEADER_SIZE - 33} bytes.")

        self._shm[33:33 + len(full_message)] = full_message
        self._shm[32] = b'\x02'

        self._wait_for_peer_signal(32, b'\x03', timeout=timeout)
        self._shm[32] = b'\x00'
        logger.info(f"Sent message: {message[:50]}...")

    def receive_message(self, timeout: float = 5.0) -> bytes:
        if self._key is None:
            raise RuntimeError("Key not established. Call establish_dh_key first.")

        self._wait_for_peer_signal(32, b'\x02', timeout=timeout)

        header_data = bytes(self._peer_shm[33:33 + MESSAGE_HEADER_SIZE])
        msg_len, received_mac = struct.unpack(MESSAGE_HEADER_FORMAT, header_data)

        payload = bytes(self._peer_shm[33 + MESSAGE_HEADER_SIZE : 33 + MESSAGE_HEADER_SIZE + msg_len])

        expected_mac = hmac.new(self._key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_mac, received_mac):
            raise ValueError("HMAC verification failed. Message integrity compromised.")

        self._peer_shm[32] = b'\x03'
        logger.info(f"Received message: {payload[:50]}...")
        return payload

    def close(self):
        pass
# --- End SharedMemoryChannel definition ---


# Worker script source code (as a string)
# This is crucial for passing the worker logic to sub-interpreters.
# In a real application, you might load this from a file.
worker_script_source = """
import sys
import os
import time
import secrets
import hashlib
import hmac
import ctypes
import struct
import logging

logging.basicConfig(level=logging.INFO,
                    format='[SubInterp-%(process)d-%(thread)d] %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

MESSAGE_HEADER_FORMAT = "!I32s"
MESSAGE_HEADER_SIZE = struct.calcsize(MESSAGE_HEADER_FORMAT)

class SharedMemoryChannel:
    def __init__(self, buffer_address: int, region_offset: int, region_size: int,
                 peer_offset: int, peer_size: int):
        self._region_offset = region_offset
        self._region_size = region_size
        self._peer_offset = peer_offset
        self._peer_size = peer_size
        self._buffer_address = buffer_address

        self._shm = (ctypes.c_char * self._region_size).from_address(self._buffer_address + self._region_offset)
        self._peer_shm = (ctypes.c_char * self._peer_size).from_address(self._buffer_address + self._peer_offset)

        self._key = None

    def _wait_for_peer_signal(self, offset: int, expected_value: bytes, timeout: float = 5.0):
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            if self._peer_shm[offset:offset + len(expected_value)] == expected_value:
                return True
            time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for peer signal at offset {{offset}}")

    def establish_dh_key(self, P: int, G: int, timeout: float = 5.0):
        logger.info("Establishing DH key...")
        priv = secrets.randbelow(P - 2) + 1
        pub = pow(G, priv, P)
        pub_bytes = pub.to_bytes(32, 'big')

        self._shm[:32] = pub_bytes
        self._shm[32] = b'\\x01'

        self._wait_for_peer_signal(32, b'\\x01', timeout=timeout)
        their_pub = int.from_bytes(bytes(self._peer_shm[:32]), 'big')

        shared = pow(their_pub, priv, P)
        self._key = hashlib.sha256(shared.to_bytes((shared.bit_length() + 7) // 8 or 1, 'big')).digest()
        logger.info("DH key established.")
        return self._key

    def send_message(self, message: bytes, timeout: float = 5.0):
        if self._key is None:
            raise RuntimeError("Key not established. Call establish_dh_key first.")

        mac = hmac.new(self._key, message, hashlib.sha256).digest()
        header = struct.pack(MESSAGE_HEADER_FORMAT, len(message), mac)
        full_message = header + message

        if len(full_message) > self._region_size - 33:
            raise ValueError(f"Message too large for region size. Max payload: {{self._region_size - MESSAGE_HEADER_SIZE - 33}} bytes.")

        self._shm[33:33 + len(full_message)] = full_message
        self._shm[32] = b'\\x02'

        self._wait_for_peer_signal(32, b'\\x03', timeout=timeout)
        self._shm[32] = b'\\x00'
        logger.info(f"Sent message: {{message[:50]}}...")

    def receive_message(self, timeout: float = 5.0) -> bytes:
        if self._key is None:
            raise RuntimeError("Key not established. Call establish_dh_key first.")

        self._wait_for_peer_signal(32, b'\\x02', timeout=timeout)

        header_data = bytes(self._peer_shm[33:33 + MESSAGE_HEADER_SIZE])
        msg_len, received_mac = struct.unpack(MESSAGE_HEADER_FORMAT, header_data)

        payload = bytes(self._peer_shm[33 + MESSAGE_HEADER_SIZE : 33 + MESSAGE_HEADER_SIZE + msg_len])

        expected_mac = hmac.new(self._key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_mac, received_mac):
            raise ValueError("HMAC verification failed. Message integrity compromised.")

        self._peer_shm[32] = b'\\x03'
        logger.info(f"Received message: {{payload[:50]}}...")
        return payload

    def close(self):
        pass

def worker_main(region_offset: int, region_size: int, peer_offset: int, peer_size: int, buf_addr: int,
                P: int, G: int, worker_id: str):
    logger.info(f"Worker '{{worker_id}}' started.")
    channel = None
    try:
        channel = SharedMemoryChannel(buf_addr, region_offset, region_size, peer_offset, peer_size)
        channel.establish_dh_key(P, G)

        if worker_id == "A":
            message_to_send = b"Hello from subinterp A! This is a longer message to test buffer handling."
            channel.send_message(message_to_send)
            received_message = channel.receive_message()
            logger.info(f"Worker A received: {{received_message.decode()}}")
        else: # worker_id == "B"
            received_message = channel.receive_message()
            logger.info(f"Worker B received: {{received_message.decode()}}")
            message_to_send = b"Hi from subinterp B! Acknowledging your message."
            channel.send_message(message_to_send)

    except TimeoutError as e:
        logger.error(f"Worker '{{worker_id}}' communication timeout: {{e}}")
    except ValueError as e:
        logger.error(f"Worker '{{worker_id}}' data integrity error: {{e}}")
    except Exception as e:
        logger.exception(f"Worker '{{worker_id}}' encountered an unexpected error.")
    finally:
        if channel:
            channel.close()
        logger.info(f"Worker '{{worker_id}}' finished.")
"""

# Constants for shared memory
BUF_SIZE = 8192
REGION_A = (0, BUF_SIZE // 2)
REGION_B = (BUF_SIZE // 2, BUF_SIZE)

# Diffie-Hellman parameters (should be strong primes in production)
P_DH = 0xE95E4A5F737059DC60DF5991D45029409E60FC09
G_DH = 2

def run_sub_interpreter_worker(interp, worker_id, region_params, peer_params, buf_addr, P, G):
    """
    Helper function to set up and run a worker in a sub-interpreter.
    """
    logger.info(f"Setting up interpreter for worker '{worker_id}' (ID: {interp.id})...")
    try:
        # Set attributes in the sub-interpreter's __main__ module
        # These will be available to the worker_script_source when executed.
        interp.set_main_attrs(
            region_offset=region_params[0],
            region_size=region_params[1] - region_params[0],
            peer_offset=peer_params[0],
            peer_size=peer_params[1] - peer_params[0],
            buf_addr=buf_addr,
            P=P,
            G=G,
            worker_id=worker_id
        )
        logger.info(f"Attributes set for worker '{worker_id}'. Executing script...")

        # Execute the worker script. This will define worker_main and then call it.
        # The script itself will call worker_main with the attributes set above.
        interp.exec(worker_script_source + "\nworker_main(region_offset, region_size, peer_offset, peer_size, buf_addr, P, G, worker_id)")
        logger.info(f"Worker '{worker_id}' execution completed.")
    except RunFailedError as e:
        logger.error(f"Worker '{worker_id}' failed to run: {e}")
        if e.__cause__:
            logger.error(f"Original exception in sub-interpreter: {e.__cause__}")
    except Exception as e:
        logger.exception(f"Error setting up or running worker '{worker_id}'.")
    finally:
        logger.info(f"Closing interpreter for worker '{worker_id}' (ID: {interp.id}).")
        interp.close()

def main():
    logger.info("Starting main application.")

    buf = None
    interp_a = None
    interp_b = None
    executor = None

    try:
        # 1. Initialize shared memory
        buf = mmap.mmap(-1, BUF_SIZE)
        buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        logger.info(f"Shared memory buffer created at address: {hex(buf_addr)}")

        # 2. Create interpreters
        interp_a = create()
        interp_b = create()
        logger.info(f"Interpreters created: A (ID: {interp_a.id}), B (ID: {interp_b.id})")

        # 3. Use ThreadPoolExecutor to run sub-interpreters in separate OS threads
        # This prevents the main interpreter from blocking during interp.exec()
        executor = ThreadPoolExecutor(max_workers=2)

        future_a = executor.submit(run_sub_interpreter_worker, interp_a, "A", REGION_A, REGION_B, buf_addr, P_DH, G_DH)
        future_b = executor.submit(run_sub_interpreter_worker, interp_b, "B", REGION_B, REGION_A, buf_addr, P_DH, G_DH)

        # Wait for both workers to complete
        future_a.result() # This will re-raise any exceptions from the worker thread
        future_b.result()

        logger.info("All sub-interpreter workers completed.")

        # Optional: Verify communication from the main interpreter's perspective
        # This requires the main interpreter to also understand the channel protocol
        # For demonstration, let's just check the raw buffer state after completion
        # In a real system, you might have a dedicated "monitor" channel.
        logger.info("Verifying buffer state (simplified)...")
        # You could instantiate SharedMemoryChannel objects in the main process
        # to read the final state if needed, but for now, we trust the logs.

    except Exception as e:
        logger.exception("An error occurred in the main application.")
    finally:
        logger.info("Cleaning up resources.")
        if executor:
            executor.shutdown(wait=True)
        if buf:
            buf.close()
            logger.info("Shared memory buffer closed.")
        # Interpreters are closed by run_sub_interpreter_worker in this design.
        # If an error occurred before they were submitted, they might need explicit closing here.
        # For robustness, you might keep a list of active interpreters and close them all.
        logger.info("Main application finished.")

if __name__ == "__main__":
    main()

# /* EndApp
# Morphism = [
# None
# ]
