# Test

from `./~`:
    `/usr/local/python/current/bin/python /workspaces/MSC/src/__init__.py benchmark uv run /usr/local/python/current/bin/python /workspaces/MSC/src/serverTest.py`


**Running the Tests**:
   - Save `serverTest.py` in `server/`.
   - Ensure `server.py` is present.
   - Run:
     ```bash
     python server/serverTest.py
     ```
   - Requirements:
     - `localhost:11434` running for `RAGKernel` (optional; falls back gracefully).
     - Write permissions for logs.
   - Output includes:
     - Server startup/shutdown logs.
     - Agency transformations, kernel resolutions, quantum encodings, frame manipulations, and quine outputs.

---

### CPythonification Guidance

1. **PyObject Manipulation**:
   - Access `PyObject` fields via `ctypes`:
     ```python
     frame = CPythonFrame.from_object("test")
     py_obj = py_object(frame)
     py_struct = PyObject.from_address(id(frame))
     ```
   - Manipulate `ob_refcnt` and `ob_type` for custom objects.

2. **Type System**:
   - Create `PyTypeObject` for `CPythonFrame`:
     ```python
     from ctypes import Structure, c_char_p, c_int
     class PyTypeObject(Structure):
         _fields_ = [("ob_base", PyObject), ("tp_name", c_char_p)]
     ```
   - Define methods via `PyMethodDef`.

3. **Reflexivity**:
   - Use `inspect` to extract and modify source code:
     ```python
     source = inspect.getsource(CPythonFrame)
     frame.value = source  # Store own source
     ```
   - Generate quines by recompiling source via `compile()`.

4. **Memory Management**:
   - Extend `PyWord` with a cache:
     ```python
     class PyWordCache:
         def __init__(self, capacity: int):
             self.cache = {}
             self.capacity = capacity
         def get(self, key: int) -> Optional[PyWord]:
             return self.cache.get(key)
         def put(self, key: int, word: PyWord):
             if len(self.cache) >= self.capacity:
                 self.cache.pop(next(iter(self.cache)))
             self.cache[key] = word
     ```
   - Use `PyMem_Malloc` for allocations.

5. **Quantum Integration**:
   - Map `Morphology` to CPython states:
     - `MARKOVIAN`: Irreversible deallocation.
     - `NON_MARKOVIAN`: Reversible object cloning.
   - Use `QuantumOperator` to transform `PyObject` data.

6. **Resources**:
   - CPython Source: `Include/object.h`, `Objects/typeobject.c`.
   - C API Docs: https://docs.python.org/3/c-api/
   - Homoiconicity: Study Lisp/Scheme for inspiration.
