"""ctypes glue over the host libraries' C ABI. The bind structs are mirrored by parsing the ABI headers (a model's
package registers its headers on import), so a header is the only place a field is declared; filling one requires
every field by name. Every library exports the core entry points (``mk_module_load``, ``mk_launch``, ...).
"""

from __future__ import annotations

import ctypes
import os
import re
import weakref
from pathlib import Path

_CTYPES = {
    "uint64_t": ctypes.c_uint64,
    "int64_t": ctypes.c_int64,
    "double": ctypes.c_double,
}


def _parse_structs(header: Path) -> dict[str, type]:
    text = header.read_text()
    sizes = {k: int(v) for k, v in re.findall(r"^#define (\w+) (\d+)", text, re.M)}
    structs = {}
    for name, body in re.findall(r"typedef struct (\w+) \{(.*?)\} \1;", text, re.S):
        fields = []
        for line in body.splitlines():
            code = line.split("//")[0].strip()
            if not code:
                continue
            m = re.fullmatch(r"(uint64_t|int64_t|double)\s+(\w+)(?:\[(\w+)\])?;", code)
            if m is None:
                raise ValueError(f"{header}: unparsed field in {name}: {code!r}")
            t = _CTYPES[m[1]]
            n = m[3] and (int(m[3]) if m[3].isdigit() else sizes[m[3]])
            fields.append((m[2], t * n if n else t))
        structs[name] = type(name, (ctypes.Structure,), {"_fields_": fields})
    return structs


STRUCTS: dict[str, type] = {}


def register(header: Path) -> None:
    """Adds a C ABI header's bind structs to STRUCTS (names must not repeat)."""
    new = _parse_structs(header)
    clash = set(new) & set(STRUCTS)
    assert not clash, f"{header}: structs already registered: {sorted(clash)}"
    STRUCTS.update(new)


def make(struct: str, **values) -> ctypes.Structure:
    """A bind struct with every field given (tensors as their data pointers, None as 0). A ``*_peer`` array holds the
    largest world (MKH_RANKS): a smaller world gives its ranks and the rest stay 0; every other array is given whole.
    """
    cls = STRUCTS[struct]
    names = [f for f, _ in cls._fields_]
    missing, extra = set(names) - set(values), set(values) - set(names)
    if missing or extra:
        raise ValueError(
            f"{struct}: missing {sorted(missing)}, unknown {sorted(extra)}"
        )
    args = []
    for name, ftype in cls._fields_:
        v = values[name]
        if issubclass(ftype, ctypes.Array):
            n = len(v)
            assert n == ftype._length_ or (
                name.endswith("_peer") and n <= ftype._length_
            ), (struct, name, n)
            args.append(ftype(*(_scalar(x) for x in v)))
        else:
            args.append(_scalar(v))
    return cls(*args)


def _scalar(v):
    if v is None:
        return 0
    if hasattr(v, "data_ptr"):
        return v.data_ptr()
    return v


class Library:
    """One host library, loaded RTLD_LOCAL: every library exports the same core symbols."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.dll = ctypes.CDLL(str(path), mode=os.RTLD_LOCAL)
        d = self.dll
        d.mk_last_error.restype = ctypes.c_char_p
        d.mk_module_load.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        d.mk_module_free.argtypes = [ctypes.c_void_p]
        d.mk_module_free.restype = None
        d.mk_launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
        d.mk_launch_free.argtypes = [ctypes.c_void_p]
        d.mk_launch_free.restype = None
        d.mk_launch_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(STRUCTS["MkLaunchInfo"]),
        ]
        d.mk_launch_params.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int64,
        ]

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(
                f"{self.path.name}: {what}: {self.dll.mk_last_error().decode()}"
            )

    def load(self, cubin: Path, code_prefetch_prefix: str | None = None) -> "Module":
        image = Path(cubin).read_bytes()
        out = ctypes.c_void_p()
        prefix = code_prefetch_prefix.encode() if code_prefetch_prefix else None
        self.check(
            self.dll.mk_module_load(image, len(image), prefix, ctypes.byref(out)),
            f"load {cubin}",
        )
        return Module(self, out)

    def bind(self, fn, module: "Module | None", bind: ctypes.Structure) -> "Launch":
        """``fn``: the library's mk_*_bind; a None module gives a pack-only record (no device call)."""
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(type(bind)),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        out = ctypes.c_void_p()
        self.check(
            fn(
                module.handle if module else None, ctypes.byref(bind), ctypes.byref(out)
            ),
            fn.__name__,
        )
        return Launch(self, out, module)


def _free_with(obj, fn, handle) -> None:
    """Frees ``handle`` when ``obj`` is collected; at interpreter exit the process teardown reclaims it instead."""
    weakref.finalize(obj, fn, handle).atexit = False


class Module:
    def __init__(self, lib: Library, handle: ctypes.c_void_p):
        self.lib, self.handle = lib, handle
        _free_with(self, lib.dll.mk_module_free, handle)


class Launch:
    def __init__(self, lib: Library, handle: ctypes.c_void_p, module: Module | None):
        # the module outlives its launches
        self.lib, self.handle, self.module = lib, handle, module
        _free_with(self, lib.dll.mk_launch_free, handle)

    def __call__(self, stream: int, pdl: bool) -> None:
        self.lib.check(self.lib.dll.mk_launch(self.handle, stream, int(pdl)), "launch")

    def info(self) -> ctypes.Structure:
        i = STRUCTS["MkLaunchInfo"]()
        self.lib.check(
            self.lib.dll.mk_launch_info(self.handle, ctypes.byref(i)), "info"
        )
        return i

    def params(self) -> tuple[bytes, list[tuple[int, int]]]:
        """The parameter block and each parameter's (offset, size)."""
        info = self.info()
        blob = ctypes.create_string_buffer(info.param_bytes)
        n = info.params
        offs, sizes = (ctypes.c_uint64 * n)(), (ctypes.c_uint64 * n)()
        self.lib.check(
            self.lib.dll.mk_launch_params(self.handle, blob, len(blob), offs, sizes, n),
            "params",
        )
        return blob.raw, list(zip(offs, sizes))
