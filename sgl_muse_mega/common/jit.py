"""nvcc builds of device cubins and the host libraries that bind them, cached by content hash.

An artifact is one nvcc call over a source tree (a model's ``sm103/``: its ``kernels/`` on the include path). Outputs
are cached under ``$SGLANG_CACHE_DIR/sgl_muse_mega`` (default ``~/.cache/sglang/sgl_muse_mega``) by a key over the
compilers (nvcc; the host C++ compiler for the host libraries), the flags, every source file of the tree and the
artifact's paths relative to it (not their absolute paths, so a cache built from one checkout serves another); processes
sharing a cache build each artifact once (a file lock per artifact).
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

CACHE = (
    Path(os.environ.get("SGLANG_CACHE_DIR", Path.home() / ".cache" / "sglang"))
    / "sgl_muse_mega"
)
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))


def toolkit(name: str) -> str:
    """A CUDA toolkit binary (nvcc, cuobjdump): $CUDA_HOME/bin/<name>, default /usr/local/cuda. The builds and the
    tests take their tools from here, so they see one compiler."""
    return str(CUDA_HOME / "bin" / name)


NVCC = toolkit("nvcc")

# --register-usage-level=10 changes the SASS, the host -O level does not
DEVICE_FLAGS = (
    "-gencode=arch=compute_103a,code=sm_103a",
    "-std=c++20",
    "-O1",
    "--ptxas-options=--register-usage-level=10",
    "--diag-suppress=39,161,174,177,186,940",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
)
HOST_FLAGS = (
    "-Xcompiler",
    "-fPIC,-fvisibility=hidden",
    "-Xlinker",
    "-Bsymbolic",
    "-shared",
    "-lcuda",
)


@dataclass(frozen=True)
class Artifact:
    name: str
    kind: str  # "cubin" | "so"
    sources: tuple[Path, ...]
    defines: tuple[str, ...] = ()
    compress_none: bool = False  # the code-prefetch patch edits an uncompressed image


def _tree_digest(root: Path) -> str:
    """Every file the builds of ``root`` can read."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        for f in sorted(filenames):
            if f.endswith((".h", ".cuh", ".cu")):
                p = Path(dirpath, f)
                h.update(str(p.relative_to(root)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def _version(tool: str) -> str:
    return subprocess.run(
        [tool, "--version"], check=True, capture_output=True, text=True
    ).stdout


def toolchain_reasons(minimum: tuple[int, int] = (13, 1)) -> list[str]:
    """Why the builds cannot run here: no nvcc of at least ``minimum`` under $CUDA_HOME (sm_103a), or no host C++
    compiler for the host libraries."""
    if not Path(NVCC).exists():
        return [
            f"no nvcc at {NVCC} (set CUDA_HOME to a CUDA {minimum[0]}.{minimum[1]}+ toolkit)"
        ]
    m = re.search(r"release (\d+)\.(\d+)", _version(NVCC))
    if m is None or (int(m[1]), int(m[2])) < minimum:
        return [f"{NVCC} is not CUDA {minimum[0]}.{minimum[1]} or later"]
    if shutil.which("c++") is None:
        return ["no host C++ compiler (c++) on PATH for the host libraries"]
    return []


def _build_one(
    a: Artifact, root: Path, include: Path, tree: str, compilers: str
) -> Path:
    flags = [*DEVICE_FLAGS, *(f"-D{d}" for d in a.defines)]
    if a.compress_none:
        flags.append("--compress-mode=none")
    flags += ["-cubin"] if a.kind == "cubin" else list(HOST_FLAGS)
    rel = [str(p.relative_to(root)) for p in a.sources]
    key = hashlib.sha256(
        "\0".join((compilers, tree, *flags, *rel)).encode()
    ).hexdigest()[:16]
    cmd = [NVCC, f"-I{include}", *flags]
    d = CACHE / f"{a.name}-{key}"
    out = d / (f"{a.name}.cubin" if a.kind == "cubin" else f"{a.name}.so")
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.exists():
            return out
        tmp = d / f".{out.name}.{os.getpid()}"
        cmd += ["-o", str(tmp), *map(str, a.sources)]
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        (d / "build.log").write_text(" ".join(cmd) + "\n" + r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(
                f"nvcc failed for {a.name} (log: {d / 'build.log'}):\n{r.stderr[-4000:]}"
            )
        os.replace(tmp, out)
        return out


def build_artifacts(todo: list[Artifact], root: Path, include: Path) -> dict[str, Path]:
    """Builds (or finds in the cache) the given artifacts of the source tree ``root`` (``include``: its device include
    directory) in parallel; returns name -> path."""
    tree, nvcc = _tree_digest(root), _version(NVCC)
    host = nvcc + _version("c++") if any(a.kind == "so" for a in todo) else nvcc

    def one(a: Artifact) -> Path:
        return _build_one(a, root, include, tree, host if a.kind == "so" else nvcc)

    with concurrent.futures.ThreadPoolExecutor(len(todo)) as pool:
        return dict(zip((a.name for a in todo), pool.map(one, todo)))
