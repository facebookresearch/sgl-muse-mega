"""The SM103 (GB300) kernels and the build-time runner that binds and launches them; no SGLang imports.

Importing the package registers the C ABI's bind structs (``host/abi.h``) with ``common.host``.
"""

from pathlib import Path

from ...common import host

ROOT = Path(__file__).resolve().parent
KERNELS, HOST = ROOT / "kernels", ROOT / "host"

host.register(HOST / "abi.h")
