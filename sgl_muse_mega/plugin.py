"""The SGLang general-plugin entry point (``sglang.srt.plugins``: ``nvidia_megakernel``).

``register`` runs in every SGLang process and only registers each model's spec by path with SGLang's megakernel
framework: SGLang imports a spec only when ``--enable-nvidia-megakernel`` serves its architecture, and the target spec's
build installs the fused DFlash2 iteration's hooks where SGLang has no DFlash iteration slot
(``muse_spark.runtime.iteration.install``).
"""

SPECS = {
    "MuseSparkForConditionalGeneration": "sgl_muse_mega.muse_spark.spec:SPEC",
    "DFlash2DraftModel": "sgl_muse_mega.muse_spark.draft:SPEC",
}
# the SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        '"nvidia_megakernel"',
        "register",
        "register(architecture)",
        "register(path)",
    ),
    "srt/models/muse_spark_omni.py": ("MuseSparkForConditionalGeneration",),
    "srt/models/dflash.py": ("DFlash2DraftModel",),
}


def register() -> None:
    """Called by SGLang's plugin loader."""
    from sglang.kernels.plugin_kernels.nvidia.spec import register as register_spec

    for architecture, path in SPECS.items():
        register_spec(architecture=architecture, path=path)
