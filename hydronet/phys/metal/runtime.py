# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Multi-step Metal program infrastructure built on HydroForge's MetalProgram."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from hydroforge.kernels.codegen.types import scalar
from hydroforge.kernels.metal import (
    METAL_KERNEL_BODY_MARKER,
    MetalArgument,
    MetalProgram,
)
from hydroforge.kernels.registry import (
    KernelCall,
    KernelDeclaration,
    KernelImplementation,
    Launch,
)
from hydroforge.kernels.spec import KernelSpec
from hydroforge.platform.backend import METAL, Backend

_NAMED_BODY_PATTERN = re.compile(
    rf"^[ \t]*{re.escape(METAL_KERNEL_BODY_MARKER)}:\s*(?P<name>[A-Za-z_]\w*)\s*$",
    re.MULTILINE,
)
_FORBIDDEN_TOKENS = (
    "#include",
    "function_constant",
    "kernel void",
    "thread_position_in_grid",
    "using namespace",
    "[[",
)
_ARGS_STRUCT_PATTERN = re.compile(r"\bstruct\s+[A-Za-z_]\w*_args\b")


def buf(
    name: str, access: str, native: str, *, optional: bool = False
) -> MetalArgument:
    """Declare one Metal buffer argument-buffer field."""
    return MetalArgument(name, access, native, optional)


def scl(name: str, kind: str) -> MetalArgument:
    """Declare one Metal runtime scalar argument-buffer field."""
    return MetalArgument(name, None, kind, False)


def _without_comments(source: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_blocks)


def split_metal_library(source: str) -> tuple[str, dict[str, str]]:
    """Split a multi-body MSL template into helpers and named kernel bodies."""
    named = tuple(_NAMED_BODY_PATTERN.finditer(source))
    if not named:
        raise ValueError("Metal library source requires named body markers")
    names = tuple(match.group("name") for match in named)
    if len(names) != len(set(names)):
        raise ValueError(f"Metal library source has duplicate bodies: {names}")
    helpers = source[: named[0].start()]
    bodies: dict[str, str] = {}
    for index, match in enumerate(named):
        start = match.end()
        end = named[index + 1].start() if index + 1 < len(named) else len(source)
        bodies[match.group("name")] = source[start:end]
    return helpers, bodies


def read_metal_helpers(path: Path) -> str:
    """Read only the shared helper section (before the first body marker)."""
    helpers, _ = split_metal_library(path.read_text())
    return helpers


def _check_fragment(name: str, fragment: str, scope: str) -> None:
    forbidden = tuple(tok for tok in _FORBIDDEN_TOKENS if tok in fragment)
    if forbidden or _ARGS_STRUCT_PATTERN.search(fragment):
        raise ValueError(
            f"{name}: Metal kernel {scope}; forbidden wrapper syntax={forbidden}"
        )


@dataclass(frozen=True, slots=True)
class SubKernelSpec:
    """Typed ABI of one entry point inside a shared Metal library."""

    name: str
    arguments: tuple[MetalArgument, ...]
    extent: tuple[str, ...]


class MetalLibrary:
    """One compiled MSL library hosting multiple typed MetalProgram entries.

    All sub-kernels share the exact same effective MSL compilation unit so
    Apple Metal compiles one ``MTLLibrary`` per physics module.
    """

    def __init__(
        self,
        source: Path | str,
        subkernels: tuple[SubKernelSpec, ...],
        *,
        prelude: str = "",
    ) -> None:
        text = source.read_text() if isinstance(source, Path) else source
        helpers, all_bodies = split_metal_library(prelude + text)
        _check_fragment("helpers", helpers, "helpers must not contain wrapper syntax")

        selected_bodies: dict[str, str] = {}
        for sub in subkernels:
            if sub.name not in all_bodies:
                raise ValueError(
                    f"Metal library has no body for {sub.name!r}; "
                    f"available={tuple(all_bodies)}"
                )
            body = all_bodies[sub.name]
            if not body.strip():
                raise ValueError(f"{sub.name}: Metal kernel body must be non-empty")
            _check_fragment(sub.name, body, "body must contain physics statements only")
            physics = _without_comments(body)
            fields = set(re.findall(r"\bargs\.([A-Za-z_]\w*)", physics))
            declared = {arg.name for arg in sub.arguments}
            unknown = fields.difference(declared)
            if unknown:
                raise ValueError(
                    f"{sub.name}: Metal body references fields outside ABI: "
                    f"{sorted(unknown)}"
                )
            unused = declared.difference(fields, sub.extent)
            if unused:
                raise ValueError(
                    f"{sub.name}: Metal body does not consume declared ABI fields: "
                    f"{sorted(unused)}"
                )
            selected_bodies[sub.name] = body

        self.helpers = helpers
        self.subkernels = subkernels
        self.bodies = selected_bodies
        self.body = "\n".join(selected_bodies[sub.name] for sub in subkernels)
        self._raw_source = self._render_library_source()
        self._programs_cache: dict[bool, dict[str, MetalProgram]] = {}

    def _render_library_source(self) -> str:
        entries: list[str] = []
        for sub in self.subkernels:
            field_lines: list[str] = []
            for idx, arg in enumerate(sub.arguments):
                if arg.access is not None:
                    qualifier = "device const" if arg.access == "read" else "device"
                    native = arg.native
                else:
                    qualifier = "constant"
                    native = scalar(arg.native, "msl")
                field_lines.append(
                    f"    {qualifier} {native}* {arg.name} [[id({idx})]];"
                )
            fields_src = "\n".join(field_lines)
            entries.append(
                f"struct {sub.name}_args {{\n"
                f"{fields_src}\n"
                f"}};\n"
                f"kernel void {sub.name}(\n"
                f"    constant {sub.name}_args& args [[buffer(0)]],\n"
                f"    uint i [[thread_position_in_grid]],\n"
                f"    uint lid [[thread_position_in_threadgroup]],\n"
                f"    uint tpg [[threads_per_threadgroup]],\n"
                f"    uint simd_lane [[thread_index_in_simdgroup]],\n"
                f"    uint simd_group [[simdgroup_index_in_threadgroup]],\n"
                f"    uint simd_width [[threads_per_simdgroup]]) {{\n"
                f"{self.bodies[sub.name]}\n"
                f"}}\n"
            )
        entries_src = "\n".join(entries)
        return (
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            f"{self.helpers}\n"
            f"{entries_src}"
        )

    def programs(self) -> dict[str, MetalProgram]:
        """Return typed MetalProgram entries for the active math mode."""
        fast_math = METAL.math.physics_fast_math
        cached = self._programs_cache.get(fast_math)
        if cached is None:
            cached = self._programs_cache[fast_math] = {
                sub.name: MetalProgram(
                    self._raw_source,
                    sub.name,
                    sub.arguments,
                    extent=sub.extent,
                    origin="physics",
                    encoding="native",
                )
                for sub in self.subkernels
            }
        return cached


class _MetalMultiStepImplementation(KernelImplementation):
    def __init__(
        self,
        spec: KernelSpec,
        backend: Backend,
        declaration: MetalMultiStepKernel,
    ) -> None:
        super().__init__(spec, backend)
        if spec.uses_precision:
            raise TypeError(
                f"{spec.name}: a Metal precision-dependent KernelSpec must be "
                "resolved to float32 before it is built"
            )
        self._library = declaration.library_factory()
        self._prepare = declaration.prepare
        self._primary_kernel = declaration.primary_kernel
        self._capturable = declaration.capturable
        if self._primary_kernel not in self._library.bodies:
            raise ValueError(
                f"{spec.name}: primary Metal kernel {self._primary_kernel!r} "
                f"not in library"
            )
        self.helpers = self._library.helpers
        self.body = self._library.body

    @property
    def capturable(self) -> bool:
        return self._capturable

    def subprograms(self) -> dict[str, MetalProgram]:
        """Return all sub-kernel MetalProgram entries of this library."""
        return self._library.programs()

    def program(
        self,
        buffer_dtypes: Mapping[str, torch.dtype | None],
        emulation: str | None = None,
    ) -> MetalProgram:
        """Validate buffer dtypes and return the primary MetalProgram of the library."""
        if emulation is not None:
            raise TypeError(
                f"{self.spec.name}: HydroNet Metal kernels require native float32 "
                f"(emulation={emulation!r} is unsupported)"
            )
        for name, dtype in buffer_dtypes.items():
            if dtype is None and name in self.spec.optional:
                continue
            if dtype not in (torch.float32, torch.int32, torch.int64, torch.bool):
                raise TypeError(
                    f"{self.spec.name}.{name}: unsupported Metal buffer dtype {dtype}"
                )
        return self._library.programs()[self._primary_kernel]

    def _validate(self, call: KernelCall) -> None:
        self.program(call.buffer_dtypes, emulation=None)
        block = call.arguments["BLOCK_SIZE"]
        if type(block) is not int or not 1 <= block <= 1024:
            raise ValueError("Metal group size must be an exact int in [1, 1024]")

    def _compile(self, call: KernelCall) -> Launch:
        return self._prepare(
            dict(call.arguments), call.buffer_dtypes, self.subprograms()
        )


class MetalMultiStepKernel(KernelDeclaration):
    """A host-orchestrated sequence or loop of native MetalProgram launches."""

    toolchain: ClassVar[str] = "metal"

    library_factory: Callable[[], MetalLibrary]
    prepare: Callable[
        [dict[str, Any], Mapping[str, torch.dtype | None], dict[str, MetalProgram]],
        Callable[[], Any],
    ]
    primary_kernel: str
    capturable: bool = True

    def _build(self, spec: KernelSpec, backend: Backend) -> KernelImplementation:
        return _MetalMultiStepImplementation(spec, backend, self)


def close_launches(launches: list[Any]) -> None:
    """Release argument bindings owned by specialized _MetalLaunch callables."""
    for item in launches:
        close = getattr(item, "close", None)
        if callable(close):
            close()
