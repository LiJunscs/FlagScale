# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""DSv4 source storage schema, scale-aware decoding and lossless byte storage.

The native checkpoint uses E4M3 + 128x128 scales, and optionally routed
experts packed low-nibble-first as E2M1 + one E8M0 scale per 32 values.
MCore's standard DCP checkpoint contains dense model tensors, not these HF byte
layouts or Transformer Engine quantizer state. All scale arithmetic uses FP32.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from tools.checkpoint.deepseek_v4.hf_io import NativeQuantizedWeight, RawEncoded, SafetensorReader


class PrecisionError(ValueError):
    pass


DENSE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
HEADER_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN"}
E8M0_DTYPES = {"F8_E8M0", "F8_E8M0FNU"}
SCALE_SUFFIXES = (".weight_scale_inv", ".scale")
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def canonical_hf_name(name: str) -> str:
    """Normalize the published HF and native inference names to one schema."""
    name = name.removeprefix("model.")
    replacements = {
        "embed_tokens": "embed", "lm_head": "head", "self_attn": "attn",
        "mlp": "ffn", "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm", "q_a_proj": "wq_a",
        "q_a_layernorm": "q_norm", "q_b_proj": "wq_b",
        "gate_proj": "w1", "up_proj": "w3", "down_proj": "w2",
        "e_score_correction_bias": "bias",
    }
    name = ".".join(replacements.get(part, part) for part in name.split("."))
    return name.replace(".indexer.weights_proj.", ".indexer.scorer.weights_proj.")


def validate_quantization_config(config: Mapping[str, Any]) -> None:
    quant = config.get("quantization_config")
    if quant is not None:
        if not isinstance(quant, Mapping) or quant.get("quant_method") != "fp8":
            raise PrecisionError("Only the native DSv4 fp8 quantization_config is supported")
        if quant.get("fmt", "e4m3") != "e4m3":
            raise PrecisionError("DSv4 FP8 weights must use E4M3")
        if list(quant.get("weight_block_size", (128, 128))) != [128, 128]:
            raise PrecisionError("DSv4 FP8 weights require weight_block_size=[128,128]")
        if quant.get("scale_fmt") not in (None, "ue8m0"):
            raise PrecisionError("Unsupported DSv4 scale_fmt")
        excluded = quant.get("modules_to_not_convert")
        if excluded is not None and (not isinstance(excluded, list) or any(
                not isinstance(name, str) or not name for name in excluded)):
            raise PrecisionError("modules_to_not_convert must be a list of nonempty module names")
    if config.get("expert_dtype") not in (None, "fp4", "fp8", "bf16", "bfloat16",
                                           "fp16", "float16", "fp32", "float32"):
        raise PrecisionError("Unsupported expert_dtype (MXFP4 is distinct from NVFP4/int8)")


def is_hf_mtp_key(name: str) -> bool:
    return re.match(r"^mtp\.\d+\.", canonical_hf_name(name)) is not None


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str = "dense"
    scale: str | None = None
    scale_dtype: str | None = None


class PrecisionSource:
    """Compile physical tensors and scales into logical, dense mapping inputs."""

    def __init__(self, reader: SafetensorReader, config: Mapping[str, Any],
                 *, dtype: torch.dtype = torch.bfloat16, device: str = "cpu", skip_mtp: bool = False):
        validate_quantization_config(config)
        self.reader = reader
        self.dtype = dtype
        self.device = device
        self.specs: dict[str, TensorSpec] = {}
        self.consumed: set[str] = set()
        dtypes, shapes = reader.tensor_dtypes(), reader.tensor_shapes()
        self.physical_keys = {name for name in reader.keys if not (skip_mtp and is_hf_mtp_key(name))}
        self.dtype_counts = dict(Counter(dtypes[name] for name in self.physical_keys))
        scales = {name for name in self.physical_keys if name.endswith(SCALE_SUFFIXES)}
        paired_scales = set()
        quant = config.get("quantization_config") or {}
        for name in sorted(self.physical_keys - scales):
            dtype_code, shape = dtypes[name], shapes[name]
            canonical = canonical_hf_name(name)
            if canonical in self.specs:
                raise PrecisionError(f"Duplicate HF semantic key: {canonical}")
            kind, scale_name = "dense", None
            if dtype_code in FP8_DTYPES:
                kind = "fp8"
                if not quant:
                    raise PrecisionError(f"FP8 weight requires quantization_config: {name}")
            elif dtype_code in {"I8", "U8"}:
                if config.get("expert_dtype") != "fp4" or not re.fullmatch(
                    r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", canonical
                ):
                    raise PrecisionError(f"Unsupported HF tensor {name} ({dtype_code}); not declared MXFP4")
                kind = "mxfp4"
                if len(shape) != 2:
                    raise PrecisionError(f"Packed MXFP4 weight must be a matrix: {name}")
                shape = (*shape[:-1], shape[-1] * 2)
            elif canonical.endswith("tid2eid"):
                if dtype_code not in {"I32", "I64"}:
                    raise PrecisionError(f"Hash router {name} must use I32/I64")
            elif dtype_code not in HEADER_DTYPES:
                raise PrecisionError(f"Unsupported HF tensor {name} ({dtype_code})")
            if kind != "dense":
                if len(shape) != 2 or not all(shape) or not name.endswith(".weight"):
                    raise PrecisionError(f"Quantized DSv4 linear weight must be a nonempty matrix: {name}")
                candidates = [name.removesuffix(".weight") + suffix for suffix in SCALE_SUFFIXES]
                matches = [candidate for candidate in candidates if candidate in scales]
                if len(matches) != 1:
                    raise PrecisionError(f"Expected exactly one scale tensor for {name}; found {matches}")
                scale_name = matches[0]
                scale_dtype = dtypes[scale_name]
                if scale_dtype not in {*HEADER_DTYPES, *E8M0_DTYPES, "U8"}:
                    raise PrecisionError(f"Unsupported scale dtype for {scale_name}: {scale_dtype}")
                if scale_dtype == "U8" and quant.get("scale_fmt") != "ue8m0":
                    raise PrecisionError(f"Byte scale requires scale_fmt=ue8m0: {scale_name}")
                rows, cols = shape
                expected = (math.ceil(rows / 128), math.ceil(cols / 128))
                if kind == "mxfp4":
                    if cols % 32:
                        raise PrecisionError(f"MXFP4 reduction dimension must be divisible by 32: {name}")
                    expected = (rows, cols // 32)
                if shapes[scale_name] != expected:
                    raise PrecisionError(f"Scale shape mismatch: {scale_name} {shapes[scale_name]} != {expected}")
                paired_scales.add(scale_name)
            self.specs[canonical] = TensorSpec(
                name, shape, dtype_code, kind, scale_name,
                dtypes[scale_name] if scale_name else None,
            )
        if scales != paired_scales:
            raise PrecisionError(f"Orphan scale tensors: {sorted(scales - paired_scales)[:20]}")

    @property
    def keys(self) -> set[str]:
        return set(self.specs)

    @property
    def low_precision_kinds(self) -> set[str]:
        return {spec.kind for spec in self.specs.values() if spec.kind != "dense"}

    def _scale(self, spec: TensorSpec) -> torch.Tensor:
        if spec.scale_dtype in E8M0_DTYPES or spec.scale_dtype == "U8":
            encoded = self.reader.get_bytes(spec.scale).to(self.device)
            if bool((encoded == 255).any()):
                raise PrecisionError(f"Invalid E8M0 NaN scale: {spec.scale}")
            result = torch.exp2(encoded.to(torch.float32) - 127)
        else:
            result = self.reader.get_tensor(spec.scale).to(self.device, dtype=torch.float32)
        if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
            raise PrecisionError(f"Scale must be finite and positive: {spec.scale}")
        return result

    def get_tensor(self, name: str) -> torch.Tensor:
        spec = self.specs[name]
        if spec.kind == "dense":
            result = self.reader.get_tensor(spec.name)
        else:
            packed = self.reader.get_bytes(spec.name)
            scale = self._scale(spec)
            result = torch.empty(spec.shape, dtype=self.dtype, device="cpu")
            # Bound temporary FP32 storage to 128 rows of one matrix.
            table = FP4_VALUES if spec.kind == "mxfp4" else _fp8_table()
            lut = torch.tensor(table, dtype=torch.float32, device=self.device)
            for row in range(0, spec.shape[0], 128):
                data = packed[row:row + 128].to(self.device)
                if spec.kind == "mxfp4":
                    decoded = torch.stack((lut[(data & 15).long()], lut[(data >> 4).long()]), -1).flatten(1)
                    values = decoded.reshape(data.shape[0], -1, 32) * scale[row:row + 128, :, None]
                    values = values.reshape(data.shape[0], -1)
                else:
                    values = lut[data.long()]
                    multiplier = scale[row // 128].repeat_interleave(128)[:spec.shape[1]]
                    values = values * multiplier
                converted = values.to(self.dtype)
                if not bool(torch.isfinite(converted).all()):
                    raise PrecisionError(f"Non-finite/overflowing dequantized weight: {spec.name}")
                result[row:row + 128].copy_(converted.cpu())
            self.consumed.add(spec.scale)
        self.consumed.add(spec.name)
        return result

    def validate_consumed(self) -> None:
        remaining = self.physical_keys - self.consumed
        if remaining:
            raise PrecisionError(f"Unconsumed physical HF tensors: {sorted(remaining)[:20]}")


def _fp8_table() -> list[float]:
    result = []
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        magnitude = (mantissa * 2.0**-9 if exponent == 0
                     else (1 + mantissa / 8) * 2.0**(exponent - 7))
        if exponent == 15 and mantissa == 7:
            magnitude = float("nan")
        result.append(-magnitude if byte & 128 else magnitude)
    return result


@dataclass(frozen=True)
class PrecisionPolicy:
    # dtype is a config default / explicit override, not every tensor's dtype.
    dtype: str
    device: str
    preserve_quantization: bool
    reason: str
    preserve_source_dtypes: bool = True
    archive_quantization: bool = False
    quantized_kinds: tuple[str, ...] = ()
    unquantized_modules: tuple[str, ...] = ()
    expert_dtype: str | None = None
    warnings: tuple[str, ...] = ()
    source_dtype_counts: tuple[tuple[str, int], ...] = ()
    source_quantized_kinds: tuple[str, ...] = ()
    native_fp8: bool = False
    native_export: bool = False

    def report(self) -> dict:
        return dict(dtype=self.dtype, device=self.device,
                    dtype_scope="config default / quantized-input decode dtype; auto preserves individual dense dtypes",
                    preserve_source_dtypes=self.preserve_source_dtypes,
                    preserve_quantization=self.preserve_quantization,
                    archive_quantization=self.archive_quantization,
                    quantized_kinds=list(self.quantized_kinds),
                    reason=self.reason, warnings=list(self.warnings),
                    source_dtype_counts=dict(self.source_dtype_counts),
                    source_quantized_kinds=list(self.source_quantized_kinds),
                    unquantized_module_count=len(self.unquantized_modules),
                    native_fp8=self.native_fp8,
                    native_export=self.native_export,
                    scale_handling=("native_te_fp8_qt_storage" if self.native_fp8
                                    else "native_quantized_byte_export" if self.native_export
                                    else "restore_original_bytes_after_model_verification" if self.preserve_quantization
                                    else "decode_for_mcore_and_archive_original_bytes" if self.archive_quantization
                                    else "no_quantization"))


def _output_quantization_layout(specs):
    """Describe the observed restored layout; never invent a quantized module."""
    modules, expert_groups = set(), {}
    for name, spec in specs.items():
        module = "lm_head" if name == "head.weight" else spec.name.removesuffix(".weight")
        expert = re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", name)
        if expert:
            module = spec.name.rsplit(".experts.", 1)[0] + ".experts"
            expert_groups.setdefault(module, set()).add(spec.kind)
        elif name.endswith(".weight") and spec.kind == "dense":
            modules.add(module)
    expert_kinds = set()
    for module, kinds in expert_groups.items():
        if len(kinds) > 1:
            raise PrecisionError(f"Mixed storage within one routed-expert module is unsupported: {module}")
        if kinds == {"dense"}:
            modules.add(module)
        else:
            expert_kinds.update(kinds)
    if len(expert_kinds) > 1:
        raise PrecisionError("A single expert_dtype cannot describe multiple routed-expert formats")
    expert_dtype = {"mxfp4": "fp4", "fp8": "fp8"}.get(next(iter(expert_kinds), "dense"))
    return tuple(sorted(modules)), expert_dtype


def resolve_precision(source, config: Mapping[str, Any], requested: str, device: str, *,
                      to_mcore: bool, archive=None, native_export: bool = False) -> PrecisionPolicy:
    """Use actual source storage, never GPU capabilities or a reference recipe.

    For export, source specs must come from DCP metadata. An archive is accepted
    only by the explicit restoration path; tensor values are checked when read.
    `native_export` keeps quantized payloads and scales as HF bytes instead of
    decoding them (mcore-to-hf only, --dtype auto, no archive).
    """
    if source is None:
        raise PrecisionError("Actual source dtype metadata is required")
    if requested not in {"auto", *DENSE_DTYPES} and requested != "fp8":
        raise PrecisionError(f"Unsupported dtype: {requested}")
    if archive is not None and (to_mcore or requested != "auto"):
        raise PrecisionError("--restore-original-quantization requires mcore-to-hf --dtype auto")
    if native_export and to_mcore:
        raise PrecisionError("--export-native-quantization is only valid for mcore-to-hf")
    if native_export and archive is not None:
        raise PrecisionError("--export-native-quantization and --restore-original-quantization are mutually exclusive")
    if native_export and requested != "auto":
        raise PrecisionError("--export-native-quantization requires --dtype auto")
    if requested == "fp8":
        if not to_mcore:
            raise PrecisionError("--dtype fp8 (native TE storage) is only valid for hf-to-mcore")
        from .fp8_native import require_native_fp8_capable
        require_native_fp8_capable()
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise PrecisionError("Conversion arithmetic supports --device cpu or cuda[:N]")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            device = "cpu"
        elif selected.index is not None and selected.index >= torch.cuda.device_count():
            raise PrecisionError(f"CUDA device does not exist: {device}")
    kinds = source.low_precision_kinds
    if kinds - {"fp8", "mxfp4"}:
        raise PrecisionError(f"Unsupported source quantization kinds: {sorted(kinds)}")
    if kinds and not to_mcore and not native_export:
        raise PrecisionError(
            "Native quantized DCP storage needs a supported MCore storage adapter; "
            "HF reference formats cannot describe it. Pass --export-native-quantization "
            "to keep FP8 payloads and scales as HF bytes")
    if native_export and kinds and kinds != {"fp8"}:
        raise PrecisionError(f"Native export requires fp8 storage only: {sorted(kinds)}")
    native_noop = False
    if native_export and not kinds:
        native_noop = True
        native_export = False
    # The largest observed floating storage population determines only the
    # config default. Auto never casts individual dense tensors to that default.
    populations = Counter()
    dtype_names = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}
    for spec in source.specs.values():
        code = "bf16" if spec.kind != "dense" else dtype_names.get(spec.dtype)
        if code:
            populations[code] += math.prod(spec.shape)
    dtype = requested if requested != "auto" else max(populations, key=populations.get, default="bf16")
    # Use a fixed BF16 model representation for HF quantized weights, independent
    # of other dense buffers and hardware. Raw weights/scales remain archived.
    if to_mcore and kinds and requested == "auto":
        dtype = "bf16"
    native_fp8 = requested == "fp8"
    if native_fp8:
        # TE Float8BlockwiseQTensor payloads declare BF16 fake metadata dtype and
        # dense tensors keep their source dtypes; "bf16" is the training dtype.
        dtype = "bf16"
    notes = []
    if native_noop:
        notes.append("--export-native-quantization ignored: source stores no quantized tensors")
    declared = bool(config.get("quantization_config")) or config.get("expert_dtype") in {"fp8", "fp4"}
    if declared and not kinds and archive is None:
        label = "HF source config" if to_mcore else "HF reference config"
        notes.append(f"{label} declares quantization, but actual source weights are dense; "
                     "preserving source tensor dtypes with --dtype auto and ignoring that recipe. "
                     "No quantization or scale tensors will be generated.")
    if to_mcore and kinds:
        notes.append("HF quantized weights are decoded for standard MCore model tensors; original "
                     "payloads and scales are archived in hf_quantized/. This archive is not TE "
                     "quantizer state and is not used automatically by training or HF export.")
    modules, expert_dtype, restored_kinds = (), None, ()
    if native_export:
        modules, expert_dtype = _output_quantization_layout(source.specs)
        restored_kinds = tuple(sorted(kinds))
    elif archive is not None:
        layout = {**source.specs, **archive.output_specs}
        modules, expert_dtype = _output_quantization_layout(layout)
        restored_kinds = tuple(sorted(archive.source.low_precision_kinds))
    reason = ("Restore source archive bytes after current model dtype/shape/value verification"
              if archive is not None else "Preserve actual source tensor dtypes; reference precision is not applied")
    if to_mcore and kinds:
        reason = "Preserve dense source dtypes; decode quantized model tensors and archive original payloads/scales"
    if native_fp8:
        reason = ("Native TE Float8BlockwiseQTensor storage for quantized tensors: FP8 payloads and "
                  "128x128 scales are preserved byte-exactly; MXFP4 experts are requantized to E4M3 "
                  "blockwise 128x128; dense tensors keep their source dtypes")
    if native_export:
        reason = ("Native byte-level export: FP8 E4M3 payloads and E8M0 scales preserved bit-exactly; "
                  "MXFP4-origin experts stay E4M3 with FP32 block scales; dense tensors keep source dtypes")
    if requested != "auto" and requested in DENSE_DTYPES:
        reason = "Explicit dense dtype override; FP32 stability tensors and integer tables remain unchanged"
    return PrecisionPolicy(
        dtype, device, archive is not None, reason,
        preserve_source_dtypes=requested == "auto", archive_quantization=to_mcore and bool(kinds),
        quantized_kinds=restored_kinds, unquantized_modules=modules, expert_dtype=expert_dtype,
        warnings=tuple(notes), source_dtype_counts=tuple(sorted(Counter(
            spec.dtype for spec in source.specs.values()).items())),
        source_quantized_kinds=tuple(sorted(kinds)),
        native_fp8=native_fp8,
        native_export=native_export,
    )


def output_config(config: Mapping[str, Any], policy: PrecisionPolicy, *, archive=None) -> dict:
    result = copy.deepcopy(dict(config))
    dtype = str(DENSE_DTYPES[policy.dtype]).removeprefix("torch.")
    result["dtype" if "dtype" in result else "torch_dtype"] = dtype
    if "dtype" in result and "torch_dtype" in result:
        result["torch_dtype"] = dtype
    # Reference-only precision declarations must never survive a dense export.
    for field in ("quantization_config", "scale_fmt", "scale_dtype"):
        result.pop(field, None)
    result["expert_dtype"] = policy.expert_dtype or policy.dtype
    if policy.native_export:
        quant = config.get("quantization_config")
        if not isinstance(quant, Mapping) or not quant:
            raise PrecisionError(
                "Native quantized export requires a quantization_config in the mapping-plan target config")
        result["quantization_config"] = copy.deepcopy(dict(quant))
        result["quantization_config"]["modules_to_not_convert"] = list(policy.unquantized_modules)
        result["expert_dtype"] = policy.expert_dtype or "fp8"
        return result
    if policy.preserve_quantization:
        if archive is None:
            raise PrecisionError("Original quantization restoration requires its source archive config")
        for field in ("quantization_config", "scale_fmt", "scale_dtype"):
            if field in archive.config:
                result[field] = copy.deepcopy(archive.config[field])
        quant = result.get("quantization_config")
        if not isinstance(quant, dict):
            raise PrecisionError("Original quantized archive lacks a supported quantization_config")
        # Rebuild exclusions from actual outputs (including runtime lm_head).
        # Old exclusions may refer to source names or dropped auxiliary layers.
        quant["modules_to_not_convert"] = list(policy.unquantized_modules)
    elif archive is not None:
        raise PrecisionError("Archive config cannot be applied without explicit restoration")
    return result


def _e8m0_encoded(scale_inv: torch.Tensor, name: str) -> torch.Tensor:
    """Encode FP32 block scales to E8M0 bytes, with an exact round-trip check.

    The 2-D grid shape of ``scale_inv`` is preserved: HF readers derive the
    block layout of the quantized weight from ``scales.shape[-2:]``, so a
    flattened scale would break dequantized loading of the exported archive.
    """
    values = scale_inv.detach().to("cpu").float().reshape(-1)
    exponents = torch.log2(values)
    rounded = torch.round(exponents)
    if not bool(torch.isfinite(exponents).all()) or not bool(torch.equal(exponents, rounded)):
        raise PrecisionError(f"Scale values are not exact powers of two: {name}")
    bias = rounded.to(torch.int64) + 127
    if bool((bias < 0).any()) or bool((bias > 254).any()):
        raise PrecisionError(f"E8M0 exponent out of range: {name}")
    encoded = bias.to(torch.uint8)
    if bool((encoded == 255).any()):
        raise PrecisionError(f"Invalid E8M0 NaN scale: {name}")
    if not bool(torch.equal(torch.exp2(encoded.float() - 127), values)):
        raise PrecisionError(f"E8M0 scale round-trip failed: {name}")
    return encoded.reshape(scale_inv.shape)


def encode_hf_tensors(tensors, source, policy: PrecisionPolicy, config: Mapping[str, Any], *, archive=None):
    if policy.preserve_quantization != (archive is not None):
        raise PrecisionError("Original quantization restoration requires an explicit source archive")
    seen = set()
    for name, tensor in tensors:
        if name in seen:
            raise PrecisionError(f"Duplicate logical HF output: {name}")
        seen.add(name)
        spec = source.specs.get(name) if source else None
        if source and spec is None:
            raise PrecisionError(f"HF output absent from mapped schema: {name}")
        if spec and name.endswith("wo_a.weight") and tensor.ndim == 3 and len(spec.shape) == 2:
            if spec.shape == (tensor.shape[0] * tensor.shape[1], tensor.shape[2]):
                tensor = tensor.flatten(0, 1)
        if not isinstance(tensor, NativeQuantizedWeight):
            if spec and tuple(tensor.shape) != spec.shape:
                raise PrecisionError(f"HF output shape mismatch: {name} {tuple(tensor.shape)} != {spec.shape}")
        output_name = spec.name if spec else name
        if policy.native_export and isinstance(tensor, NativeQuantizedWeight):
            if spec is None or spec.kind != "fp8" or not spec.scale:
                raise PrecisionError(f"Native export requires an fp8 spec with a scale: {name}")
            payload = tensor.data.detach().to("cpu").contiguous()
            if payload.dtype != torch.uint8:
                raise PrecisionError(f"Native FP8 payload must be uint8 bytes: {name}")
            rows, cols = tensor.shape
            expected = (math.ceil(rows / 128), math.ceil(cols / 128))
            scale_inv = tensor.scale_inv.detach().to("cpu").float().contiguous()
            if tuple(scale_inv.shape) != expected:
                raise PrecisionError(
                    f"Native scale shape mismatch: {name} {tuple(scale_inv.shape)} != {expected}")
            if not bool(torch.isfinite(scale_inv).all()) or bool((scale_inv <= 0).any()):
                raise PrecisionError(f"Scale must be finite and positive: {name}")
            yield output_name, RawEncoded(payload, "F8_E4M3")
            if spec.scale_dtype in E8M0_DTYPES:
                yield spec.scale, RawEncoded(_e8m0_encoded(scale_inv, name), "F8_E8M0")
            elif spec.scale_dtype == "F32":
                yield spec.scale, scale_inv
            else:
                raise PrecisionError(
                    f"Unsupported native export scale dtype: {name}: {spec.scale_dtype}")
            continue
        if archive is not None and name in archive.output_specs:
            weight, scale = archive.restore(name, tensor)
            yield output_name, weight
            yield archive.output_specs[name].scale, scale
        else:
            if tensor.is_floating_point():
                if tensor.dtype not in HEADER_DTYPES.values():
                    raise PrecisionError(f"Unsupported native MCore storage dtype: {name}: {tensor.dtype}")
                if not policy.preserve_source_dtypes and tensor.dtype != torch.float32:
                    tensor = tensor.to(DENSE_DTYPES[policy.dtype])
                if not bool(torch.isfinite(tensor).all()):
                    raise PrecisionError(f"Non-finite/overflowing HF output: {name}")
            elif tensor.dtype not in {torch.int32, torch.int64} or not name.endswith("tid2eid"):
                raise PrecisionError(f"Unsupported integer HF output: {name}: {tensor.dtype}")
            yield output_name, tensor
    if source and seen != source.keys:
        raise PrecisionError(f"Unwritten HF targets: {sorted(source.keys - seen)[:20]}")
    if archive is not None:
        archive.validate_consumed()
