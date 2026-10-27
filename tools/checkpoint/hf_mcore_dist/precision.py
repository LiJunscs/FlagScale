# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Storage descriptors shared by model presets and mapping execution."""
from dataclasses import dataclass
import torch

DENSE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
HEADER_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN"}
E8M0_DTYPES = {"F8_E8M0", "F8_E8M0FNU"}
SCALE_SUFFIXES = (".weight_scale_inv", ".scale")

class PrecisionError(ValueError):
    pass

@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str = "dense"
    scale: str | None = None
    scale_dtype: str | None = None

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
    preserve_storage: bool = False
    storage_mode: str = "dense"

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
                    preserve_storage=self.preserve_storage, storage_mode=self.storage_mode,
                    load_decode_dtype="fp32" if self.preserve_storage else None,
                    scale_handling=("original_payloads_and_scales_in_dcp" if self.preserve_storage
                                    else "native_te_fp8_qt_storage" if self.native_fp8
                                    else "native_quantized_byte_export" if self.native_export
                                    else "restore_original_bytes_after_model_verification" if self.preserve_quantization
                                    else "decode_for_mcore_and_archive_original_bytes" if self.archive_quantization
                                    else "decode_to_dense_dcp" if self.source_quantized_kinds and self.storage_mode == "dense"
                                    else "no_quantization"))
