# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Native FP8 output storage for the DSv4 converter.

Quantized HF sources are written as TransformerEngine ``Float8BlockwiseQTensor``
payloads instead of decoded BF16 tensors, so MCore training can load and run
native FP8 (blockwise 128x128) directly from the converted checkpoint.

Storage contract (verified against megatron-core 0.17 DCP + torch 2.9):
- one DCP write item per mapping piece; ``torch.save`` pickles the whole qt,
  so payload bytes (uint8 E4M3) and FP32 scales travel inside ``.distcp``;
- DCP metadata declares the fake high-precision dtype (BF16);
- MCore's load path (``force_all_tensors_to_non_fp8``) dequantizes each qt
  object into a dense tensor before copying into model parameters;
- routed experts keep the registry's 3-D layout: a (1, rows, cols) chunk of the
  global (experts, rows, cols) tensor. The qt logical shape must match the
  chunk exactly, which is achieved by viewing a 2-D qt to the piece shape (TE
  dispatches ``aten.view`` and keeps the 2-D physical storage).

FP8 sources keep their original payload bytes and scale values; shapes must be
divisible by 128 (no padding is applied). MXFP4 sources are decoded to FP32 and
requantized to E4M3 row-block-major 128x128 scales (the TE native format); the
original bytes stay archived in ``hf_quantized/``.
"""

from __future__ import annotations

import torch

from .precision import PrecisionError

FP8_BLOCK = 128
FP8_MAX = 448.0

TE_IMPORT_ERROR = (
    "Native FP8 output requires TransformerEngine (transformer_engine.pytorch "
    "with Float8BlockwiseQTensor); install the FL TE build to use --dtype fp8")


class NativeFp8Error(PrecisionError):
    pass


class NativeFp8Target:
    """Sentinel storage-dtype marker: a key's chunks materialize as TE qt
    payloads instead of dense tensors (see streaming._storage_dtype)."""

    def __repr__(self):
        return "NativeFp8Target"


def _te_qt_types():
    try:
        from transformer_engine.pytorch.cpp_extensions import DType as te_dtype
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
            Float8BlockQuantizer, Float8BlockwiseQTensor,
        )
    except ImportError as error:  # pragma: no cover - env dependent
        raise NativeFp8Error(f"{TE_IMPORT_ERROR}: {error}") from error
    return te_dtype, Float8BlockwiseQTensor, Float8BlockQuantizer


def require_native_fp8_capable() -> None:
    _te_qt_types()


def register_native_fp8_safe_globals() -> None:
    """Allowlist the qt pickle globals for torch>=2.6 ``weights_only=True``.

    torch's DCP reader hard-codes ``torch.load(..., weights_only=True)``
    (torch/distributed/checkpoint/filesystem.py), and stock megatron-core's
    SAFE_GLOBALS covers only mcore/numpy classes — a checkpoint written by this
    converter cannot be loaded back until the TE qt globals are allowlisted.
    The five required entries were verified live against torch 2.9 + TE-FL:
    ``getattr`` (quantizer enum lookup) plus the qt/quantizer/dtype objects and
    the bound ``_make_in_reduce_ex`` reduce callable (rejected as an
    "unrecognized function" unless registered explicitly). Call this in the
    training entrypoint BEFORE any distributed checkpoint load.
    """
    import builtins

    torch.serialization.add_safe_globals([builtins.getattr])
    te_dtype, qt_cls, quantizer_cls = _te_qt_types()
    torch.serialization.add_safe_globals(
        [qt_cls, quantizer_cls, te_dtype, qt_cls._make_in_reduce_ex])


def build_native_qt(data: torch.Tensor, scale_inv: torch.Tensor,
                    fake_dtype: torch.dtype = torch.bfloat16):
    """Build a 2-D TE Float8BlockwiseQTensor from E4M3 bytes + FP32 scale_inv.

    ``data`` is uint8 E4M3 storage of shape (rows, cols); ``scale_inv`` is the
    dequantization multiplier (block_amax / 448) of shape (rows/128, cols/128).
    Dequantize semantics match the HF DSv4 128x128 block-scale format exactly.
    """
    te_dtype, Float8BlockwiseQTensor, Float8BlockQuantizer = _te_qt_types()
    if data.dtype == torch.float8_e4m3fn:
        data = data.view(torch.uint8)
    if data.dtype != torch.uint8 or data.ndim != 2:
        raise NativeFp8Error(f"FP8 payload must be a 2-D uint8 tensor: {tuple(data.shape)}")
    rows, cols = data.shape
    if rows % FP8_BLOCK or cols % FP8_BLOCK:
        raise NativeFp8Error(
            f"Native FP8 storage needs shapes divisible by {FP8_BLOCK}: {rows}x{cols}")
    expected = (rows // FP8_BLOCK, cols // FP8_BLOCK)
    if tuple(scale_inv.shape) != expected:
        raise NativeFp8Error(f"Scale shape {tuple(scale_inv.shape)} != {expected}")
    # torch.save serializes the WHOLE backing storage of every tensor embedded
    # in the qt. Pieces are sliced from the key-level payload cached in
    # streaming._LazySaveData, so the slice shares that (possibly huge) storage
    # and each written chunk would otherwise re-serialize the full payload (the
    # 423GB runaway). Materialize exact-size storage before building the qt —
    # the same isolation streaming.resolve_data does for dense tensors.
    data = data.contiguous().clone()
    scale_inv = scale_inv.float().contiguous().clone()
    return Float8BlockwiseQTensor(
        (rows, cols), fake_dtype,
        rowwise_data=data, rowwise_scale_inv=scale_inv,
        columnwise_data=None, columnwise_scale_inv=None,
        fp8_dtype=te_dtype.kFloat8E4M3,
        quantizer=Float8BlockQuantizer(fp8_dtype=te_dtype.kFloat8E4M3, rowwise=True,
                                       columnwise=False, force_pow_2_scales=False),
        is_2D_scaled=True,
    )


def requantize_e4m3(dense: torch.Tensor, name: str):
    """Quantize an FP32 (or higher) dense matrix to E4M3 with 128x128 block scales.

    Row-block-major layout, matching TransformerEngine's Float8BlockQuantizer
    and the DSv4 HF scale semantics: scale_inv = block_amax / 448 (FP32), the
    dequantization multiplier. Per-128-row-block independence is what makes
    fused w1|w3 pieces quantize identically piecewise and whole.
    """
    if not dense.is_floating_point():
        raise NativeFp8Error(f"Requantization needs a floating tensor: {name}")
    if not bool(torch.isfinite(dense).all()):
        raise NativeFp8Error(f"Non-finite values cannot be requantized: {name}")
    rows, cols = dense.shape
    blocks = dense.float().view(rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    scale_inv = (amax / FP8_MAX).clamp(min=1e-12)
    quantized = (blocks / scale_inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    data = quantized.view(rows, cols).view(torch.uint8).clone()
    return data, scale_inv.view(rows // FP8_BLOCK, cols // FP8_BLOCK).contiguous()


def payload_for_key(pieces, source):
    """Collect the raw E4M3 payload and scale factors for one stored key.

    ``pieces`` holds every Piece of the key in global-row order: the fused fc1
    w1|w3 rows for one module, or the (expert, half) chunks of the shared
    routed-expert tensor. HF quantization never splits one tensor across HF
    shards, so each piece maps to a contiguous row range of exactly one
    physical tensor (or the whole tensor). Raw uint8 E4M3 bytes and FP32
    scales are concatenated (never qt objects, whose cat dispatch is
    unnecessary here) so a key is ONE row-block-major (rows, cols) payload.
    """
    payloads, scale_invs = [], []
    for piece in pieces:
        spec = source.specs[piece.hf]
        rows, cols = spec.shape
        if rows % FP8_BLOCK or cols % FP8_BLOCK:
            raise NativeFp8Error(
                f"Native FP8 storage needs shapes divisible by {FP8_BLOCK}: "
                f"{piece.hf} is {rows}x{cols}; decode to a dense --dtype instead")
        if spec.kind == "fp8":
            if spec.scale is None:
                raise NativeFp8Error(f"FP8 weight without a scale tensor: {piece.hf}")
            payloads.append(source.reader.get_bytes(spec.name).reshape(rows, cols))
            scale_invs.append(source._scale(spec).reshape(rows // FP8_BLOCK, cols // FP8_BLOCK))
            source.consumed.update((spec.name, spec.scale))
        elif spec.kind == "mxfp4":
            # Decode at FP32 (not the bf16 model dtype) so requantization sees
            # the exact stored values; experts stay on the integer path.
            previous_dtype = source.dtype
            source.dtype = torch.float32
            try:
                dense = source.get_tensor(piece.hf).to(torch.float32).reshape(rows, cols)
            finally:
                source.dtype = previous_dtype
            data, scale_inv = requantize_e4m3(dense, piece.hf)
            payloads.append(data)
            scale_invs.append(scale_inv)
        else:
            raise NativeFp8Error(
                f"Native FP8 output only covers fp8/mxfp4 sources, not {spec.kind}: {piece.hf}")
    data = payloads[0] if len(payloads) == 1 else torch.cat(payloads, dim=0)
    scale_inv = scale_invs[0] if len(scale_invs) == 1 else torch.cat(scale_invs, dim=0)
    return data, scale_inv


def qt_for_payload(data, scale_inv, piece_shape, *, fake_dtype=torch.bfloat16):
    """Build the chunk qt for one piece of a key payload.

    ``piece_shape`` is 2-D (rows, cols) or 3-D (1, rows, cols) for routed
    experts. The qt logical shape must equal the chunk shape: 3-D chunks are
    produced by viewing the 2-D qt (TE dispatches ``aten.view`` and keeps the
    2-D physical storage), never by constructing a 3-D qt (the TE constructor
    flattens logical shapes).
    """
    rows, cols = data.shape
    if len(piece_shape) == 3:
        if piece_shape[0] != 1 or tuple(piece_shape[1:]) != (rows, cols):
            raise NativeFp8Error(
                f"Chunk shape {tuple(piece_shape)} does not match payload {rows}x{cols}")
        return build_native_qt(data, scale_inv, fake_dtype).view(*piece_shape)
    if tuple(piece_shape) != (rows, cols):
        raise NativeFp8Error(
            f"Chunk shape {tuple(piece_shape)} does not match payload {rows}x{cols}")
    return build_native_qt(data, scale_inv, fake_dtype)
