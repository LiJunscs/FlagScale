"""DeepSeek-V4 checkpoint conversion helpers."""


class MappingError(RuntimeError):
    """Existing checkpoint/config metadata cannot satisfy the DSv4 mapping."""
