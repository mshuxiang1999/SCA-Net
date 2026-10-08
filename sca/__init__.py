"""SCA-Net: semantic-aware cross-modal alignment.

Submodules are intentionally imported explicitly so preprocessing and formula
tests do not have to initialize the optional Transformer stack.
"""

__all__ = ["artifacts", "data", "encoders", "evaluation", "losses", "model", "similarity"]
