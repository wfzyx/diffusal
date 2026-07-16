"""Auto-imported when the shim dir is on PYTHONPATH.

The MDLM release's 2024-era lightning checkpoints (ar.ckpt) pickle a few plain
numpy objects that torch>=2.6's weights_only unpickler blocks by default.
Allowlist ONLY those benign numpy types — the restricted unpickler stays
active, so this is not a weights_only bypass: arbitrary code execution during
unpickling remains impossible.
"""
import numpy as np
import torch.serialization as _ts

_safe = [np.core.multiarray.scalar, np.core.multiarray._reconstruct,
         np.ndarray, np.dtype]
try:
  from numpy import dtypes as _np_dtypes
  _safe += [getattr(_np_dtypes, _n) for _n in dir(_np_dtypes)
            if _n.endswith('DType')]
except ImportError:
  pass
_ts.add_safe_globals(_safe)
