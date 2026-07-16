"""Import-only stub. mdlm's models/__init__.py unconditionally imports the
DiMamba backbone; Exp 0 uses only the DiT and AR backbones, so the compiled
mamba kernels are never called. Any actual call fails loudly."""


def _unavailable(*_args, **_kwargs):
  raise RuntimeError(
    'causal_conv1d stub was called: the real package is not installed. '
    'Exp 0 must not touch the DiMamba backbone.')


causal_conv1d_fn = _unavailable
causal_conv1d_update = _unavailable
