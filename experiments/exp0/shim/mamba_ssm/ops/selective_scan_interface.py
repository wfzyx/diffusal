def _unavailable(*_args, **_kwargs):
  raise RuntimeError('mamba_ssm stub was called: real package not installed.')


mamba_inner_fn = _unavailable
selective_scan_fn = _unavailable
