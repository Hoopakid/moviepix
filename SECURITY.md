# Security

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/Hoopakid/moviepix/security/advisories/new),
not in public issues.

Note: model checkpoints (`.pt`) are loaded with `torch.load`, which can run code
from untrusted files. Only load checkpoints you trust; released weights will
also be provided as ONNX.
