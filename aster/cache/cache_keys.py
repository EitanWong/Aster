from __future__ import annotations

import hashlib


def prefix_hash(model_name: str, tokens: list[int]) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b":")
    h.update(",".join(str(t) for t in tokens).encode("utf-8"))
    return h.hexdigest()
