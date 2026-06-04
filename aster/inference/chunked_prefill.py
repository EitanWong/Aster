"""Chunked prefill monkey-patch for mlx_lm.BatchGenerator.

Patching approach (matching vllm-mlx):
1. Replace _next on the instance with _chunked_next
2. _chunked_next processes one prefill chunk, then runs one generation step
3. This prevents long prompts from starving already-generating requests
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def install_chunked_prefill(
    batch_gen: Any,
    chunk_size: int = 1024,
) -> bool:
    """Monkey-patch BatchGenerator for chunked prefill.
    
    Args:
        batch_gen: mlx_lm.generate.BatchGenerator instance
        chunk_size: Max tokens per prefill chunk (default 1024)
    
    Returns:
        True if patch was installed
    """
    if getattr(batch_gen, "_chunked_prefill_installed", False):
        return False

    orig_next = batch_gen._next
    orig_process_prompts = batch_gen._process_prompts

    def _chunked_next(self) -> tuple[list[Any], list[Any]]:
        """Replacement for _next() with chunked prefill.
        
        Flow per call:
        1. If there's a partial prefill in progress, process one more chunk
        2. If no partial prefill, process one chunk from a new prompt
        3. After the chunk, run one generation step for active requests
        """
        # Step 1: Handle generation first (existing generation batch)
        generation_responses: list[Any] = []
        if hasattr(self, "_generation_batch") and self._generation_batch:
            try:
                generation_responses = self._generation_step()
            except Exception:
                pass

        # Step 2: Process one chunk of prefill
        prompt_responses: list[Any] = []
        if self._unprocessed_sequences or getattr(self, "_partial", None):
            try:
                prompt_responses = _process_one_chunk(self, chunk_size)
            except Exception as exc:
                logger.warning("chunked_prefill_chunk_failed", exc_info=True, extra={"error": str(exc)})

        return prompt_responses, generation_responses

    def _process_one_chunk(bg: Any, max_chunk: int) -> list[Any]:
        """Process at most max_chunk tokens from one prompt."""
        # Check for partial prefill state
        partial = getattr(bg, "_partial", None)
        if partial is not None:
            # Resume partial prefill
            uid = partial["uid"]
            tokens = partial["remaining_tokens"]
            cache = partial["cache"]
            sampler = partial.get("sampler")
            logits_processors = partial.get("logits_processors", [])
        elif bg._unprocessed_sequences:
            # Start new prefill from unprocessed queue
            seq = bg._unprocessed_sequences[0]
            uid = seq[0]
            segments = seq[1]
            if not segments or not segments[0]:
                bg._unprocessed_sequences.pop(0)
                return []
            tokens = segments[0]
            cache = bg._make_new_cache() if len(seq) < 4 or seq[3] is None else seq[3]
            sampler = seq[5] if len(seq) > 5 else None
            logits_processors = seq[6] if len(seq) > 6 and seq[6] else []
        else:
            return []

        # Process one chunk
        chunk = tokens[:max_chunk]
        remaining = tokens[max_chunk:]

        # Run model forward for this chunk
        import mlx.core as mx
        logits = bg.language_model(mx.array([chunk]), cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        mx.clear_cache()

        if remaining:
            # Save partial state for next call
            bg._partial = {
                "uid": uid,
                "remaining_tokens": remaining,
                "cache": cache,
                "sampler": sampler,
                "logits_processors": logits_processors,
            }
            # Move uid from unprocessed to keep it alive
            return []
        else:
            # Prefill complete — move to generation batch
            bg._partial = None
            if bg._unprocessed_sequences:
                bg._unprocessed_sequences.pop(0)
            # Insert into generation batch
            try:
                bg._prompt_batch.add(uid, cache, sampler, logits_processors)
            except Exception:
                pass
            return []

    # Install the patch
    batch_gen._next = lambda: _chunked_next(batch_gen)
    batch_gen._chunked_prefill_installed = True
    logger.info("chunked_prefill_installed", extra={"chunk_size": chunk_size})
    return True
