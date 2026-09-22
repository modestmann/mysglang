from __future__ import annotations

from collections.abc import Sequence


class NGramProposer:
    """Propose a continuation copied from an earlier matching history suffix.

    This is deliberately model-free: it searches only tokens already present in
    one request, preferring the longest suffix and then its most recent earlier
    occurrence. The target model remains authoritative and verifies the whole
    proposed chain in one causal forward.
    """

    def __init__(self, *, min_match: int, max_match: int) -> None:
        if min_match <= 0:
            raise ValueError("min_match must be positive")
        if max_match < min_match:
            raise ValueError("max_match must be at least min_match")
        self.min_match = min_match
        self.max_match = max_match

    def propose(self, token_ids: Sequence[int], max_tokens: int) -> tuple[int, ...]:
        """Return at most ``max_tokens`` known tokens after a suffix match."""
        if max_tokens <= 0:
            return ()
        history = tuple(token_ids)
        # Besides the suffix itself, an earlier match must leave at least one
        # already-observed token that can serve as a proposal.
        max_match = min(self.max_match, len(history) - 1)
        for match_length in range(max_match, self.min_match - 1, -1):
            suffix_start = len(history) - match_length
            suffix = history[suffix_start:]
            # For an equal-length match, later start means a more recent source.
            # Overlap is valid because every proposed token still comes from the
            # already-observed history, never from recursively invented output.
            for start in range(suffix_start - 1, -1, -1):
                if history[start : start + match_length] != suffix:
                    continue
                continuation_start = start + match_length
                continuation_end = min(len(history), continuation_start + max_tokens)
                continuation = history[continuation_start:continuation_end]
                if continuation:
                    return continuation
        return ()
