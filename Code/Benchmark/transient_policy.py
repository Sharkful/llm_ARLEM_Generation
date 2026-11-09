"""
The transient-error retry policy, shared by everything that re-sends a network
request: LLM calls (``benchmark._transient_retryer``, issue #54), narration TTS
requests (#93) and provisioning downloads (#107).

Only the values live here. This module imports nothing heavy and has no import-time
side effects, so callers that don't need ``benchmark`` (which loads ``.env``, edits
``sys.path`` and imports instructor) can share the policy without it. ``benchmark``
keeps its ``_TRANSIENT_*`` names, set from these, and each caller keeps its own
test hook for the wait.
"""

ATTEMPTS = 3                                          # 1 initial try + 2 retries
STATUS = frozenset({429, 500, 502, 503, 504, 529})    # 529: Anthropic "overloaded"


def default_wait():
    """The back-off between attempts: exponential, 2 to 20 s."""
    import tenacity
    return tenacity.wait_exponential(multiplier=2, min=2, max=20)
