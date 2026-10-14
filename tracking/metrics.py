"""
Data structures for tracking Instructor API calls.

Provides standardized dataclasses for token usage, error tracking,
per-attempt metrics, per-call metrics, and aggregate session metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Any
from enum import Enum


class ErrorType(Enum):
    """Categorizes errors for separate tracking."""
    PARSE_ERROR = "parse_error"           # Pydantic validation failures
    API_ERROR = "api_error"               # Provider API errors (rate limits, auth, etc.)
    COMPLETION_ERROR = "completion_error" # General completion failures
    TIMEOUT_ERROR = "timeout_error"       # Request timeouts
    UNKNOWN = "unknown"


@dataclass
class TokenUsage:
    """
    Standardized token usage across providers.

    Supports addition for accumulation across retries.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Extended details (optional, provider-dependent). Both cache counts are part
    # of prompt_tokens on every provider (Anthropic's are folded in below).
    cached_tokens: int = 0          # input read from the provider's prompt cache
    cache_write_tokens: int = 0     # input written to the cache (OpenAI GPT-5.6+, Anthropic)
    reasoning_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Enable accumulation via addition."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @classmethod
    def from_completion(cls, completion: Any) -> 'TokenUsage':
        """
        Factory to extract usage from various provider response formats.

        Handles OpenAI, Anthropic, and Google Gemini (google.genai) responses.
        """
        if completion is None:
            return cls()

        # Handle raw response wrapper
        if hasattr(completion, '_raw_response'):
            completion = completion._raw_response

        # --- Google Gemini (google.genai) responses ---
        # Gemini uses .usage_metadata with *_token_count fields instead of .usage
        if hasattr(completion, 'usage_metadata') and completion.usage_metadata is not None:
            um = completion.usage_metadata
            prompt = getattr(um, 'prompt_token_count', 0) or 0
            candidates = getattr(um, 'candidates_token_count', 0) or 0
            total = getattr(um, 'total_token_count', 0) or 0
            cached = getattr(um, 'cached_content_token_count', 0) or 0
            thinking = getattr(um, 'thoughts_token_count', 0) or 0

            # NOTE: In GENAI_STRUCTURED_OUTPUTS mode the Pydantic response schema is
            # passed as a native parameter (response_schema). Gemini counts ONLY the
            # text prompt here; the schema tokens are billed as input but reported
            # nowhere in usage_metadata (total == prompt + candidates + thinking
            # exactly). They are recovered separately at the benchmark layer via
            # estimate_gemini_schema_tokens(); we keep prompt as the raw reported value.

            return cls(
                prompt_tokens=prompt,
                completion_tokens=candidates,
                total_tokens=total,
                cached_tokens=cached,
                reasoning_tokens=thinking,
            )

        # --- OpenAI / Anthropic responses (use .usage) ---
        if not hasattr(completion, 'usage') or completion.usage is None:
            return cls()

        usage = completion.usage

        # --- OpenAI Responses API (GPT-6): input_tokens already includes cached
        # tokens; both cache counts and reasoning sit in the *_details objects.
        input_details = getattr(usage, 'input_tokens_details', None)
        if input_details is not None:
            output_details = getattr(usage, 'output_tokens_details', None)
            prompt = getattr(usage, 'input_tokens', 0) or 0
            output = getattr(usage, 'output_tokens', 0) or 0
            return cls(
                prompt_tokens=prompt,
                completion_tokens=output,   # includes reasoning tokens
                total_tokens=getattr(usage, 'total_tokens', None) or prompt + output,
                cached_tokens=getattr(input_details, 'cached_tokens', 0) or 0,
                cache_write_tokens=getattr(input_details, 'cache_write_tokens', 0) or 0,
                reasoning_tokens=getattr(output_details, 'reasoning_tokens', 0) or 0,
            )

        # Extract base token counts
        cached = 0
        cache_write = 0
        prompt = getattr(usage, 'prompt_tokens', None)
        if prompt is None:
            # Anthropic uses input_tokens, which excludes cache reads and writes;
            # fold them in so prompt_tokens means all input on every provider.
            cached = getattr(usage, 'cache_read_input_tokens', 0) or 0
            cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            prompt = (getattr(usage, 'input_tokens', 0) or 0) + cached + cache_write

        completion_tokens = getattr(usage, 'completion_tokens', None)
        if completion_tokens is None:
            # Anthropic uses output_tokens
            completion_tokens = getattr(usage, 'output_tokens', 0)

        total = getattr(usage, 'total_tokens', None)
        if total is None:
            total = (prompt or 0) + (completion_tokens or 0)

        # Extract extended details if available (OpenAI)
        reasoning = 0

        prompt_details = getattr(usage, 'prompt_tokens_details', None)
        if prompt_details:
            cached = getattr(prompt_details, 'cached_tokens', 0) or 0
            # GPT-5.6 and later bill cache writes at 1.25x input and report them here
            cache_write = getattr(prompt_details, 'cache_write_tokens', 0) or 0

        completion_details = getattr(usage, 'completion_tokens_details', None)
        if completion_details:
            reasoning = getattr(completion_details, 'reasoning_tokens', 0) or 0

        return cls(
            prompt_tokens=prompt or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total or 0,
            cached_tokens=cached,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning,
        )


def served_model_of(completion: Any) -> Optional[str]:
    """The model id a provider response says answered it: Gemini
    ``model_version``, OpenAI / Anthropic ``model``. A provider can serve a
    request with a different model than the one asked for (Google routes
    gemini-3.5-flash to gemini-3.6-flash since 2026-10-08)."""
    if completion is None:
        return None
    if hasattr(completion, '_raw_response'):
        completion = completion._raw_response
    for attr in ('model_version', 'model'):
        val = getattr(completion, attr, None)
        if isinstance(val, str) and val:
            return val.removeprefix('models/')
    return None


@dataclass
class ErrorRecord:
    """Detailed record of an error occurrence."""
    error_type: ErrorType
    exception_class: str
    message: str
    timestamp: datetime
    attempt_number: int
    raw_exception: Optional[Exception] = None


@dataclass
class AttemptMetrics:
    """Metrics for a single attempt within a call."""
    attempt_number: int
    timestamp: datetime
    usage: TokenUsage
    success: bool
    error: Optional[ErrorRecord] = None
    duration_ms: Optional[float] = None
    responded: bool = False         # a response arrived for this request
    served_model: Optional[str] = None  # model id the response reports


@dataclass
class CallMetrics:
    """
    Complete metrics for a single API call (including all retries).

    Tracks all attempts, accumulated token usage, error counts,
    and timing information.
    """
    call_id: str
    model: str
    provider: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Accumulated usage across all attempts
    total_usage: TokenUsage = field(default_factory=TokenUsage)

    # Individual attempt tracking
    attempts: List[AttemptMetrics] = field(default_factory=list)

    # Error categorization
    parse_errors: int = 0
    api_errors: int = 0
    completion_errors: int = 0

    # Final status
    success: bool = False
    final_attempt_number: int = 0
    # An error that ended the call after its last attempt was already marked
    # failed by a hook (e.g. raised while instructor built the reask). None
    # otherwise: an ending error with no hook is recorded on the attempt itself.
    terminal_error: Optional[ErrorRecord] = None

    # Request context (optional)
    response_model: Optional[str] = None

    @property
    def total_attempts(self) -> int:
        """Total number of attempts made."""
        return len(self.attempts)

    @property
    def retry_count(self) -> int:
        """Number of retries (attempts - 1, or 0 if no retries)."""
        return max(0, len(self.attempts) - 1)

    @property
    def duration_ms(self) -> Optional[float]:
        """Total duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    @property
    def errors(self) -> List[ErrorRecord]:
        """All errors from all attempts, then the call's terminal error if any."""
        errs = [a.error for a in self.attempts if a.error is not None]
        if self.terminal_error is not None:
            errs.append(self.terminal_error)
        return errs

    @property
    def final_error(self) -> Optional[ErrorRecord]:
        """For a failed call, the recorded error that ended it; None for a call
        that succeeded, or whose ending error was never recorded (one raised
        before any request was sent)."""
        if self.success or not self.attempts:
            return None
        return self.terminal_error or self.attempts[-1].error


@dataclass
class AggregateMetrics:
    """
    Accumulated metrics across multiple calls for benchmarking.

    Provides summary statistics, totals, and optionally stores
    individual call records for detailed analysis.
    """
    session_id: str
    start_time: datetime
    model: str
    provider: str

    # Totals
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

    # Token accumulation
    total_usage: TokenUsage = field(default_factory=TokenUsage)

    # Retry statistics
    total_retries: int = 0
    calls_with_retries: int = 0
    max_retries_single_call: int = 0

    # Error breakdown
    total_parse_errors: int = 0
    total_api_errors: int = 0
    total_completion_errors: int = 0

    # Timing
    total_duration_ms: float = 0.0

    # Individual call records (optional, for detailed analysis)
    calls: List[CallMetrics] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Proportion of successful calls."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def retry_rate(self) -> float:
        """Proportion of calls that required retries."""
        if self.total_calls == 0:
            return 0.0
        return self.calls_with_retries / self.total_calls

    @property
    def avg_retries_per_call(self) -> float:
        """Average number of retries per call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_retries / self.total_calls

    @property
    def avg_duration_ms(self) -> float:
        """Average duration per call in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls
