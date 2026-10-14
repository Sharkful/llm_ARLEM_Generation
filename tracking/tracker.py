"""
Main InstructorTracker class for token and retry tracking.

Provides automatic tracking of all Instructor API calls via hooks,
with support for per-call metrics and session-level aggregation.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import threading

from .metrics import (
    TokenUsage,
    ErrorRecord,
    AttemptMetrics,
    CallMetrics,
    AggregateMetrics,
    ErrorType,
)
from .hooks import (
    classify_error,
    create_kwargs_handler,
    create_response_handler,
    create_completion_error_handler,
    create_parse_error_handler,
    create_last_attempt_handler,
)
from .pricing import PricingCalculator


class InstructorTracker:
    """
    Main class for tracking token usage and retries across Instructor calls.

    Supports two usage patterns:
    1. Simple drop-in: tracker.wrap(client) for automatic tracking
    2. Context manager: with tracker.track_call(client) for per-call control

    Thread-safe for concurrent usage.

    Example:
        tracker = InstructorTracker(model="gpt-5-nano", provider="openai")
        tracked_client = tracker.wrap(client)
        result = tracked_client.create(...)
        tracker.print_summary()
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        session_id: Optional[str] = None,
        store_call_details: bool = True,
        store_raw_responses: bool = False,
        pricing_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the tracker.

        Args:
            model: Model identifier (e.g., "gpt-5-nano", "claude-3-opus")
            provider: Provider name (e.g., "openai", "anthropic")
            session_id: Optional session identifier for grouping
            store_call_details: Whether to keep individual CallMetrics (True)
                               or only aggregates (False, memory efficient)
            store_raw_responses: Whether to store raw API responses in metrics
            pricing_config: Optional dict with 'input' and 'output' prices per 1M tokens
        """
        self.model = model
        self.provider = provider
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.store_call_details = store_call_details
        self.store_raw_responses = store_raw_responses

        # Pricing calculator
        self.pricing = PricingCalculator(
            config=pricing_config,
            model=model
        )

        # Aggregate metrics
        self.aggregate = AggregateMetrics(
            session_id=self.session_id,
            start_time=datetime.now(),
            model=model,
            provider=provider,
        )

        # Active call tracking (thread-safe)
        self._lock = threading.Lock()
        self._active_calls: Dict[str, CallMetrics] = {}
        self._current_attempt: Dict[str, int] = {}

        # Hook references for cleanup
        self._registered_hooks: Dict[str, List] = {}

    # =========================================================================
    # Public API - Simple Drop-in Usage
    # =========================================================================

    def wrap(self, client) -> 'TrackedClient':
        """
        Wrap an Instructor client for automatic tracking of all calls.

        Usage:
            client = instructor.from_provider("openai/gpt-5-nano")
            tracked_client = tracker.wrap(client)
            result = tracked_client.create(...)  # Automatically tracked
        """
        return TrackedClient(client, self)

    # =========================================================================
    # Public API - Context Manager Usage
    # =========================================================================

    @contextmanager
    def track_call(self, client, call_id: Optional[str] = None):
        """
        Context manager for tracking a single call.

        Usage:
            with tracker.track_call(client) as call_id:
                try:
                    result, completion = client.create_with_completion(...)
                except Exception as e:
                    tracker.record_terminal_error(call_id, e)
                    raise

            metrics = tracker.get_call_metrics(call_id)

        Hooks record what instructor reports. An error it fires no hook for
        (max_tokens truncation, a non-retryable parse failure) is only recorded
        if the caller passes the instructor call's exception to
        ``record_terminal_error``, as above. Exceptions from the caller's own code
        in the block should not be passed, so they aren't pinned on a request.
        TrackedClient does this for you.
        """
        call_id = call_id or str(uuid.uuid4())[:12]

        try:
            self._start_call(client, call_id)
            yield call_id
            self._end_call(call_id, success=True)
        except Exception:
            self._end_call(call_id, success=False)
            raise
        finally:
            self._cleanup_hooks(client, call_id)

    # =========================================================================
    # Public API - Metrics Access
    # =========================================================================

    def get_call_metrics(self, call_id: str) -> Optional[CallMetrics]:
        """Get metrics for a specific call by ID."""
        for call in self.aggregate.calls:
            if call.call_id == call_id:
                return call
        return None

    def get_last_call_metrics(self) -> Optional[CallMetrics]:
        """Get metrics for the most recent call."""
        if self.aggregate.calls:
            return self.aggregate.calls[-1]
        return None

    def get_aggregate_metrics(self) -> AggregateMetrics:
        """Get accumulated metrics across all tracked calls."""
        return self.aggregate

    def get_total_cost(self) -> float:
        """Calculate total cost based on accumulated tokens."""
        return self.pricing.calculate_cost(self.aggregate.total_usage)

    def get_call_cost(self, call_id: str) -> Optional[float]:
        """Calculate cost for a specific call."""
        metrics = self.get_call_metrics(call_id)
        if metrics:
            return self.pricing.calculate_cost(metrics.total_usage)
        return None

    # =========================================================================
    # Public API - Reporting
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Generate a summary dictionary of all metrics."""
        agg = self.aggregate
        return {
            "session_id": self.session_id,
            "model": self.model,
            "provider": self.provider,
            "total_calls": agg.total_calls,
            "successful_calls": agg.successful_calls,
            "failed_calls": agg.failed_calls,
            "success_rate": f"{agg.success_rate:.1%}",
            "tokens": {
                "prompt": agg.total_usage.prompt_tokens,
                "completion": agg.total_usage.completion_tokens,
                "total": agg.total_usage.total_tokens,
                "cached": agg.total_usage.cached_tokens,
                "cache_write": agg.total_usage.cache_write_tokens,
            },
            "retries": {
                "total": agg.total_retries,
                "calls_with_retries": agg.calls_with_retries,
                "retry_rate": f"{agg.retry_rate:.1%}",
                "avg_per_call": f"{agg.avg_retries_per_call:.2f}",
                "max_single_call": agg.max_retries_single_call,
            },
            "errors": {
                "parse_errors": agg.total_parse_errors,
                "api_errors": agg.total_api_errors,
                "completion_errors": agg.total_completion_errors,
            },
            "timing": {
                "total_duration_ms": agg.total_duration_ms,
                "avg_duration_ms": agg.avg_duration_ms,
            },
            "cost": {
                "total_usd": self.get_total_cost(),
                "formatted": f"${self.get_total_cost():.6f}",
            },
        }

    def print_summary(self):
        """Print a formatted summary to console."""
        s = self.summary()
        print(f"\n{'=' * 60}")
        print(f"TRACKING SUMMARY - {s['model']} ({s['provider']})")
        print(f"{'=' * 60}")
        print(f"Session ID: {s['session_id']}")
        print(
            f"\nCalls: {s['total_calls']} total, "
            f"{s['successful_calls']} success, "
            f"{s['failed_calls']} failed ({s['success_rate']})"
        )
        print(f"\nTokens:")
        print(f"  Prompt:     {s['tokens']['prompt']:,}")
        print(f"  Completion: {s['tokens']['completion']:,}")
        print(f"  Total:      {s['tokens']['total']:,}")
        print(f"\nRetries:")
        print(f"  Total retries: {s['retries']['total']}")
        print(
            f"  Calls needing retries: {s['retries']['calls_with_retries']} "
            f"({s['retries']['retry_rate']})"
        )
        print(f"  Avg retries/call: {s['retries']['avg_per_call']}")
        print(f"  Max retries (single call): {s['retries']['max_single_call']}")
        print(f"\nErrors:")
        print(f"  Parse errors: {s['errors']['parse_errors']}")
        print(f"  API errors: {s['errors']['api_errors']}")
        print(f"  Completion errors: {s['errors']['completion_errors']}")
        print(f"\nTiming:")
        print(f"  Total: {s['timing']['total_duration_ms']:.1f}ms")
        print(f"  Average: {s['timing']['avg_duration_ms']:.1f}ms per call")
        print(f"\nCost: {s['cost']['formatted']}")
        print(f"{'=' * 60}\n")

    def reset(self):
        """Reset all metrics for a new benchmarking session."""
        with self._lock:
            self.session_id = str(uuid.uuid4())[:8]
            self.aggregate = AggregateMetrics(
                session_id=self.session_id,
                start_time=datetime.now(),
                model=self.model,
                provider=self.provider,
            )
            self._active_calls.clear()
            self._current_attempt.clear()

    # =========================================================================
    # Internal - Call Lifecycle Management
    # =========================================================================

    def _start_call(self, client, call_id: str):
        """Initialize tracking for a new call."""
        with self._lock:
            call_metrics = CallMetrics(
                call_id=call_id,
                model=self.model,
                provider=self.provider,
                start_time=datetime.now(),
            )
            self._active_calls[call_id] = call_metrics
            self._current_attempt[call_id] = 0

        # Register hooks
        self._register_hooks(client, call_id)

    def _end_call(self, call_id: str, success: bool):
        """Finalize tracking for a completed call."""
        with self._lock:
            if call_id not in self._active_calls:
                return

            call_metrics = self._active_calls[call_id]
            call_metrics.end_time = datetime.now()
            call_metrics.success = success
            call_metrics.final_attempt_number = self._current_attempt.get(call_id, 0)

            # Update aggregate metrics
            self._update_aggregate(call_metrics)

            # Store call details if configured
            if self.store_call_details:
                self.aggregate.calls.append(call_metrics)

            del self._active_calls[call_id]
            del self._current_attempt[call_id]

    def _update_aggregate(self, call: CallMetrics):
        """Update aggregate metrics with completed call data."""
        agg = self.aggregate

        agg.total_calls += 1
        if call.success:
            agg.successful_calls += 1
        else:
            agg.failed_calls += 1

        agg.total_usage = agg.total_usage + call.total_usage

        retry_count = call.retry_count
        agg.total_retries += retry_count
        if retry_count > 0:
            agg.calls_with_retries += 1
        agg.max_retries_single_call = max(
            agg.max_retries_single_call,
            retry_count
        )

        agg.total_parse_errors += call.parse_errors
        agg.total_api_errors += call.api_errors
        agg.total_completion_errors += call.completion_errors

        if call.duration_ms:
            agg.total_duration_ms += call.duration_ms

    # =========================================================================
    # Internal - Hook Registration
    # =========================================================================

    def _register_hooks(self, client, call_id: str):
        """Register all hooks on the client."""
        hooks = [
            ("completion:kwargs", create_kwargs_handler(self, call_id)),
            ("completion:response", create_response_handler(self, call_id)),
            ("completion:error", create_completion_error_handler(self, call_id)),
            ("parse:error", create_parse_error_handler(self, call_id)),
            ("completion:last_attempt", create_last_attempt_handler(self, call_id)),
        ]

        self._registered_hooks[call_id] = []
        for event, handler in hooks:
            client.on(event, handler)
            self._registered_hooks[call_id].append((event, handler))

    def _cleanup_hooks(self, client, call_id: str):
        """Remove registered hooks after call completes."""
        if call_id in self._registered_hooks:
            for event, handler in self._registered_hooks[call_id]:
                try:
                    client.off(event, handler)
                except Exception:
                    pass  # Ignore cleanup errors
            del self._registered_hooks[call_id]

    # =========================================================================
    # Internal - Hook Callbacks
    # =========================================================================

    # One AttemptMetrics per request (issue #71). instructor fires
    # completion:kwargs once before every request, so that hook opens the
    # attempt; the response and error hooks only fill it in. Attempts therefore
    # equal requests by construction, and retry_count = attempts - 1 = reasks.

    def _open_attempt(self, call: CallMetrics, attempt_num: int) -> AttemptMetrics:
        """Start the record for one request (caller holds the lock)."""
        attempt = AttemptMetrics(
            attempt_number=attempt_num,
            timestamp=datetime.now(),
            usage=TokenUsage(),
            success=False,          # set by the response; cleared again by an error
        )
        call.attempts.append(attempt)
        return attempt

    def _current_attempt_record(
        self, call_id: str, call: CallMetrics, for_response: bool = False
    ) -> AttemptMetrics:
        """The attempt the current request belongs to.

        A request gets at most one response. So a response arriving for an attempt
        that already has one, or that already failed, belongs to a request whose
        kwargs hook didn't fire: it opens a new attempt rather than merging two
        requests. Errors always attach to the current attempt (one request can't
        produce two attempts through its errors); an attempt is opened for them
        only if none exists yet."""
        last = call.attempts[-1] if call.attempts else None
        if last is not None and not (for_response and (last.responded or last.error is not None)):
            return last
        self._current_attempt[call_id] = self._current_attempt.get(call_id, 0) + 1
        return self._open_attempt(call, self._current_attempt[call_id])

    def _on_kwargs(self, call_id: str, args, kwargs):
        """Called when completion kwargs are prepared (before each attempt)."""
        with self._lock:
            if call_id in self._active_calls:
                self._current_attempt[call_id] += 1
                call = self._active_calls[call_id]
                self._open_attempt(call, self._current_attempt[call_id])

                # Extract response model name if available
                response_model = kwargs.get('response_model')
                if response_model:
                    call.response_model = getattr(
                        response_model, '__name__',
                        str(response_model)
                    )

    def _on_response(self, call_id: str, response):
        """Called on successful completion response (before it is parsed)."""
        with self._lock:
            if call_id not in self._active_calls:
                return

            call = self._active_calls[call_id]
            attempt = self._current_attempt_record(call_id, call, for_response=True)

            usage = TokenUsage.from_completion(response)
            attempt.usage = attempt.usage + usage
            attempt.responded = True
            attempt.success = True      # provisional: a parse error clears it
            call.total_usage = call.total_usage + usage

    def _on_completion_error(self, call_id: str, error: Exception):
        """Called on completion error (API errors, not validation)."""
        self._record_error(call_id, error, ErrorType.COMPLETION_ERROR)

    def _on_parse_error(self, call_id: str, error: Exception):
        """Called on parse/validation error."""
        self._record_error(call_id, error, ErrorType.PARSE_ERROR)

    def _on_last_attempt(self, call_id: str):
        """Called on final retry attempt."""
        # Could be used to flag that next error is final
        pass

    def _record_error(
        self,
        call_id: str,
        error: Exception,
        default_type: ErrorType
    ):
        """Record an error with proper categorization (from an error hook)."""
        with self._lock:
            if call_id not in self._active_calls:
                return
            call = self._active_calls[call_id]
            self._mark_attempt_failed(
                call, self._current_attempt_record(call_id, call), error, default_type)

    def record_terminal_error(self, call_id: str, error: Exception):
        """Record the exception an instructor call ended with, unless a hook
        already recorded that same error.

        instructor fires no error hook for max_tokens truncation
        (IncompleteOutputException) or for a non-retryable parse failure (raised as
        InstructorRetryException). Pass only exceptions raised by the instructor
        call itself (TrackedClient does), so errors in a caller's own code are
        never pinned on a request.

        - No request was sent (no attempt): nothing is recorded.
        - The last attempt has no error yet: the error is recorded on it.
        - The last attempt already has a different error (instructor raised again
          after a hooked error, e.g. while building the reask): it is recorded as
          the call's ``terminal_error``, without adding an attempt.
        """
        with self._lock:
            call = self._active_calls.get(call_id)
            if call is None or not call.attempts:
                return
            attempt = call.attempts[-1]
            # instructor's retry exception (it always has ``failed_attempts``) wraps
            # the error that ended the call, and its str() is a dump of every
            # attempt: record the cause. Other chained errors are recorded as is.
            cause = error
            if hasattr(error, "failed_attempts") and error.__cause__ is not None:
                cause = error.__cause__
            if attempt.error is not None:
                if (attempt.error.exception_class == type(cause).__name__
                        and attempt.error.message == str(cause)):
                    return          # the hook already recorded this error
                call.terminal_error = self._error_record(cause, attempt.attempt_number,
                                                         ErrorType.COMPLETION_ERROR)
                self._count_error(call, call.terminal_error)
                return
            # A truncation raised by the request itself (no response hook) carries
            # this request's completion. Any other last_completion (e.g. on the
            # retry exception) belongs to an earlier attempt and is not added.
            truncation = next((x for x in (error, cause)
                               if type(x).__name__ == "IncompleteOutputException"), None)
            if truncation is not None and not attempt.responded:
                completion = getattr(truncation, "last_completion", None)
                if completion is not None:
                    usage = TokenUsage.from_completion(completion)
                    attempt.usage = attempt.usage + usage
                    call.total_usage = call.total_usage + usage
            self._mark_attempt_failed(call, attempt, cause, ErrorType.COMPLETION_ERROR)

    def _mark_attempt_failed(
        self,
        call: CallMetrics,
        attempt: AttemptMetrics,
        error: Exception,
        default_type: ErrorType,
    ):
        """Mark ``attempt`` failed with ``error`` and count it (caller holds the
        lock). If the attempt already has an error (a second error hook for one
        request, which instructor 1.17 doesn't send), it keeps the first; the
        second is still counted."""
        record = self._error_record(error, attempt.attempt_number, default_type)
        attempt.success = False
        if attempt.error is None:
            attempt.error = record
        self._count_error(call, record)

    def _error_record(self, error: Exception, attempt_number: int,
                      default_type: ErrorType) -> ErrorRecord:
        error_type = classify_error(error)
        if error_type == ErrorType.UNKNOWN:
            error_type = default_type
        return ErrorRecord(
            error_type=error_type,
            exception_class=type(error).__name__,
            message=str(error),
            timestamp=datetime.now(),
            attempt_number=attempt_number,
            raw_exception=error if self.store_raw_responses else None,
        )

    @staticmethod
    def _count_error(call: CallMetrics, record: ErrorRecord):
        if record.error_type == ErrorType.PARSE_ERROR:
            call.parse_errors += 1
        elif record.error_type == ErrorType.API_ERROR:
            call.api_errors += 1
        else:
            call.completion_errors += 1


class TrackedClient:
    """
    Wrapper that automatically tracks all calls made through the client.

    Provides transparent access to the underlying client's methods.
    """

    def __init__(self, client, tracker: InstructorTracker):
        self._client = client
        self._tracker = tracker

    def create(self, *args, **kwargs):
        """Tracked version of client.create()."""
        return self._tracked(self._client.create, *args, **kwargs)

    def create_with_completion(self, *args, **kwargs):
        """Tracked version of client.create_with_completion()."""
        return self._tracked(self._client.create_with_completion, *args, **kwargs)

    def _tracked(self, method, *args, **kwargs):
        with self._tracker.track_call(self._client) as call_id:
            try:
                return method(*args, **kwargs)
            except Exception as e:
                self._tracker.record_terminal_error(call_id, e)
                raise

    def __getattr__(self, name):
        """Pass through other attributes to underlying client."""
        return getattr(self._client, name)
