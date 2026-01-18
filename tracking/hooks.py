"""
Hook handler functions for Instructor event integration.

Provides factory functions that create closures for each hook event type.
These handlers are designed to be stateless - state is managed by InstructorTracker.
"""

from typing import Callable, TYPE_CHECKING

from .metrics import ErrorType

if TYPE_CHECKING:
    from .tracker import InstructorTracker


def classify_error(exception: Exception) -> ErrorType:
    """
    Classify an exception into error categories.

    Args:
        exception: The exception to classify

    Returns:
        ErrorType indicating the category of error
    """
    exc_name = type(exception).__name__.lower()
    exc_str = str(exception).lower()

    # Parse/Validation errors (Pydantic, JSON parsing)
    if any(term in exc_name for term in ['validation', 'parse', 'pydantic', 'json']):
        return ErrorType.PARSE_ERROR

    # API errors (rate limits, auth, provider issues)
    if any(term in exc_name for term in ['rate', 'auth', 'api', 'provider', 'permission']):
        return ErrorType.API_ERROR
    if any(term in exc_str for term in ['rate limit', '429', '401', '403', '500', '502', '503']):
        return ErrorType.API_ERROR

    # Timeout errors
    if any(term in exc_name for term in ['timeout', 'timedout']):
        return ErrorType.TIMEOUT_ERROR

    # Completion errors (catch-all for completion phase)
    if 'completion' in exc_name:
        return ErrorType.COMPLETION_ERROR

    return ErrorType.UNKNOWN


def create_kwargs_handler(
    tracker: 'InstructorTracker',
    call_id: str
) -> Callable:
    """
    Create handler for completion:kwargs hook.

    Fired before each API call attempt with the request arguments.
    Used to track attempt counts and extract request metadata.

    Args:
        tracker: The InstructorTracker instance
        call_id: Unique identifier for this call

    Returns:
        Handler function for the hook
    """
    def handler(*args, **kwargs):
        tracker._on_kwargs(call_id, args, kwargs)
    return handler


def create_response_handler(
    tracker: 'InstructorTracker',
    call_id: str
) -> Callable:
    """
    Create handler for completion:response hook.

    Fired when a successful response is received (before validation).
    Captures token usage from the response.

    Args:
        tracker: The InstructorTracker instance
        call_id: Unique identifier for this call

    Returns:
        Handler function for the hook
    """
    def handler(response):
        tracker._on_response(call_id, response)
    return handler


def create_completion_error_handler(
    tracker: 'InstructorTracker',
    call_id: str
) -> Callable:
    """
    Create handler for completion:error hook.

    Fired on API/completion errors (not validation errors).
    Includes rate limits, auth errors, network issues.

    Args:
        tracker: The InstructorTracker instance
        call_id: Unique identifier for this call

    Returns:
        Handler function for the hook
    """
    def handler(error: Exception):
        tracker._on_completion_error(call_id, error)
    return handler


def create_parse_error_handler(
    tracker: 'InstructorTracker',
    call_id: str
) -> Callable:
    """
    Create handler for parse:error hook.

    Fired on Pydantic validation failures when the response
    doesn't match the expected schema.

    Args:
        tracker: The InstructorTracker instance
        call_id: Unique identifier for this call

    Returns:
        Handler function for the hook
    """
    def handler(error: Exception):
        tracker._on_parse_error(call_id, error)
    return handler


def create_last_attempt_handler(
    tracker: 'InstructorTracker',
    call_id: str
) -> Callable:
    """
    Create handler for completion:last_attempt hook.

    Fired when the final retry attempt is being made.
    Can be used to flag that the next error will be terminal.

    Args:
        tracker: The InstructorTracker instance
        call_id: Unique identifier for this call

    Returns:
        Handler function for the hook
    """
    def handler(*args, **kwargs):
        tracker._on_last_attempt(call_id)
    return handler
