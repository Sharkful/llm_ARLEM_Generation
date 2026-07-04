from types import SimpleNamespace

import pytest

from pipeline import source_relevance
from pipeline.source_relevance import (
    RelevanceReview,
    RelevanceReviewError,
    review_candidate_image,
    review_candidate_text,
    review_candidates,
)


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_call_kwargs = None

    def create_with_completion(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response, SimpleNamespace(usage=None)


class _FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def _candidate(name="Vesta - Snowman Craters", thumbnail_url="https://raw.example/x.png"):
    return SimpleNamespace(name=name, source="nasa3d", thumbnail_url=thumbnail_url)


def test_review_candidate_text_returns_verdict_and_log(monkeypatch):
    fake = _FakeClient(RelevanceReview(matches=False, confidence=0.9, reasoning="craters, not a snowman"))
    monkeypatch.setattr(source_relevance, "get_instructor_client", lambda provider: fake)

    review, entry = review_candidate_text("a snowman", "Vesta - Snowman Craters", "nasa3d")

    assert review.matches is False
    assert entry.purpose == "source_relevance_text"
    assert entry.success is True
    kwargs = fake.chat.completions.last_call_kwargs
    assert kwargs["response_model"] is RelevanceReview
    assert "Vesta - Snowman Craters" in kwargs["messages"][1]["content"]


def test_review_candidate_text_wraps_failure(monkeypatch):
    class _AlwaysFails:
        chat = SimpleNamespace(completions=SimpleNamespace(
            create_with_completion=lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
        ))

    monkeypatch.setattr(source_relevance, "get_instructor_client", lambda provider: _AlwaysFails())

    with pytest.raises(RelevanceReviewError):
        review_candidate_text("a snowman", "X", "y")


def test_review_candidate_image_rejects_non_vision_provider():
    with pytest.raises(RelevanceReviewError, match="anthropic or openai"):
        review_candidate_image("a snowman", "https://raw.example/x.png", provider="google")


def test_review_candidate_image_wraps_fetch_failure(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(source_relevance.requests, "get", boom)

    with pytest.raises(RelevanceReviewError, match="fetch"):
        review_candidate_image("a snowman", "https://raw.example/x.png", provider="anthropic")


def test_review_candidates_filters_and_annotates(monkeypatch):
    def fake_text(description, name, candidate_context="", provider=None, model=None):
        matches = "snowman" in name.lower() and "craters" not in name.lower()
        return (
            RelevanceReview(matches=matches, confidence=0.8, reasoning=f"verdict for {name}"),
            source_relevance.LLMCallEntry(provider="anthropic", model="m", purpose="source_relevance_text"),
        )

    monkeypatch.setattr(source_relevance, "review_candidate_text", fake_text)
    monkeypatch.setattr(
        source_relevance, "review_candidate_image",
        lambda *a, **k: pytest.fail("image review should not run when use_vision=False"),
    )

    candidates = [_candidate("Vesta - Snowman Craters"), _candidate("Snowman Figure 01")]
    reviews, logs = review_candidates("a snowman", candidates, use_vision=False)

    assert [r.matches for r in reviews] == [False, True]
    assert len(logs) == 2


def test_review_candidates_runs_image_review_on_survivors(monkeypatch):
    monkeypatch.setattr(
        source_relevance, "review_candidate_text",
        lambda description, name, candidate_context="", provider=None, model=None: (
            RelevanceReview(matches=True, confidence=0.6, reasoning="text ok"),
            source_relevance.LLMCallEntry(provider="anthropic", model="m", purpose="source_relevance_text"),
        ),
    )
    monkeypatch.setattr(
        source_relevance, "review_candidate_image",
        lambda description, url, provider=None, model=None: (
            RelevanceReview(matches=False, confidence=0.95, reasoning="image shows terrain, not a snowman"),
            source_relevance.LLMCallEntry(provider="anthropic", model="m", purpose="source_relevance_image"),
        ),
    )

    reviews, logs = review_candidates("a snowman", [_candidate()], use_vision=True)

    assert reviews[0].matches is False
    assert reviews[0].reviewed_with_image is True
    assert len(logs) == 2  # text + image


def test_review_candidates_caps_image_reviews(monkeypatch):
    monkeypatch.setattr(
        source_relevance, "review_candidate_text",
        lambda description, name, candidate_context="", provider=None, model=None: (
            RelevanceReview(matches=True, confidence=0.6, reasoning="ok"),
            source_relevance.LLMCallEntry(provider="anthropic", model="m", purpose="source_relevance_text"),
        ),
    )
    image_calls = []

    def fake_image(description, url, provider=None, model=None):
        image_calls.append(url)
        return (
            RelevanceReview(matches=True, confidence=0.9, reasoning="ok"),
            source_relevance.LLMCallEntry(provider="anthropic", model="m", purpose="source_relevance_image"),
        )

    monkeypatch.setattr(source_relevance, "review_candidate_image", fake_image)

    candidates = [_candidate(f"Snowman {i}", f"https://raw.example/{i}.png") for i in range(5)]
    review_candidates("a snowman", candidates, use_vision=True, max_image_reviews=2)

    assert len(image_calls) == 2


def test_review_candidates_survives_individual_failure(monkeypatch):
    def flaky_text(description, name, candidate_context="", provider=None, model=None):
        if name == "bad":
            raise RelevanceReviewError("transient")
        return (
            RelevanceReview(matches=True, confidence=0.7, reasoning="ok"),
            source_relevance.LLMCallEntry(provider="anthropic", model="m", purpose="source_relevance_text"),
        )

    monkeypatch.setattr(source_relevance, "review_candidate_text", flaky_text)
    monkeypatch.setattr(source_relevance, "review_candidate_image",
                        lambda *a, **k: pytest.fail("no survivors expected to reach image review here"))

    reviews, logs = review_candidates(
        "a snowman", [_candidate("bad"), _candidate("good")], use_vision=False
    )

    assert reviews[0].matches is None
    assert reviews[1].matches is True
