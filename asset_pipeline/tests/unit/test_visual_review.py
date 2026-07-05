from types import SimpleNamespace

import pytest

from pipeline import visual_review
from pipeline.visual_review import (
    AssetVisualReview,
    VisualReviewError,
    review_asset_render,
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


def test_review_asset_render_rejects_non_vision_provider(tmp_path):
    image = tmp_path / "render.png"
    image.write_bytes(b"fake png bytes")

    with pytest.raises(VisualReviewError, match="anthropic or openai"):
        review_asset_render("a methane molecule", image, provider="google")


def test_review_asset_render_requires_existing_image(tmp_path):
    with pytest.raises(VisualReviewError, match="No render found"):
        review_asset_render("a methane molecule", tmp_path / "missing.png", provider="anthropic")


def test_review_asset_render_returns_verdict_and_log(monkeypatch, tmp_path):
    image = tmp_path / "render.png"
    image.write_bytes(b"fake png bytes")
    fake = _FakeClient(AssetVisualReview(
        matches=False, confidence=0.9,
        reasoning="the bonds don't reach the hydrogen atoms",
        suggested_fix="recompute each bond's length/rotation from the real atom positions",
    ))
    monkeypatch.setattr(visual_review, "get_instructor_client", lambda provider: fake)

    review, entry = review_asset_render("a methane molecule", image, provider="anthropic")

    assert review.matches is False
    assert "bonds" in review.reasoning
    assert review.suggested_fix
    assert entry.purpose == "visual_review"
    assert entry.success is True
    kwargs = fake.chat.completions.last_call_kwargs
    assert kwargs["response_model"] is AssetVisualReview
    # image content block present alongside the text description
    content = kwargs["messages"][1]["content"]
    assert any(block.get("type") == "image" for block in content)
    assert any("a methane molecule" in block.get("text", "") for block in content)


def test_review_asset_render_wraps_llm_failure(monkeypatch, tmp_path):
    image = tmp_path / "render.png"
    image.write_bytes(b"fake png bytes")

    class _AlwaysFails:
        chat = SimpleNamespace(completions=SimpleNamespace(
            create_with_completion=lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
        ))

    monkeypatch.setattr(visual_review, "get_instructor_client", lambda provider: _AlwaysFails())

    with pytest.raises(VisualReviewError):
        review_asset_render("a methane molecule", image, provider="anthropic")
