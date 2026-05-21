import pytest

from narration_definitions import ARLabNarrationScript, NarrationClip, NarrationModule, SaySegment
from pipeline.filename_utils import enumerate_clips, make_clip_filename, make_safe_slug


class TestMakeSafeSlug:
    def test_spaces_become_underscores(self):
        assert make_safe_slug("hello world") == "hello_world"

    def test_lowercase(self):
        assert make_safe_slug("Hello World") == "hello_world"

    def test_punctuation_stripped(self):
        assert make_safe_slug("loss: surface!") == "loss_surface"

    def test_multiple_spaces_collapsed(self):
        assert make_safe_slug("a   b") == "a_b"

    def test_leading_trailing_stripped(self):
        assert make_safe_slug("  hello  ") == "hello"

    def test_long_string_truncated(self):
        long = "a" * 100
        assert len(make_safe_slug(long)) <= 50

    def test_empty_string(self):
        assert make_safe_slug("") == ""

    def test_hyphens_collapsed_to_underscores(self):
        # hyphens are treated as separators and normalized to underscores for consistent filenames
        assert make_safe_slug("step-by-step") == "step_by_step"

    def test_numbers_preserved(self):
        assert make_safe_slug("module 2 intro") == "module_2_intro"


class TestMakeClipFilename:
    def test_with_title(self):
        assert make_clip_filename(1, 1, "Introduction") == "m01_c001_introduction.mp3"

    def test_without_title(self):
        assert make_clip_filename(1, 1, None) == "m01_c001.mp3"

    def test_module_padding(self):
        assert make_clip_filename(9, 1, None) == "m09_c001.mp3"
        assert make_clip_filename(10, 1, None) == "m10_c001.mp3"

    def test_clip_padding(self):
        assert make_clip_filename(1, 99, None) == "m01_c099.mp3"
        assert make_clip_filename(1, 100, None) == "m01_c100.mp3"

    def test_title_slug_applied(self):
        assert make_clip_filename(2, 3, "Loss Surface: Overview!") == "m02_c003_loss_surface_overview.mp3"

    def test_different_modules_same_title_no_collision(self):
        assert make_clip_filename(1, 1, "Introduction") != make_clip_filename(2, 1, "Introduction")

    def test_whitespace_only_title_treated_as_no_title(self):
        assert make_clip_filename(1, 1, "   ") == "m01_c001.mp3"


class TestEnumerateClips:
    def _make_clip(self, title: str | None = None) -> NarrationClip:
        return NarrationClip(title=title, segments=[SaySegment(type="say", text="Hello.")])

    def _make_script(self, module_clip_counts: list[int]) -> ARLabNarrationScript:
        modules = [
            NarrationModule(clips=[self._make_clip(f"Clip {i+1}") for i in range(n)])
            for n in module_clip_counts
        ]
        return ARLabNarrationScript(modules=modules)

    def test_single_module_single_clip(self):
        results = enumerate_clips(self._make_script([1]))
        assert len(results) == 1
        m_idx, c_idx, clip, fname = results[0]
        assert m_idx == 1
        assert c_idx == 1
        assert fname == "m01_c001_clip_1.mp3"

    def test_two_modules(self):
        results = enumerate_clips(self._make_script([2, 2]))
        assert len(results) == 4
        assert results[0][:2] == (1, 1)
        assert results[1][:2] == (1, 2)
        assert results[2][:2] == (2, 1)
        assert results[3][:2] == (2, 2)

    def test_clip_indices_reset_per_module(self):
        results = enumerate_clips(self._make_script([3, 3]))
        assert results[3][1] == 1
        assert results[4][1] == 2
        assert results[5][1] == 3

    def test_no_filename_collisions_across_modules(self):
        results = enumerate_clips(self._make_script([2, 2]))
        fnames = [r[3] for r in results]
        assert len(fnames) == len(set(fnames))
