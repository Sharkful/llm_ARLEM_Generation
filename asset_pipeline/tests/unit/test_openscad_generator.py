import pytest

from models.generation_models import OpenSCADPlan
from models.spec_models import AssetSpec
from pipeline import openscad_generator as gen


def _spec():
    return AssetSpec(object_id="bracket_01", description="an L-shaped mounting bracket")


def _plan(bounds=(0.05, 0.05, 0.03)):
    return OpenSCADPlan(
        parameters={"width": 0.05, "height": 0.05, "depth": 0.03},
        scad_source="cube([0.05, 0.05, 0.03]);",
        expected_bounds_m=list(bounds),
        notes="simple bracket",
    )


def test_bounds_diverge_flags_large_deviation():
    assert gen._bounds_diverge([0.05, 0.05, 0.03], [0.05, 0.05, 0.03], 0.20) is False
    assert gen._bounds_diverge([0.10, 0.05, 0.03], [0.05, 0.05, 0.03], 0.20) is True  # 100% off in X


def test_bounds_diverge_handles_zero_expected():
    # expected 0 (e.g. a flat quad's Z) shouldn't divide by zero
    assert gen._bounds_diverge([0.0, 1.0, 1.0], [0.0, 1.0, 1.0], 0.20) is False
    assert gen._bounds_diverge([0.5, 1.0, 1.0], [0.0, 1.0, 1.0], 0.20) is True


def test_generate_parametric_asset_succeeds_on_first_try(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    def fake_compile(scad_source, scad_path, stl_path):
        scad_path.parent.mkdir(parents=True, exist_ok=True)
        scad_path.write_text(scad_source, encoding="utf-8")
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        stl_path.write_bytes(b"fake stl content")
        return True, ""

    monkeypatch.setattr(gen, "compile_scad", fake_compile)
    monkeypatch.setattr(gen, "measure_stl_bounds_m", lambda path: [0.05, 0.05, 0.03])

    calls = []

    def fake_plan_fn(spec, repair_context):
        calls.append(repair_context)
        return _plan(), None

    result, logs = gen.generate_parametric_asset(_spec(), "bracket_01", generate_plan_fn=fake_plan_fn)

    assert result.success is True
    assert result.bounds_diverge is False
    assert len(result.repair_attempts) == 1
    assert len(calls) == 1
    assert calls[0] is None  # first attempt has no repair context


def test_generate_parametric_asset_repairs_on_compile_failure(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    call_count = {"n": 0}

    def fake_compile(scad_source, scad_path, stl_path):
        call_count["n"] += 1
        scad_path.parent.mkdir(parents=True, exist_ok=True)
        scad_path.write_text(scad_source, encoding="utf-8")
        if call_count["n"] == 1:
            return False, "ERROR: syntax error on line 1"
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        stl_path.write_bytes(b"fake stl content")
        return True, ""

    monkeypatch.setattr(gen, "compile_scad", fake_compile)
    monkeypatch.setattr(gen, "measure_stl_bounds_m", lambda path: [0.05, 0.05, 0.03])

    repair_contexts = []

    def fake_plan_fn(spec, repair_context):
        repair_contexts.append(repair_context)
        return _plan(), None

    result, logs = gen.generate_parametric_asset(_spec(), "bracket_01", generate_plan_fn=fake_plan_fn)

    assert result.success is True
    assert len(result.repair_attempts) == 2
    assert result.repair_attempts[0].success is False
    assert result.repair_attempts[1].success is True
    assert repair_contexts[0] is None
    assert repair_contexts[1] is not None
    assert "syntax error" in repair_contexts[1][1]


def test_generate_parametric_asset_gives_up_after_attempt_limit(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    def always_fail_compile(scad_source, scad_path, stl_path):
        scad_path.parent.mkdir(parents=True, exist_ok=True)
        scad_path.write_text(scad_source, encoding="utf-8")
        return False, "ERROR: still broken"

    monkeypatch.setattr(gen, "compile_scad", always_fail_compile)

    def fake_plan_fn(spec, repair_context):
        return _plan(), None

    result, logs = gen.generate_parametric_asset(_spec(), "bracket_01", generate_plan_fn=fake_plan_fn)

    assert result.success is False
    assert len(result.repair_attempts) == gen.REPAIR_ATTEMPT_LIMIT
    assert "failed after" in result.error_message


def test_generate_parametric_asset_flags_bounds_divergence(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    def fake_compile(scad_source, scad_path, stl_path):
        scad_path.parent.mkdir(parents=True, exist_ok=True)
        scad_path.write_text(scad_source, encoding="utf-8")
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        stl_path.write_bytes(b"fake stl content")
        return True, ""

    monkeypatch.setattr(gen, "compile_scad", fake_compile)
    # Actual bounds wildly different from the LLM's claimed expected_bounds_m.
    monkeypatch.setattr(gen, "measure_stl_bounds_m", lambda path: [5.0, 5.0, 5.0])

    def fake_plan_fn(spec, repair_context):
        return _plan(bounds=(0.05, 0.05, 0.03)), None

    result, logs = gen.generate_parametric_asset(_spec(), "bracket_01", generate_plan_fn=fake_plan_fn)

    assert result.success is True
    assert result.bounds_diverge is True


def test_generate_parametric_asset_wraps_plan_failure(tmp_path, monkeypatch):
    import config
    from models.log_models import LLMCallEntry

    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)

    failed_entry = LLMCallEntry(
        provider="anthropic", model="claude-sonnet-4-6", purpose="openscad_generation",
        success=False, error_message="boom",
    )

    def raising_plan_fn(spec, repair_context):
        raise gen.OpenSCADGenerationError("boom", failed_entry)

    with pytest.raises(gen.OpenSCADGenerationError):
        gen.generate_parametric_asset(_spec(), "bracket_01", generate_plan_fn=raising_plan_fn)
