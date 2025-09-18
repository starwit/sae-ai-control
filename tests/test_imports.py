import pytest

def test_rediswriter_import():
    try:
        from sae_ai_control.detectionselector import DetectionSelector
    except ImportError as e:
        pytest.fail(f"Failed to import DetectionSelector: {e}")

    assert DetectionSelector is not None, "DetectionSelector should be imported successfully"