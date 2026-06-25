"""Test the corrupt-image guard in detectTools.Detector.bestBoxDetection.

Ultralytics returns an empty results list when it cannot read an image (a
corrupt or truncated file). Before the guard, the following line indexed
results[0] and raised IndexError, killing the whole batch on one bad file. The
guard now returns the standard "nothing detected" tuple instead.

The YOLO model is mocked, so no real weights, GPU or even a working torch are
needed. Where the heavy DeepFaune dependencies are absent (as in this CI
sandbox) they are stubbed just enough to import the module; where they are
present (as on the box) the real packages are imported and used unchanged.
"""

import importlib
import sys
import types

import numpy as np
import pytest


def _ensure_module(name, **attrs):
    """Return the real module if importable, else register a minimal stub.

    Only genuinely missing modules are stubbed, so a machine with the full
    DeepFaune stack installed imports and uses the real packages unchanged.
    """
    try:
        return importlib.import_module(name)
    except Exception:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        for key, value in attrs.items():
            setattr(module, key, value)
        if "." in name:
            parent_name, _, child = name.rpartition(".")
            parent = _ensure_module(parent_name)
            setattr(parent, child, module)
        return module


# Stub the heavy, optional dependencies that detectTools imports at module load.
_ensure_module("cv2")
_ensure_module("torch")
_ensure_module("PIL")
_ensure_module("PIL.Image", fromarray=lambda *a, **k: None)
_ensure_module("torchvision")
_ensure_module("torchvision.ops", batched_nms=lambda *a, **k: None)
_ensure_module("ultralytics", YOLO=object)
_ensure_module("ultralytics.engine")
_ensure_module("ultralytics.engine.results", Results=object)
_ensure_module("yolov5")
_ensure_module("yolov5.utils")
_ensure_module(
    "yolov5.utils.general",
    non_max_suppression=lambda *a, **k: None,
    scale_boxes=lambda *a, **k: None,
)
_ensure_module("yolov5.utils.augmentations", letterbox=lambda *a, **k: None)

detectTools = pytest.importorskip("detectTools")


class _FakeYOLO:
    """Stand-in for the YOLO callable; returns a preset value on every call."""

    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value


class _RaisingYOLO:
    """Stand-in YOLO that raises, to exercise the except branches."""

    def __init__(self, exc):
        self.exc = exc

    def __call__(self, *args, **kwargs):
        raise self.exc


def _make_detector(yolo_value):
    # Bypass __init__ so no weights load; inject the mocked model directly.
    detector = detectTools.Detector.__new__(detectTools.Detector)
    detector.device = "cpu"
    detector.yolo = _FakeYOLO(yolo_value)
    return detector


def _make_raising_detector(exc):
    detector = detectTools.Detector.__new__(detectTools.Detector)
    detector.device = "cpu"
    detector.yolo = _RaisingYOLO(exc)
    return detector


def _assert_nothing_detected(result):
    croppedimage, category, box, count, humanboxes = result
    assert croppedimage is None
    assert category == 0
    assert np.array_equal(box, np.zeros(4))
    assert count == 0
    assert humanboxes == []


def test_best_box_detection_empty_results_returns_nothing():
    # Ultralytics yields an empty list on an unreadable image.
    detector = _make_detector([])
    _assert_nothing_detected(detector.bestBoxDetection("corrupt.jpg"))


def test_best_box_detection_none_results_returns_nothing():
    # Defensive: a None result must also be handled without raising.
    detector = _make_detector(None)
    _assert_nothing_detected(detector.bestBoxDetection("missing.jpg"))


def test_skipped_image_hook_records_unreadable():
    recorded = []
    detectTools.skipped_image_hook = lambda fn, reason: recorded.append((fn, reason))
    try:
        _assert_nothing_detected(_make_detector([]).bestBoxDetection("corrupt.jpg"))
        _assert_nothing_detected(
            _make_raising_detector(FileNotFoundError()).bestBoxDetection("gone.jpg")
        )
        _assert_nothing_detected(
            _make_raising_detector(RuntimeError("boom")).bestBoxDetection("bad.jpg")
        )
    finally:
        detectTools.skipped_image_hook = None
    assert recorded == [
        ("corrupt.jpg", "unreadable"),
        ("gone.jpg", "missing"),
        ("bad.jpg", "error"),
    ]


def test_skipped_image_hook_ignores_non_filename():
    # Video frames and arrays have no filename; the hook must not be called.
    recorded = []
    detectTools.skipped_image_hook = lambda fn, reason: recorded.append(fn)
    try:
        _make_detector([]).bestBoxDetection(np.zeros((4, 4, 3)))
    finally:
        detectTools.skipped_image_hook = None
    assert recorded == []
