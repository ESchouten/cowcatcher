from aidetector.utils.config import (
    Config,
    DetectionConfig,
    DetectorConfig,
    OnnxConfig,
    YoloConfig,
)
from aidetector.utils.onnx import (
    _STATE,
    _nvtensorrtx_options,
    _patch_inference_session,
    _uses_yolo_profile,
)


def test_tensorrt_rtx_profile_covers_all_configured_detectors():
    config = Config(
        detectors=[
            DetectorConfig(
                detection=DetectionConfig(source=["camera-1", "camera-2"]),
                yolo=YoloConfig(model="small.pt", imgsz=640),
            ),
            DetectorConfig(
                detection=DetectionConfig(source="camera-3"),
                yolo=YoloConfig(model="large.pt", imgsz=960),
            ),
        ]
    )

    assert _nvtensorrtx_options(config) == {
        "nv_profile_min_shapes": "images:1x3x640x640",
        "nv_profile_opt_shapes": "images:2x3x960x960",
        "nv_profile_max_shapes": "images:2x3x960x960",
    }


def test_configured_provider_overrides_ultralytics_provider(monkeypatch):
    calls = []

    class FakeOrt:
        @staticmethod
        def InferenceSession(path_or_bytes, **kwargs):
            calls.append((path_or_bytes, kwargs))
            return "session"

    monkeypatch.setattr(_STATE, "devices", [])
    monkeypatch.setattr(_STATE, "providers", ["UnsupportedByUltralyticsProvider"])
    config = Config(
        detectors=[DetectorConfig(detection=DetectionConfig(source="camera"))],
        onnx=OnnxConfig(provider="UnsupportedByUltralyticsProvider"),
    )

    _patch_inference_session(FakeOrt, config)
    result = FakeOrt.InferenceSession(
        "model.onnx",
        providers=["CPUExecutionProvider"],
        provider_options=[{"device_id": "0"}],
    )

    assert result == "session"
    assert calls == [
        (
            "model.onnx",
            {
                "sess_options": None,
                "providers": [("UnsupportedByUltralyticsProvider", {})],
            },
        )
    ]


def test_miewid_does_not_receive_yolo_tensorrt_profile():
    assert not _uses_yolo_profile("/models/miewid.onnx")
    assert _uses_yolo_profile("/models/detector.onnx")
