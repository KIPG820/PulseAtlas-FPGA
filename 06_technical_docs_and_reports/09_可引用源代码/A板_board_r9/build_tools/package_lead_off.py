"""Create the R9 ECG paper-grid refresh fix board-test archive."""

from hashlib import sha256
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[4]
BOARD = ROOT / "ui" / "pulseatlas" / "board_r9"
BASE = ROOT / "artifacts" / "PulseAtlas_R9_LVGL_HDMI_UI_20261003_ecgfix7.zip"
OUTPUT = ROOT / "artifacts" / "PulseAtlas_R9_LVGL_HDMI_UI_20261003_ecgfix8.zip"
BUILT = ROOT / "bld" / "ecg_nn_r9" / "ecgfix_build"


def main():
    with zipfile.ZipFile(BUILT / "base_hwh_bundle.zip") as bundle:
        hwh = bundle.read("base.hwh")
    assert b"btns_gpio" in hwh and b"ecg_adc_0" in hwh
    replacements = {
        "ecg_frontend.bit": (BUILT / "ecg_frontend.bit").read_bytes(),
        "ecg_frontend.hwh": hwh,
        "ecg_full_pipeline.py": (ROOT / "PYNQ-3.1.1" / "boards" /
                                 "Pynq-Z2" / "ecg_nn_r9" / "board_test" /
                                 "ecg_full_pipeline.py").read_bytes(),
        "pulseatlas_r9_app.py": (BOARD / "pulseatlas_r9_app.py").read_bytes(),
        "pulseatlas_lvgl_backend.py": (BOARD / "pulseatlas_lvgl_backend.py").read_bytes(),
        "pulseatlas_board_render.py": (BOARD / "pulseatlas_board_render.py").read_bytes(),
        "run_pulseatlas_r9.py": (BOARD / "run_pulseatlas_r9.py").read_bytes(),
        "libpulseatlas_lvgl.so": (BOARD / "libpulseatlas_lvgl.so").read_bytes(),
        "nn_accel.py": (ROOT / "PYNQ-3.1.1" / "boards" /
                        "Pynq-Z2" / "ecg_nn_r9" / "board_test" /
                        "nn_accel.py").read_bytes(),
        "fonts/PulseAtlasCJK.ttf": (BOARD / "fonts" / "PulseAtlasCJK.ttf").read_bytes(),
        "build_tools/test_ai_comparison.py": (BOARD / "build_tools" /
                                               "test_ai_comparison.py").read_bytes(),
        "build_tools/test_nn_timing.py": (BOARD / "build_tools" /
                                           "test_nn_timing.py").read_bytes(),
        "build_tools/test_button_navigation.py": (BOARD / "build_tools" /
                                                   "test_button_navigation.py").read_bytes(),
        "build_tools/test_touch_routes.py": (BOARD / "build_tools" /
                                              "test_touch_routes.py").read_bytes(),
        "build_tools/test_lead_off_contract.py": (BOARD / "build_tools" /
                                                  "test_lead_off_contract.py").read_bytes(),
        "build_tools/build_button_overlay.tcl": (BOARD / "build_tools" /
                                                  "build_button_overlay.tcl").read_bytes(),
        "build_tools/build_leadoff_inprocess.tcl": (BOARD / "build_tools" /
                                                     "build_leadoff_inprocess.tcl").read_bytes(),
        "README.md": (BOARD / "README_LEAD_OFF.md").read_bytes(),
    }
    with zipfile.ZipFile(BASE) as original:
        assert "ecg_frontend.bit" in original.namelist()
        payload = {name: original.read(name) for name in original.namelist()
                   if not name.endswith("/") and name not in ("SHA256SUMS.txt",)}
    payload.update(replacements)
    payload["SHA256SUMS.txt"] = "".join(
        "{}  {}\n".format(sha256(data).hexdigest(), name)
        for name, data in sorted(payload.items())
    ).encode("ascii")
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as target:
        for name, data in sorted(payload.items()):
            target.writestr(name, data)
    with zipfile.ZipFile(OUTPUT) as check:
        assert check.testzip() is None
        assert check.read("ecg_frontend.bit") == replacements["ecg_frontend.bit"]
        assert check.read("ecg_frontend.hwh") == hwh
    print("{} ({} bytes, {} files)".format(OUTPUT, OUTPUT.stat().st_size,
                                         len(payload)))


if __name__ == "__main__":
    main()
