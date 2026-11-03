"""Regenerate the compact board UI font from an installed Noto Sans SC VF."""

from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
OUTPUT = ROOT / "fonts" / "PulseAtlasCJK.ttf"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(path.read_text(encoding="utf-8") for path in ROOT.glob("*.py"))
    options = subset.Options()
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.name_legacy = True
    options.name_languages = ["*"]
    options.hinting = False
    font = subset.load_font(str(SOURCE), options)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text=text)
    subsetter.subset(font)
    names = font["name"]
    replacements = {
        1: "PulseAtlas CJK", 2: "Regular", 4: "PulseAtlas CJK Regular",
        6: "PulseAtlasCJK-Regular", 16: "PulseAtlas CJK", 17: "Regular",
    }
    for name_id, value in replacements.items():
        names.setName(value, name_id, 3, 1, 0x409)
        names.setName(value, name_id, 1, 0, 0)
    subset.save_font(font, str(OUTPUT), options)
    print("Created {} ({} bytes)".format(OUTPUT, OUTPUT.stat().st_size))


if __name__ == "__main__":
    main()
