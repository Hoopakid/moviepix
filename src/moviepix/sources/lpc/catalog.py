"""Read the LPC generator's sheet and palette definitions as plain data.

Only the JSON and PNG files of the Universal LPC Spritesheet Character
Generator are read; none of its (GPL) code is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BODY_TYPES = ("male", "female", "teen", "muscular", "pregnant", "child")
FRAME = 64

# A credit entry is usable when at least one of the licenses it is offered under
# is permissive (attribution at most). Multi-licensed art may be used under any
# one of its licenses, so "OGA-BY 3.0, CC-BY-SA 3.0, GPL 3.0" qualifies via OGA-BY.
PERMISSIVE = {
    "CC0",
    "CC-BY 3.0", "CC-BY 3.0+", "CC-BY 4.0",
    "OGA-BY 3.0", "OGA-BY 3.0+", "OGA-BY 4.0",
}


def is_permissive(licenses: list[str]) -> bool:
    return any(lic.strip() in PERMISSIVE for lic in licenses)


@dataclass
class Layer:
    z: int
    paths: dict[str, str]  # body type -> folder under spritesheets/


@dataclass(eq=False)  # hashed by identity, so it can key caches
class Item:
    id: str  # definition path relative to sheet_definitions, without .json
    name: str
    layers: list[Layer]
    credits: list[dict]
    variants: list[str] = field(default_factory=list)
    recolors: list[dict] = field(default_factory=list)  # [{material, base?, palettes}]
    match_body_color: bool = False
    tags: list[str] = field(default_factory=list)
    required_tags: list[str] = field(default_factory=list)
    replace_in_path: dict = field(default_factory=dict)

    @property
    def permissive(self) -> bool:
        return bool(self.credits) and all(is_permissive(c.get("licenses", [])) for c in self.credits)

    def credits_for(self, body_type: str) -> list[dict]:
        """Credit entries that cover the folders this item uses for a body type.

        A credit's "file" is a folder prefix; when none matches, all credits apply.
        """
        paths = [layer.paths[body_type] for layer in self.layers if body_type in layer.paths]
        hit = [c for c in self.credits if any(p.startswith(c.get("file", "\0").rstrip("/") + "/") or p.rstrip("/") == c.get("file", "").rstrip("/") for p in paths)]
        return hit or self.credits

    def permissive_for(self, body_type: str) -> bool:
        credits = self.credits_for(body_type)
        return bool(credits) and all(is_permissive(c.get("licenses", [])) for c in credits)

    def supports(self, body_type: str, permissive: bool = True) -> bool:
        if body_type not in self.layers[0].paths:
            return False
        return self.permissive_for(body_type) if permissive else True


def _recolor_slots(raw) -> list[dict]:
    if not isinstance(raw, dict):
        return []
    if "material" in raw:
        return [raw]
    return [v for k, v in sorted(raw.items()) if k.startswith("color_") and isinstance(v, dict) and "material" in v]


class Catalog:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.sheets = self.root / "spritesheets"
        self.items: dict[str, Item] = {}
        defs = self.root / "sheet_definitions"
        for path in sorted(defs.rglob("*.json")):
            if path.name.startswith("meta_"):
                continue
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or "layer_1" not in data:
                continue
            layers = []
            for i in range(1, 10):
                raw = data.get(f"layer_{i}")
                if not raw:
                    break
                if "custom_animation" in raw or raw.get("is_mask"):
                    continue
                paths = {k: v for k, v in raw.items() if k in BODY_TYPES and isinstance(v, str)}
                if paths:
                    layers.append(Layer(int(raw.get("zPos", 100)), paths))
            if not layers:
                continue
            item_id = str(path.relative_to(defs).with_suffix(""))
            self.items[item_id] = Item(
                id=item_id,
                name=data.get("name", item_id),
                layers=layers,
                credits=data.get("credits", []),
                variants=list(dict.fromkeys(data.get("variants", []))),
                recolors=_recolor_slots(data.get("recolors")),
                match_body_color=bool(data.get("match_body_color")),
                tags=data.get("tags", []),
                required_tags=data.get("required_tags", []),
                replace_in_path=data.get("replace_in_path", {}),
            )

    def under(self, *prefixes: str, body_type: str | None = None, permissive: bool = True) -> list[Item]:
        out = [it for it in self.items.values() if it.id.startswith(prefixes)]
        if body_type is not None:
            return [it for it in out if it.supports(body_type, permissive)]
        return [it for it in out if it.permissive] if permissive else out

    # ---- palettes -------------------------------------------------------

    @lru_cache(maxsize=None)
    def palette_sets(self, material: str) -> dict[str, dict[str, list[str]]]:
        """Return {set name: {color name: [hex, ...]}} for a material."""
        folder = self.root / "palette_definitions" / material
        out = {}
        for f in sorted(folder.glob(f"{material}_*.json")):
            data = json.loads(f.read_text())
            flat = {k: v for k, v in data.items() if isinstance(v, list) and v and all(isinstance(c, str) for c in v)}
            if flat:
                out[f.stem.split("_", 1)[1]] = flat
        return out

    @lru_cache(maxsize=None)
    def palette_base(self, material: str) -> str | None:
        meta = self.root / "palette_definitions" / material / f"meta_{material}.json"
        if meta.exists():
            return json.loads(meta.read_text()).get("base")
        return None
