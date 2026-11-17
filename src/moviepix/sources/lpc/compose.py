"""Compose random LPC characters, layer by layer, for every animation.

DRAFT carried over from before the scaffold; it is reviewed and finished in
milestone 1 (switch to moviepix.actions, wrap as a registered Source).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .catalog import FRAME, Catalog, Item

# Animations as they are named on disk. Directions: rows are up, left, down, right.
ANIMATIONS = (
    "walk", "idle", "run", "slash", "thrust", "spellcast", "shoot", "hurt",
    "climb", "jump", "sit", "emote", "combat_idle", "backslash", "halfslash",
)
DIRECTIONS = ("up", "left", "down", "right")

BODY_WEIGHTS = {"male": 0.35, "female": 0.35, "teen": 0.2, "child": 0.1}

# slot name, definition folders, probability of wearing something from it.
# "optional" slots may be missing from an animation without spoiling it
# (a sword has no climb frames); clothing may not.
SLOTS = (
    ("head", ("head/heads/human",), 1.0),
    ("eyes", ("head/eyes",), 0.7),
    ("hair", tuple(f"hair/{h}" for h in ("afro", "bob", "braids", "curly", "long", "short", "spiky", "xlong", "bald")), 0.9),
    ("beard", ("hair/beards", "hair/mustaches"), 0.15),
    ("torso", ("torso/shirts", "torso/dresses", "torso/torso_chainmail"), 0.9),
    ("torso_over", ("torso/armour", "torso/jacket", "torso/aprons"), 0.3),
    ("waist", ("torso/waist",), 0.25),
    ("cape", ("torso/cape",), 0.12),
    ("backpack", ("torso/backpack",), 0.08),
    ("legs", ("legs",), 0.9),
    ("feet", ("feet/shoes", "feet/boots", "feet/feet_sandals", "feet/feet_armour", "feet/socks"), 0.75),
    ("arms", ("arms",), 0.2),
    ("hat", ("headwear/hats", "headwear/helmets", "headwear/coverings"), 0.3),
    ("neck", ("headwear/neck",), 0.1),
    ("weapon", ("weapons/sword", "weapons/blunt", "weapons/magic", "weapons/ranged", "tools"), 0.45),
    ("shield", ("weapons/shields",), 0.12),
)
OPTIONAL_SLOTS = {"weapon", "shield", "backpack", "cape", "neck", "arms", "beard"}


@dataclass
class Choice:
    slot: str
    item: Item
    variant: str | None = None
    colors: dict[int, tuple[str, str, str]] = field(default_factory=dict)  # slot idx -> (material, set, target)


@dataclass
class Character:
    body_type: str
    skin: str
    choices: list[Choice]
    animations: dict[str, np.ndarray]  # name -> uint8 [dirs, frames, 64, 64, 4]
    skipped: dict[str, str]  # animation -> reason it was left out

    def credits(self) -> list[dict]:
        out = []
        for ch in self.choices:
            for c in ch.item.credits_for(self.body_type):
                out.append({"item": ch.item.id, **c})
        return out

    def describe(self) -> dict:
        return {
            "body_type": self.body_type,
            "skin": self.skin,
            "items": [{"slot": c.slot, "item": c.item.id, "variant": c.variant,
                       "colors": {str(k): list(v) for k, v in c.colors.items()}} for c in self.choices],
            "animations": {k: list(v.shape[:2]) for k, v in self.animations.items()},
            "skipped": self.skipped,
        }


@lru_cache(maxsize=4096)
def _load(path: str) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    return np.asarray(Image.open(p).convert("RGBA"))


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def recolor(img: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    out = img.copy()
    rgb = img[..., :3]
    opaque = img[..., 3] > 0
    for s, t in zip(source, target):
        m = opaque & np.all(rgb == np.array(_hex(s), np.uint8), axis=-1)
        out[m, :3] = _hex(t)
    return out


class Composer:
    def __init__(self, catalog: Catalog, permissive: bool = True):
        self.cat = catalog
        self.permissive = permissive

    # ---- file resolution ----------------------------------------------

    def _folder(self, item: Item, layer_idx: int, body_type: str, head: Item | None) -> str | None:
        layer = item.layers[layer_idx]
        folder = layer.paths.get(body_type)
        if folder is None:
            return None
        if "${head}" in folder:
            if head is None:
                return None
            mapping = item.replace_in_path.get("head", {})
            key = head.name.replace(" ", "_")
            if key not in mapping:
                return None
            folder = folder.replace("${head}", mapping[key])
        if "${" in folder:
            return None
        return str(self.cat.sheets / folder)

    def _file(self, folder: str, anim: str, variant: str | None) -> str:
        if variant is not None:
            v = Path(folder) / anim / f"{variant.replace(' ', '_')}.png"
            if v.exists():
                return str(v)
        return str(Path(folder) / f"{anim}.png")

    # ---- color choice ---------------------------------------------------

    def _pick_colors(self, rng: np.random.Generator, item: Item, sample: np.ndarray | None, skin: str) -> dict:
        """For each recolor slot choose (material, palette set, target color name)."""
        colors = {}
        slots = list(item.recolors)
        if item.match_body_color and not any(s["material"] == "body" for s in slots):
            slots.insert(0, {"material": "body"})
        for idx, slot in enumerate(slots[:1] if not item.match_body_color else slots[:2]):
            material = slot["material"]
            sets = self.cat.palette_sets(material)
            base = slot.get("base") or self.cat.palette_base(material)
            if not sets or base is None:
                continue
            best, best_hits = None, -1
            for set_name, pal in sets.items():
                if base not in pal:
                    continue
                hits = 0
                if sample is not None:
                    flat = sample[sample[..., 3] > 0][:, :3]
                    src = np.array([_hex(c) for c in pal[base]], np.uint8)
                    hits = int((flat[:, None, :] == src[None]).all(-1).any(-1).sum())
                if hits > best_hits:
                    best, best_hits = set_name, hits
            if best is None:
                continue
            names = list(sets[best])
            target = skin if material == "body" and skin in sets[best] else names[rng.integers(len(names))]
            colors[idx] = (material, best, target)
        return colors

    def _apply_colors(self, img: np.ndarray, choice: Choice) -> np.ndarray:
        for material, set_name, target in choice.colors.values():
            pal = self.cat.palette_sets(material)[set_name]
            base = self.cat.palette_base(material)
            src = pal.get(base)
            if src and target != base:
                img = recolor(img, src, pal[target])
        return img

    # ---- sampling -------------------------------------------------------

    def sample(self, rng: np.random.Generator, animations=ANIMATIONS) -> Character:
        types = list(BODY_WEIGHTS)
        body_type = types[rng.choice(len(types), p=np.array(list(BODY_WEIGHTS.values())))]
        skin_names = list(self.cat.palette_sets("body").get("ulpc", {"light": []}))
        skin = skin_names[rng.integers(len(skin_names))]

        body = self.cat.items["body/body"]
        choices = [Choice("body", body)]
        tags = set(body.tags)
        head = None
        for slot, folders, prob in SLOTS:
            if rng.random() >= prob:
                continue
            pool = [it for it in self.cat.under(*folders, body_type=body_type, permissive=self.permissive)
                    if set(it.required_tags) <= tags]
            if not pool:
                continue
            item = pool[rng.integers(len(pool))]
            choices.append(Choice(slot, item))
            if slot == "head":
                head = item
                tags |= set(item.tags)

        for ch in choices:
            if ch.item.variants:
                ch.variant = ch.item.variants[rng.integers(len(ch.item.variants))]
            folder = self._folder(ch.item, 0, body_type, head)
            sample = _load(self._file(folder, "walk", ch.variant)) if folder else None
            ch.colors = self._pick_colors(rng, ch.item, sample, skin)

        out, skipped = {}, {}
        for anim in animations:
            result = self._render(anim, body_type, head, choices)
            if isinstance(result, str):
                skipped[anim] = result
            else:
                out[anim] = result
        return Character(body_type, skin, choices, out, skipped)

    def _render(self, anim: str, body_type: str, head: Item | None, choices: list[Choice]):
        base = _load(self._file(self._folder(choices[0].item, 0, body_type, head), anim, None))
        if base is None:
            return "body has no frames"
        h, w = base.shape[:2]
        if h % FRAME or w % FRAME:
            return f"odd body sheet size {w}x{h}"
        draws = []
        for ch in choices:
            for li, layer in enumerate(ch.item.layers):
                folder = self._folder(ch.item, li, body_type, head)
                img = _load(self._file(folder, anim, ch.variant)) if folder else None
                if img is None or img.shape[:2] != (h, w):
                    if ch.slot in OPTIONAL_SLOTS or li > 0:
                        continue
                    return f"{ch.slot} {ch.item.id} has no matching {anim} sheet"
                draws.append((layer.z, len(draws), self._apply_colors(img, ch)))
        canvas = np.zeros((h, w, 4), np.float32)
        for _, _, img in sorted(draws, key=lambda d: (d[0], d[1])):
            a = img[..., 3:4].astype(np.float32) / 255
            canvas[..., :3] = img[..., :3] * a + canvas[..., :3] * (1 - a)
            canvas[..., 3:4] = a + canvas[..., 3:4] * (1 - a)
        sheet = canvas.round().clip(0, 255).astype(np.uint8)
        sheet[..., 3] = np.where(sheet[..., 3] > 0, 255, 0)  # pixel art: alpha is on or off
        rows, cols = h // FRAME, w // FRAME
        return sheet.reshape(rows, FRAME, cols, FRAME, 4).transpose(0, 2, 1, 3, 4)
