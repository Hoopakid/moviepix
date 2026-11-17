"""Compose random LPC characters, layer by layer, for every action.

A character is a body type, a skin color and one item per slot (hair, shirt,
legs, weapon ...), each in a random palette color. Rendering stacks the item
layers by z-order for every action sheet, exactly as the LPC generator does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from moviepix.actions import ACTIONS

from .catalog import FRAME, Catalog, Item

# The child body lacks 9 of the 15 actions and has almost no permissive
# clothing, so it is left out for now (docs/decisions 0007).
BODY_WEIGHTS = {"male": 0.4, "female": 0.4, "teen": 0.2}
# Clothing that lacks some actions is still allowed now and then, for variety;
# those characters simply have fewer actions.
PARTIAL_ITEM_PROB = 0.2

# slot, definition folders, probability of wearing something from it.
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
# These may lack an action's sheet (a sword has no climb frames) and are then
# simply not drawn. A missing sheet for anything else leaves that action out,
# so a shirt never vanishes halfway through an animation.
OPTIONAL_SLOTS = {"weapon", "shield", "backpack", "cape", "neck", "arms", "beard", "waist", "hat", "eyes"}


@dataclass
class Choice:
    slot: str
    item: Item
    variant: str | None = None
    # (material, palette set, source color name, target color name)
    colors: list[tuple[str, str, str, str]] = field(default_factory=list)


@dataclass
class Composed:
    body_type: str
    skin: str
    choices: list[Choice]
    sheets: dict[str, np.ndarray]  # action -> uint8 [dirs, frames, 64, 64, 4]
    skipped: dict[str, str]  # action -> why it was left out

    def credits(self) -> list[dict]:
        return [{"item": ch.item.id, **c} for ch in self.choices for c in ch.item.credits_for(self.body_type)]

    def attributes(self) -> dict:
        return {
            "body_type": self.body_type,
            "skin": self.skin,
            "items": [{"slot": c.slot, "item": c.item.id, "name": c.item.name, "variant": c.variant,
                       "colors": [t for _, _, _, t in c.colors]} for c in self.choices],
            "skipped": self.skipped,
        }


@lru_cache(maxsize=8192)
def _load(path: str) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    img = np.asarray(Image.open(p).convert("RGBA"))
    img.setflags(write=False)
    return img


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def recolor(img: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    """Swap palette colors in one lookup pass, so swaps never chain."""
    src = np.array([_hex(c) for c in source], np.uint8)
    dst = np.array([_hex(c) for c in target[: len(src)]], np.uint8)
    src = src[: len(dst)]
    keys = _pack(src)
    order = np.argsort(keys)
    keys, dst = keys[order], dst[order]
    out = img.copy()
    opaque = np.nonzero(img[..., 3] > 0)
    px = _pack(img[opaque][:, :3])
    pos = np.clip(np.searchsorted(keys, px), 0, len(keys) - 1)
    hit = keys[pos] == px
    rows = tuple(o[hit] for o in opaque)
    out[rows + (slice(0, 3),)] = dst[pos[hit]]
    return out


class Composer:
    def __init__(self, catalog: Catalog, permissive: bool = True):
        self.cat = catalog
        self.permissive = permissive

    # ---- file resolution ----------------------------------------------

    def _folder(self, item: Item, layer_idx: int, body_type: str, head: Item | None) -> str | None:
        folder = item.layers[layer_idx].paths.get(body_type)
        if folder is None:
            return None
        if "${head}" in folder:
            mapping = item.replace_in_path.get("head", {})
            key = head.name.replace(" ", "_") if head else None
            if key not in mapping:
                return None
            folder = folder.replace("${head}", mapping[key])
        if "${" in folder:
            return None
        return str(self.cat.sheets / folder)

    @staticmethod
    def _file(folder: str, action: str, variant: str | None) -> str:
        if variant is not None:
            v = Path(folder) / action / f"{variant.replace(' ', '_')}.png"
            if v.exists():
                return str(v)
        return str(Path(folder) / f"{action}.png")

    @lru_cache(maxsize=None)
    def coverage(self, item: Item, body_type: str, head: Item | None) -> int:
        """How many actions this item has sheets for (first variant)."""
        folder = self._folder(item, 0, body_type, head)
        if folder is None:
            return 0
        variant = item.variants[0] if item.variants else None
        return sum(Path(self._file(folder, a.name, variant)).exists() for a in ACTIONS)

    # ---- colors ---------------------------------------------------------

    def _pick_colors(self, rng: np.random.Generator, item: Item, sample: np.ndarray | None, skin: str):
        slots = list(item.recolors[:1])  # secondary color slots keep their drawn colors
        if item.match_body_color and not any(s["material"] == "body" for s in slots):
            slots.insert(0, {"material": "body"})
        flat = sample[sample[..., 3] > 0][:, :3] if sample is not None else None
        out = []
        for slot in slots:
            material = slot["material"]
            sets = self.cat.palette_sets(material)
            base = slot.get("base") or self.cat.palette_base(material)
            best, best_hits = None, -1
            for set_name, pal in sets.items():
                if base not in pal:
                    continue
                hits = 0
                if flat is not None and len(flat):
                    src = np.array([_hex(c) for c in pal[base]], np.uint8)
                    hits = int((flat[:, None, :] == src[None]).all(-1).any(-1).sum())
                if hits > best_hits:
                    best, best_hits = set_name, hits
            if best is None or best_hits == 0:
                continue  # art is not drawn in a known base palette: keep it as drawn
            names = list(sets[best])
            target = skin if material == "body" and skin in sets[best] else names[rng.integers(len(names))]
            out.append((material, best, base, target))
        return out

    def _apply_colors(self, img: np.ndarray, choice: Choice) -> np.ndarray:
        for material, set_name, base, target in choice.colors:
            pal = self.cat.palette_sets(material)[set_name]
            img = recolor(img, pal[base], pal[target])
        return img

    # ---- sampling -------------------------------------------------------

    def sample(self, rng: np.random.Generator, actions=tuple(a.name for a in ACTIONS),
               exclude: frozenset[str] = frozenset()) -> Composed:
        types = list(BODY_WEIGHTS)
        body_type = types[rng.choice(len(types), p=np.array(list(BODY_WEIGHTS.values())))]
        skins = list(self.cat.palette_sets("body").get("ulpc", {"light": []}))
        skin = skins[rng.integers(len(skins))]

        body = self.cat.items["body/body"]
        choices = [Choice("body", body)]
        tags, head = set(body.tags), None
        for slot, folders, prob in SLOTS:
            if rng.random() >= prob:
                continue
            pool = [it for it in self.cat.under(*folders, body_type=body_type, permissive=self.permissive)
                    if set(it.required_tags) <= tags and it.id not in exclude]
            if slot not in OPTIONAL_SLOTS and rng.random() >= PARTIAL_ITEM_PROB:
                full = [it for it in pool if self.coverage(it, body_type, head) >= self.coverage(body, body_type, None)]
                pool = full or pool
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

        sheets, skipped = {}, {}
        for action in actions:
            result = self._render(action, body_type, head, choices)
            if isinstance(result, str):
                skipped[action] = result
            else:
                sheets[action] = result
        return Composed(body_type, skin, choices, sheets, skipped)

    def _render(self, action: str, body_type: str, head: Item | None, choices: list[Choice]):
        base = _load(self._file(self._folder(choices[0].item, 0, body_type, head), action, None))
        if base is None:
            return "body has no sheet"
        h, w = base.shape[:2]
        draws = []
        for ch in choices:
            for li, layer in enumerate(ch.item.layers):
                folder = self._folder(ch.item, li, body_type, head)
                img = _load(self._file(folder, action, ch.variant)) if folder else None
                if img is None or img.shape[:2] != (h, w):
                    if ch.slot in OPTIONAL_SLOTS or li > 0:
                        continue
                    return f"{ch.slot} {ch.item.id} has no matching sheet"
                draws.append((layer.z, len(draws), self._apply_colors(img, ch)))
        sheet = np.zeros((h, w, 4), np.uint8)
        for _, _, img in sorted(draws, key=lambda d: (d[0], d[1])):
            m = img[..., 3] > 0  # pixel art: a drawn pixel replaces what is below
            sheet[m, :3] = img[m, :3]
            sheet[m, 3] = 255
        rows, cols = h // FRAME, w // FRAME
        return sheet.reshape(rows, FRAME, cols, FRAME, 4).transpose(0, 2, 1, 3, 4)
