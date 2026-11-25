"""Artists whose work must never be trained on.

DEV_ARTISTS made the out-of-style packs we tune against; FINAL_ARTISTS made the final
test, which is scored once. Nothing by either may reach a training set, including their
other packs. Every entry of an extra training dataset therefore carries attributes.artist
and is checked by `check_manifest`.
"""

from __future__ import annotations

import re

DEV_ARTISTS = ("shade", "puny characters", "pixel-boy", "aaa", "ninja adventure", "superpowers", "sparklin labs",
               "chasersgaming", "antifarea", "bit era", "imogia", "pav creations", "calciumtrice", "luizmelo")
FINAL_ARTISTS = ("ansimuz", "gibbongl", "gamekrazzy", "hormelz", "robocelot", "axulart")
TRAIN_ARTISTS = ("ordinary bumblebee", "lexlom", "foozle", "jik-a-4")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def check_manifest(entries: list[dict], allowed=TRAIN_ARTISTS) -> None:
    """Raise unless every entry names an artist on the training list and none on a held-out list."""
    ok = {_norm(a) for a in allowed}
    banned = {_norm(a) for a in (*DEV_ARTISTS, *FINAL_ARTISTS)}
    for e in entries:
        artist = (e.get("attributes") or {}).get("artist")
        if not artist:
            raise ValueError(f"{e['id']}: an extra training character must name its artist (attributes.artist)")
        key = _norm(artist)
        if key in banned:
            raise ValueError(f"{e['id']}: {artist!r} is a held-out artist and must never be trained on")
        if key not in ok:
            raise ValueError(f"{e['id']}: {artist!r} is not on the training list in data/heldout.py")
