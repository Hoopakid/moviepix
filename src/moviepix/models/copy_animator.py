"""The main model: copy by position, decoded iteratively.

For every pixel of every output frame the model picks one of
    transparent
    "new color"  (a part hidden in the still pose; a small RGB head paints it)
    copy reference pixel j   (j over the reference's opaque pixels)

Pieces
    reference encoder   a U-Net over the still sprite, run once per animation,
                        giving a key per opaque reference pixel
    frame decoder       a conditioned U-Net per output frame; its input is the
                        reference, the partly decided frame ("canvas") and a
                        mask of which pixels are already decided. Frames of one
                        animation attend to each other at the bottleneck.
    copy attention      query (per output pixel) . key (per reference pixel)

Training (masked, MaskGIT-style): in each frame a random share of the true
pixels is shown as already decided and the model predicts the rest. Often
everything is hidden, which is exactly one-shot decoding. Several reference
pixels may have the target color; the loss rewards probability on any of them
(marginal likelihood), so no labels beyond the frames themselves are needed.

Decoding: `steps` rounds. Each round decides the most confident undecided
pixels, on a cosine schedule, so later pixels see earlier choices.
steps=1 is one-shot. temperature > 0 samples, which gives variations.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import Conditioning, CondUNet, FrameAttention, groups, masked_select_frames
from .interface import Animator, Batch
from .registry import register

TRANSPARENT, NEW = 0, 1  # class ids; copy classes start at 2


def _pack(rgb: torch.Tensor) -> torch.Tensor:
    q = (rgb.float() * 255).round().long()
    return (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]


@register("copy_animator")
class CopyAnimator(Animator):
    def __init__(self, size: int = 64, channels=(32, 64, 128, 128), ref_channels=(32, 64, 128),
                 cond_dim: int = 128, dim: int = 64, frame_heads: int = 4,
                 full_mask_prob: float = 0.25, steps: int = 8, loss_chunk: int = 32,
                 alpha_head: bool = False, self_cond_prob: float = 0.0):
        super().__init__()
        self.loss_chunk = loss_chunk
        # alpha_head: a separate "is this pixel drawn?" output (like the U-Net baseline),
        # instead of "transparent" competing with every copy choice (decision 0018).
        self.alpha_head = alpha_head
        # self_cond_prob: share of training frames whose "already decided" pixels are the
        # model's own most confident predictions, exactly as in iterative decoding.
        self.self_cond_prob = self_cond_prob
        self.size = size
        self.steps = steps
        self.full_mask_prob = full_mask_prob
        self.cond = Conditioning(cond_dim)
        self.ref_cond = nn.Parameter(torch.zeros(cond_dim))
        self.ref_unet = CondUNet(ref_channels, cond_dim, in_ch=4)
        self.to_key = nn.Conv2d(self.ref_unet.out_channels, dim, 1)
        self.dec_unet = CondUNet(channels, cond_dim, in_ch=9)  # reference 4 + canvas 4 + known 1
        self.frame_attn = FrameAttention(channels[-1], frame_heads)
        out = self.dec_unet.out_channels
        self.dec_norm = nn.Sequential(nn.GroupNorm(groups(out), out), nn.SiLU())
        self.to_query = nn.Conv2d(out, dim, 1)
        self.special = nn.Conv2d(out, 2, 1)  # transparent, new
        self.new_rgb = nn.Conv2d(out, 3, 1)
        self.alpha = nn.Conv2d(out, 1, 1) if alpha_head else None
        self.pos = nn.Parameter(torch.randn(dim, size, size) * 0.02)
        self.scale = dim ** -0.5

    # ---- pieces ---------------------------------------------------------

    def encode_reference(self, reference):
        """Keys and colors of the opaque reference pixels, padded to the batch max.

        Returns keys [B, K, D], colors [B, K, 3], valid [B, K].
        """
        B = reference.shape[0]
        feat = self.ref_unet(reference, self.ref_cond.expand(B, -1))
        keys = (self.to_key(feat) + self.pos).flatten(2).transpose(1, 2)  # [B, S*S, D]
        opaque = reference[:, 3].flatten(1) > 0.5
        k = max(int(opaque.sum(1).max()), 1)
        order = torch.argsort((~opaque).to(torch.int8), dim=1, stable=True)[:, :k]  # opaque first
        valid = torch.gather(opaque, 1, order)
        keys = torch.gather(keys, 1, order[..., None].expand(-1, -1, keys.shape[-1]))
        rgb = reference[:, :3].flatten(2).transpose(1, 2)
        colors = torch.gather(rgb, 1, order[..., None].expand(-1, -1, 3))
        return keys, colors, valid

    def features(self, reference, canvas, known, b, t, cond, batch: int, t_max: int):
        """Per output pixel: query [N, P, D], transparent/new logits [N, P, 2], new RGB [N, 3, S, S],
        and alpha logits [N, P] (None without the alpha head)."""
        x = torch.cat([reference[b], canvas, known], 1)
        hook = lambda h: self.frame_attn(h, b, t, batch, t_max)  # noqa: E731
        feat = self.dec_norm(self.dec_unet(x, cond, mid_hook=hook))
        q = (self.to_query(feat) + self.pos).flatten(2).transpose(1, 2)
        special = self.special(feat).flatten(2).transpose(1, 2).float()
        alpha = self.alpha(feat).flatten(1).float() if self.alpha is not None else None
        return q, special, torch.sigmoid(self.new_rgb(feat)), alpha

    def copy_logits(self, q, keys, valid):
        """[n, P, D] x [n, K, D] -> [n, P, K], in float32 (they feed a log-sum-exp)."""
        copy = torch.einsum("npd,nkd->npk", q.float(), keys.float()) * self.scale
        return copy.masked_fill(~valid[:, None, :], float("-inf"))

    def decode(self, reference, canvas, known, ref, b, t, cond, batch: int, t_max: int):
        """Full logits [N, P, 2+K] and new-color RGB, for inference."""
        keys, _, valid = ref
        q, special, new_rgb, _ = self.features(reference, canvas, known, b, t, cond, batch, t_max)
        parts = [torch.cat([special[i:i + 32], self.copy_logits(q[i:i + 32], keys[b[i:i + 32]], valid[b[i:i + 32]])], -1)
                 for i in range(0, len(b), 32)]
        return torch.cat(parts, 0), new_rgb

    def _choose(self, reference, canvas, known, ref, b, t, cond, batch, t_max, temperature, generator,
                chunk: int | None = None):
        """Each pixel's choice and its probability, computed a chunk of frames at a time
        so the full [frames, pixels, reference pixels] table never exists at once."""
        keys, _, valid = ref
        chunk = chunk or max(self.loss_chunk, 4)
        q, special, new_rgb, alpha = self.features(reference, canvas, known, b, t, cond, batch, t_max)
        choices, confs = [], []
        for i in range(0, len(b), chunk):
            j = slice(i, i + chunk)
            logits = torch.cat([special[j], self.copy_logits(q[j], keys[b[j]], valid[b[j]])], -1)
            if alpha is not None:
                logits[..., TRANSPARENT] = float("-inf")  # colour choice only; alpha decides "drawn or not"
            probs = F.softmax(logits / temperature if temperature > 0 else logits, -1)
            if temperature > 0:
                n, p, c = probs.shape
                choice = torch.multinomial(probs.reshape(n * p, c), 1, generator=generator).reshape(n, p)
            else:
                choice = probs.argmax(-1)
            conf = torch.gather(probs, -1, choice[..., None])[..., 0]
            if alpha is not None:
                p_on = torch.sigmoid(alpha[j])
                if temperature > 0:
                    on = torch.rand(p_on.shape, generator=generator, device=p_on.device) < p_on
                else:
                    on = p_on > 0.5
                choice = torch.where(on, choice, torch.full_like(choice, TRANSPARENT))
                conf = torch.where(on, p_on * conf, 1 - p_on)
            choices.append(choice)
            confs.append(conf)
        return torch.cat(choices), torch.cat(confs), new_rgb

    def _chunk_nll(self, q, keys, valid, special, tgt_key, color_key, opaque, hidden):
        """Summed loss over the hidden pixels of a chunk of frames.

        nll = logsumexp(all choices) - logsumexp(correct choices); several reference
        pixels may share the target color and all of them count as correct.
        With the alpha head, this is the colour loss of drawn pixels only.
        """
        copy = self.copy_logits(q, keys, valid)
        if self.alpha is not None:
            lse_all = torch.logsumexp(torch.cat([special[..., NEW:NEW + 1], copy], -1), -1)
            match = (tgt_key[..., None] == color_key[:, None, :]) & valid[:, None, :]
            found = match.any(-1)
            lse_copy = torch.logsumexp(copy.masked_fill(~match, float("-inf")), -1)
            picked = torch.where(found, lse_copy, special[..., NEW])
            return ((lse_all - picked) * hidden * opaque).sum()
        lse_all = torch.logsumexp(torch.cat([special, copy], -1), -1)
        match = (tgt_key[..., None] == color_key[:, None, :]) & valid[:, None, :]
        found = match.any(-1)
        lse_copy = torch.logsumexp(copy.masked_fill(~match, float("-inf")), -1)
        picked = torch.where(opaque, torch.where(found, lse_copy, special[..., NEW]), special[..., TRANSPARENT])
        return ((lse_all - picked) * hidden).sum()

    @staticmethod
    def render(choice, new_rgb, colors_b):
        """Class ids [N, S*S] -> RGBA [N, 4, S, S]."""
        N, P = choice.shape
        S = int(math.isqrt(P))
        idx = (choice - 2).clamp(min=0)
        rgb = torch.gather(colors_b.float(), 1, idx[..., None].expand(-1, -1, 3))  # [N, P, 3]
        rgb = torch.where((choice == NEW)[..., None], new_rgb.float().flatten(2).transpose(1, 2), rgb)
        alpha = (choice != TRANSPARENT).float()[..., None]
        return torch.cat([rgb * alpha, alpha], -1).transpose(1, 2).reshape(N, 4, S, S)

    # ---- training -------------------------------------------------------

    def loss(self, batch: Batch):
        ref_img, n_frames = batch.reference, batch.n_frames
        B, S = ref_img.shape[0], ref_img.shape[-1]
        t_max = int(n_frames.max())
        b, t, _ = masked_select_frames(n_frames, t_max)
        N, P = len(b), S * S
        target = batch.target[b, t]  # [N, 4, S, S]

        # Which pixels are hidden, per frame:
        #   full mask (one-shot practice)       nothing given
        #   random hints                        a random share of the TRUE pixels is given
        #   self-conditioned                    the model's OWN most confident predictions are
        #                                       given, as in iterative decoding, and the rest
        #                                       must be predicted (correcting its own guesses)
        dev = ref_img.device
        u = torch.rand(N, device=dev)
        share = torch.cos(u * math.pi / 2)
        mode = torch.rand(N, device=dev)
        full = mode < self.full_mask_prob
        selfc = (~full) & (mode < self.full_mask_prob + self.self_cond_prob)
        share = torch.where(full, torch.ones_like(share), share)
        hidden = torch.rand(N, P, device=dev) < share[:, None]  # [N, P]

        ref = self.encode_reference(ref_img)
        cond = self.cond(batch.action[b], batch.direction[b], t, n_frames[b])
        canvas = target * (~hidden).float().reshape(N, 1, S, S)
        if selfc.any():
            with torch.no_grad():
                zeros = torch.zeros_like(target)
                choice, conf, new0 = self._choose(ref_img, zeros, zeros[:, :1], ref, b, t, cond, B, t_max, 0.0, None)
                guess = self.render(choice, new0, ref[1][b]).to(target.dtype)
                n_known = ((1 - share) * P).long()  # same share as the random hints
                rank = conf.argsort(1, descending=True).argsort(1)
                own_known = rank < n_known[:, None]
            hidden = torch.where(selfc[:, None], ~own_known, hidden)
            m = selfc[:, None, None, None]
            canvas = torch.where(m, guess * (~hidden).float().reshape(N, 1, S, S), canvas)
        known = (~hidden).float().reshape(N, 1, S, S)

        q, special, new_rgb, alpha = self.features(ref_img, canvas, known, b, t, cond, B, t_max)

        keys, colors, valid = ref
        tgt = target.flatten(2).transpose(1, 2)  # [N, P, 4]
        opaque = tgt[..., 3] > 0.5
        tgt_key, color_key = _pack(tgt[..., :3]), _pack(colors)
        hid = hidden.float()
        # The [frames, pixels, reference pixels] table is the memory hog: build it a
        # chunk of frames at a time and recompute it in the backward pass (checkpoint).
        total = q.new_zeros((), dtype=torch.float32)
        for i in range(0, N, self.loss_chunk):
            j = slice(i, i + self.loss_chunk)
            bj = b[j]
            total = total + checkpoint(self._chunk_nll, q[j], keys[bj], valid[bj], special[j], tgt_key[j],
                                       color_key[bj], opaque[j], hid[j], use_reentrant=False)
        if alpha is not None:
            nll = total / (hidden & opaque).sum().clamp(min=1)
            alpha_bce = (F.binary_cross_entropy_with_logits(alpha, opaque.float(), reduction="none") * hid).sum() \
                / hidden.sum().clamp(min=1)
        else:
            nll = total / hidden.sum().clamp(min=1)
            alpha_bce = nll.new_zeros(())
        with torch.no_grad():
            found = torch.zeros_like(opaque)
            for i in range(0, N, self.loss_chunk):
                j = slice(i, i + self.loss_chunk)
                bj = b[j]
                found[j] = ((tgt_key[j][..., None] == color_key[bj][:, None, :]) & valid[bj][:, None, :]).any(-1)

        new_mask = (hidden & opaque & ~found).reshape(N, 1, S, S).float()
        new_l1 = ((new_rgb.float() - target[:, :3]).abs() * new_mask).sum() / (new_mask.sum() * 3).clamp(min=1)
        loss = nll + new_l1 + alpha_bce
        return loss, {"loss": loss.item(), "nll": nll.item(), "alpha_bce": alpha_bce.item(),
                      "new_rgb_l1": new_l1.item(), "hidden_share": hidden.float().mean().item(),
                      "self_cond_share": selfc.float().mean().item()}

    # ---- inference ------------------------------------------------------

    ORDERS = ("confidence", "balanced", "random", "silhouette")

    @staticmethod
    def _top(score, k, allowed):
        """Mask of the k highest-scoring allowed pixels per row (k: [N])."""
        score = score.masked_fill(~allowed, float("-inf"))
        rank = score.argsort(1, descending=True).argsort(1)
        return (rank < k[:, None]) & allowed

    @torch.no_grad()
    def generate(self, reference, action, direction, n_frames, text=None, steps: int | None = None,
                 temperature: float = 0.0, generator: torch.Generator | None = None,
                 canvas: torch.Tensor | None = None, known: torch.Tensor | None = None, order: str = "random"):
        """Decode in `steps` rounds (1 = one-shot). Optional canvas/known [B, T, ...]
        fix some pixels in advance (editing).

        `order` decides which undecided pixels are committed each round:
            confidence  the most confident first (MaskGIT). Commits empty background
                        first and leaves the character for the last round.
            balanced    the most confident first, but background and character pixels
                        in proportion, so decided pixels look like training's random mix
            random      a random order, exactly like training
            silhouette  round 1 commits everything predicted empty (the silhouette);
                        the character is then decided over the remaining rounds
        """
        if order not in self.ORDERS:
            raise ValueError(f"order must be one of {self.ORDERS}")
        steps = steps or self.steps
        B, S = reference.shape[0], reference.shape[-1]
        t_max = int(n_frames.max())
        b, t, _ = masked_select_frames(n_frames, t_max)
        N, P = len(b), S * S
        ref = self.encode_reference(reference)
        _, colors, _ = ref
        cond = self.cond(action[b], direction[b], t, n_frames[b])

        cur = canvas[b, t].float() if canvas is not None else reference.new_zeros(N, 4, S, S).float()
        done = (known[b, t].flatten(1) > 0.5) if known is not None else torch.zeros(N, P, dtype=torch.bool, device=reference.device)
        start_hidden = (~done).sum(1)  # [N]
        start_char = None  # silhouette order: character pixels left after round 1

        for s in range(1, steps + 1):
            choice, conf, new_rgb = self._choose(reference, cur.to(reference.dtype),
                                                 done.float().reshape(N, 1, S, S).to(reference.dtype),
                                                 ref, b, t, cond, B, t_max, temperature, generator)
            if order == "random":
                score = torch.rand(conf.shape, generator=generator, device=conf.device) if generator is not None \
                    else torch.rand(conf.shape, device=conf.device)
            else:
                score = conf
                if temperature > 0:  # a little randomness in which pixels go first
                    g = -torch.log(-torch.log(torch.rand(conf.shape, generator=generator, device=conf.device).clamp(1e-9, 1)))
                    score = conf.log() + temperature * g
            undecided = ~done
            empty = choice == TRANSPARENT

            if s == steps:
                commit = undecided
            elif order == "silhouette" and steps > 1:
                if s == 1:
                    commit = undecided & empty
                    start_char = (undecided & ~empty).sum(1)
                else:
                    keep = torch.floor(start_char.float() * math.cos(math.pi / 2 * (s - 1) / (steps - 1))).long()
                    commit = self._top(score, (undecided.sum(1) - keep).clamp(min=0), undecided)
            else:
                keep_hidden = torch.floor(start_hidden.float() * math.cos(math.pi / 2 * s / steps)).long()
                n_commit = (undecided.sum(1) - keep_hidden).clamp(min=0)  # [N]
                if order == "balanced":
                    n_char = (undecided & ~empty).sum(1)
                    k_char = torch.round(n_commit * n_char / undecided.sum(1).clamp(min=1)).long()
                    commit = self._top(score, k_char, undecided & ~empty) | self._top(score, n_commit - k_char, undecided & empty)
                else:
                    commit = self._top(score, n_commit, undecided)

            rgba = self.render(choice, new_rgb, colors[b])
            cur = torch.where(commit.reshape(N, 1, S, S), rgba, cur)
            done = done | commit

        out = reference.new_zeros(B, t_max, 4, S, S)
        out[b, t] = cur.to(out.dtype)
        return out

    def forward(self, reference, action, direction, n_frames, text=None):
        return self.generate(reference, action, direction, n_frames, text)
