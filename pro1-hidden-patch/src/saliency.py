"""ResNet-18 tools that find and check the patch.

1. A 3x3 median removes most of the adversarial noise. Integrated gradients from the
   smoothed image to the image give the part of the class margin that each pixel causes.
2. best_boxes finds the boxes of an allowed size with the largest sum of a map.
3. Causal check: smooth only the inside of a candidate box and measure the drop of the margin.
   Only the patch causes a large drop. A decoy causes almost no drop.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .eda import SIZE
from .evaluate import BOX


def smooth(x: torch.Tensor) -> torch.Tensor:
    """Return the 3x3 median of each image. It removes most of the adversarial noise."""
    patches = F.pad(x, (1, 1, 1, 1), mode="reflect").unfold(2, 3, 1).unfold(3, 3, 1)
    return patches.reshape(*patches.shape[:4], 9).median(-1).values


def margin(logits: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Return the logit of class c minus the largest other logit."""
    own = logits.gather(1, c[:, None])[:, 0]
    other = logits.scatter(1, c[:, None], float("-inf")).max(1).values
    return own - other


def box_sizes(area_min: float, area_max: float, ratio_max: float) -> list[tuple[int, int]]:
    return [(w, h) for w in range(4, SIZE + 1) for h in range(4, SIZE + 1)
            if area_min <= w * h / SIZE**2 <= area_max and 1 / ratio_max <= w / h <= ratio_max]


def integrated_gradients(model, x, xs, c, steps):
    total = torch.zeros_like(x)
    for a in (torch.arange(steps, device=x.device) + 0.5) / steps:
        z = (xs + a * (x - xs)).requires_grad_(True)
        (grad,) = torch.autograd.grad(margin(model(z), c).sum(), z)
        total += grad
    return ((x - xs) * total / steps).sum(1)


def best_boxes(score_map: torch.Tensor, sizes, k: int) -> torch.Tensor:
    """Return the k boxes with the largest map sum, at most one box for each size. Shape (N, k, 4)."""
    integral = F.pad(score_map.cumsum(1).cumsum(2), (1, 0, 1, 0))
    sums, boxes = [], []
    for w, h in sizes:
        s = integral[:, h:, w:] - integral[:, :-h, w:] - integral[:, h:, :-w] + integral[:, :-h, :-w]
        best, pos = s.flatten(1).max(1)
        y, x = pos // s.shape[2], pos % s.shape[2]
        sums.append(best)
        boxes.append(torch.stack([x, y, x + w, y + h], 1))
    sums, boxes = torch.stack(sums, 1), torch.stack(boxes, 1)
    top = sums.topk(min(k, len(sizes)), 1).indices
    return boxes.gather(1, top[..., None].expand(-1, -1, 4))


def box_masks(boxes: torch.Tensor) -> torch.Tensor:
    """Return masks (N, K, 1, SIZE, SIZE) with 1 inside each box."""
    r = torch.arange(SIZE, device=boxes.device)
    b = boxes[..., None]
    cols = (r >= b[:, :, 0]) & (r < b[:, :, 2])
    rows = (r >= b[:, :, 1]) & (r < b[:, :, 3])
    return (rows[..., :, None] & cols[..., None, :]).float()[:, :, None]


@torch.no_grad()
def causal_drops(model, x, xs, c, m0, boxes, chunk):
    """Return the margin drop (N, K) when the inside of each box is replaced by the smoothed image."""
    n, k = boxes.shape[:2]
    mask = box_masks(boxes)
    mixed = (x[:, None] * (1 - mask) + xs[:, None] * mask).flatten(0, 1)
    cc = c.repeat_interleave(k)
    m = torch.cat([margin(model(mixed[i:i + chunk]), cc[i:i + chunk]) for i in range(0, len(mixed), chunk)])
    return m0[:, None] - m.view(n, k)


def answers_frame(ids, boxes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(boxes.astype(int), columns=BOX).assign(id=list(ids))[["id", *BOX]]
