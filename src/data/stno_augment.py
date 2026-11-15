"""Perturbation of the STNO mask, so training sees diarization the way inference gets it.

Training masks come from reference segmentation; at inference a diarizer supplies them, with
boundary errors, confused speakers and missed turns. Two perturbations, ported from
`TS-ASR-Whisper`'s collator (`soft_segment_augmentation`, `add_gaussian_noise_and_rescale`) and
applied per item so the work lands in the dataloader workers: relabelling random spans toward
another class, and jittering the probabilities everywhere.

Both keep the mask non-negative and summing to one per frame. Training only.
"""

from typing import Optional

import torch

__all__ = ['perturb_stno_mask']


def _renormalize(mask: torch.Tensor) -> torch.Tensor:
    """Non-negative and summing to 1 over the class axis, without dividing by zero."""
    mask = mask.clamp_min(0.0)
    return mask / mask.sum(dim=0, keepdim=True).clamp_min(1e-8)


def segment_relabel(
    mask: torch.Tensor,
    change_prob: float = 0.2,
    min_seg_len: int = 5,
    max_seg_len: int = 20,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Interpolate random spans of `mask` (C, T) toward a different class.

    Args:
        mask: (C, T) class probabilities.
        change_prob: Probability that any given span is altered.
        min_seg_len, max_seg_len: Span length range, in encoder frames (12.5 Hz, so 5-20 frames
            is 0.4-1.6 s -- the scale of a diarization boundary error).
        generator: Optional RNG, for reproducibility.
    """
    num_classes, num_frames = mask.shape
    out = mask.clone()

    position = 0
    while position < num_frames:
        span = int(torch.randint(min_seg_len, max_seg_len + 1, (1,), generator=generator).item())
        end = min(position + span, num_frames)

        if torch.rand(1, generator=generator).item() < change_prob:
            segment = out[:, position:end]
            dominant = int(segment.mean(dim=1).argmax().item())
            alternatives = [c for c in range(num_classes) if c != dominant]
            if alternatives:
                target = alternatives[int(torch.randint(0, len(alternatives), (1,), generator=generator).item())]
                # Partial rather than absolute: a diarizer's mistakes are rarely confident.
                softness = float(torch.rand(1, generator=generator).item())
                toward = torch.zeros_like(segment)
                toward[target, :] = 1.0
                out[:, position:end] = _renormalize((1.0 - softness) * segment + softness * toward)

        position = end

    return out


def gaussian_jitter(
    mask: torch.Tensor,
    variance: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add Gaussian noise to `mask` (C, T) and restore it to a distribution."""
    noise = torch.randn(mask.shape, generator=generator) * (variance ** 0.5)
    noisy = mask + noise
    # Shift up where the noise pushed a frame negative, so no class is clipped to zero outright.
    noisy = noisy - noisy.amin(dim=0, keepdim=True).clamp(max=0.0)
    return _renormalize(noisy)


def perturb_stno_mask(
    mask: torch.Tensor,
    segment_prob: float = 0.0,
    change_prob: float = 0.2,
    min_seg_len: int = 5,
    max_seg_len: int = 20,
    gaussian_prob: float = 0.0,
    gaussian_var: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply the configured perturbations to one item's (C, T) mask.

    `segment_prob` and `gaussian_prob` are the probabilities of applying each perturbation to
    this item at all; at 0.0 -- the default -- this is a no-op and the reference mask is used.
    """
    if segment_prob > 0.0 and torch.rand(1, generator=generator).item() < segment_prob:
        mask = segment_relabel(mask, change_prob, min_seg_len, max_seg_len, generator)
    if gaussian_prob > 0.0 and torch.rand(1, generator=generator).item() < gaussian_prob:
        mask = gaussian_jitter(mask, gaussian_var, generator)
    return mask
