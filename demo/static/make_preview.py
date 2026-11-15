"""Draw the link-preview card (Open Graph image, 1200x630) shown when the demo's URL is shared.

    python static/make_preview.py      # writes static/preview.png
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK, MUTED, ACCENT, BG = (17, 24, 39), (107, 114, 128), (14, 159, 110), (248, 250, 252)
PALETTE = [(37, 99, 235), (220, 38, 38), (22, 163, 74), (147, 51, 234)]  # the app's first four speakers


def tint(c, a):
    return tuple(round(255 + (v - 255) * a) for v in c)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
d.rectangle([0, 0, W, 10], fill=ACCENT)
d.text((80, 64), "Pandemonium", font=ImageFont.truetype(BOLD, 76), fill=INK)
d.text((84, 162), "Who said what, in meetings.", font=ImageFont.truetype(FONT, 38), fill=INK)
d.text((84, 216), "Multi-speaker transcription with timestamps · research preview",
       font=ImageFont.truetype(FONT, 25), fill=MUTED)

# A small timeline in the app's style: one lane per speaker, turns with a solid left edge.
top, lane_h, left, right = 300, 62, 290, 1120
label = ImageFont.truetype(BOLD, 24)
rng = random.Random(7)
d.rounded_rectangle([80, top - 14, right + 20, top + 4 * lane_h + 6], radius=16, fill=(255, 255, 255),
                    outline=(229, 231, 235), width=2)
for k, c in enumerate(PALETTE):
    y = top + k * lane_h
    if k:
        d.line([80, y - 4, right + 20, y - 4], fill=(229, 231, 235), width=2)
    d.text((104, y + 14), f"Speaker {k + 1}", font=label, fill=c)
    x = left + rng.randint(0, 60)
    while x < right - 40:
        w = rng.randint(40, 190)
        x2 = min(x + w, right)
        d.rounded_rectangle([x, y + 8, x2, y + lane_h - 18], radius=7, fill=tint(c, .18))
        d.rectangle([x, y + 8, x + 5, y + lane_h - 18], fill=c)
        x = x2 + rng.randint(60, 260)
d.line([270, top - 14, 270, top + 4 * lane_h + 6], fill=(229, 231, 235), width=2)  # name column
d.line([left + 300, top - 14, left + 300, top + 4 * lane_h + 6], fill=ACCENT, width=4)
d.polygon([(left + 290, top - 14), (left + 310, top - 14), (left + 300, top - 2)], fill=ACCENT)

d.text((80, 586), "BUT Speech@FIT · Brno University of Technology", font=ImageFont.truetype(FONT, 22), fill=MUTED)
img.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview.png"), optimize=True)
