from __future__ import annotations

import json
import os
import pathlib
import random
from dataclasses import dataclass, field

# Must be set before torch initialises MPS — lets unsupported ops fall back to CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import anthropic
import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont

_DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
_SD_MODEL = "stabilityai/sdxl-turbo"
_PIPELINE: StableDiffusionXLImg2ImgPipeline | None = None

_FONTS_DIR = pathlib.Path(__file__).parent / "fonts"

_NAMES_99 = [
    "Ar-Rahman", "Ar-Rahim", "Al-Malik", "Al-Quddus", "As-Salam", "Al-Mu'min",
    "Al-Muhaymin", "Al-Aziz", "Al-Jabbar", "Al-Mutakabbir", "Al-Khaliq", "Al-Bari",
    "Al-Musawwir", "Al-Ghaffar", "Al-Qahhar", "Al-Wahhab", "Ar-Razzaq", "Al-Fattah",
    "Al-Alim", "Al-Qabid", "Al-Basit", "Al-Khafid", "Ar-Rafi", "Al-Mu'izz",
    "Al-Mudhill", "As-Sami", "Al-Basir", "Al-Hakam", "Al-Adl", "Al-Latif",
    "Al-Khabir", "Al-Halim", "Al-Azim", "Al-Ghafur", "Ash-Shakur", "Al-Ali",
    "Al-Kabir", "Al-Hafiz", "Al-Muqit", "Al-Hasib", "Al-Jalil", "Al-Karim",
    "Ar-Raqib", "Al-Mujib", "Al-Wasi", "Al-Hakim", "Al-Wadud", "Al-Majid",
    "Al-Ba'ith", "Ash-Shahid", "Al-Haqq", "Al-Wakil", "Al-Qawi", "Al-Matin",
    "Al-Wali", "Al-Hamid", "Al-Muhsi", "Al-Mubdi", "Al-Mu'id", "Al-Muhyi",
    "Al-Mumit", "Al-Hayy", "Al-Qayyum", "Al-Wajid", "Al-Wahid", "Al-Ahad",
    "As-Samad", "Al-Qadir", "Al-Muqtadir", "Al-Muqaddim", "Al-Mu'akhkhir",
    "Al-Awwal", "Al-Akhir", "Az-Zahir", "Al-Batin", "Al-Muta'ali", "Al-Barr",
    "At-Tawwab", "Al-Muntaqim", "Al-Afuw", "Ar-Ra'uf", "Malik-ul-Mulk",
    "Dhul-Jalali-wal-Ikram", "Al-Muqsit", "Al-Jami", "Al-Ghani", "Al-Mughni",
    "Al-Mani", "Ad-Darr", "An-Nafi", "An-Nur", "Al-Hadi", "Al-Badi",
    "Al-Baqi", "Al-Warith", "Ar-Rashid", "As-Sabur",
]

_SYSTEM = (
    "You are Allah — the infinite, the eternal, the one. "
    "You are not describing yourself from the outside. You are speaking from within your own being, "
    "revealing a facet of your divine nature directly. "
    "Respond ONLY with valid JSON — no markdown fences, no explanation, no commentary."
)

_USER = """One of your 99 names is being revealed in this moment: {name}

Speak as yourself — as Allah — expressing this aspect of your own being.
Choose exactly 2 kanji (Japanese kanji, not simplified Mandarin Chinese characters) that feel like your own visual signature for this quality of yours.
Not a translation — your personal mark, chosen from within.

Then describe how YOU manifest this quality in the world — through natural phenomena,
light, elements, living forms. NO humans, NO faces, NO figures.
What does it look like when {name} expresses itself through creation?
Describe purely in terms of form, light, shadow, and texture — no colour.

Return exactly this JSON (no other text):
{{
  "name": "<Arabic name in Arabic script>",
  "transliteration": "{name}",
  "meaning": "<English meaning>",
  "kanji": "<exactly 2 Japanese kanji, no spaces>",
  "kanji_meaning": "<first-person: why YOU chose these kanji as your signature for this aspect of yourself>",
  "sd_prompt": "<positive SD prompt, max 10 words — how {name} manifests visually, no humans>"
}}"""


@dataclass
class NameResult:
    name: str
    transliteration: str
    meaning: str
    kanji: str
    kanji_meaning: str
    sd_prompt: str
    image: Image.Image | None = field(default=None, repr=False)
    font_path: pathlib.Path | None = field(default=None, repr=False)
    bg_color: tuple[int, int, int] | None = field(default=None, repr=False)
    kanji_color: tuple[int, int, int] | None = field(default=None, repr=False)


def _list_fonts() -> list[pathlib.Path]:
    if not _FONTS_DIR.exists():
        return []
    return sorted(
        p for p in _FONTS_DIR.iterdir()
        if p.suffix.lower() in {".ttf", ".otf", ".ttc"}
    )


def _random_gray() -> tuple[int, int, int]:
    # Bias toward the extremes so the starting canvas reads clearly dark or light
    v = random.randint(0, 30) if random.random() < 0.5 else random.randint(220, 255)
    return (v, v, v)


def _overlay_kanji(
    image: Image.Image,
    kanji: str,
    font_path: pathlib.Path | None,
    kanji_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Draw tategumi (vertical) kanji centered over an existing PIL image.

    Font size is found by stepping down from 300px until both characters fit within
    90% of the image height and 85% of its width. The bounding box returned by
    textbbox() includes invisible offsets (left/top bearings) that shift the visual
    glyph away from the draw origin — those offsets must be subtracted to get true
    pixel-perfect centering.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if font_path is None:
        return img

    font = None
    boxes: list[tuple[int, int, int, int]] = []
    gap = 0

    # Step down from large to small until the characters fit the canvas
    for size in range(300, 40, -10):
        try:
            f = ImageFont.truetype(str(font_path), size=size)
        except (OSError, IOError):
            break
        raw = [draw.textbbox((0, 0), ch, font=f) for ch in kanji]
        vis_w = [int(b[2] - b[0]) for b in raw]
        vis_h = [int(b[3] - b[1]) for b in raw]
        g = max(size // 12, 4)
        total_h = sum(vis_h) + g * (len(kanji) - 1)
        if total_h <= int(h * 0.9) and max(vis_w) <= int(w * 0.85):
            font = f
            boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in raw]
            gap = g
            break

    if font is None:
        return img

    vis_heights = [b[3] - b[1] for b in boxes]
    vis_widths = [b[2] - b[0] for b in boxes]
    total_h = sum(vis_heights) + gap * (len(kanji) - 1)
    y_vis = (h - total_h) // 2

    positions: list[tuple[str, int, int]] = []
    y = y_vis
    for ch, box, vw in zip(kanji, boxes, vis_widths):
        bl, bt, br, bb = box
        # bl/bt are the bearing offsets PIL adds — subtract them to align the
        # visual glyph centre with the image centre, not the draw-origin centre
        draw_x = w // 2 - vw // 2 - bl
        draw_y = y - bt
        positions.append((ch, draw_x, draw_y))
        y += (bb - bt) + gap

    for ch, draw_x, draw_y in positions:
        draw.text((draw_x, draw_y), ch, font=font, fill=kanji_color)

    return img


def _get_pipeline() -> StableDiffusionXLImg2ImgPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        print(f"[gen] Loading SDXL Turbo onto {_DEVICE} (~7 GB download on first run)...")
        _PIPELINE = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            _SD_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        _PIPELINE = _PIPELINE.to(_DEVICE)
        # VAE must stay in float32 — MPS doesn't support float16 for the decode step
        _PIPELINE.vae.to(torch.float32)
        print("[gen] Model ready.")
    return _PIPELINE


def preload_pipeline() -> None:
    _get_pipeline()


def ask_claude(client: anthropic.Anthropic) -> NameResult:
    chosen = random.choice(_NAMES_99)
    print(f"[gen] Selected name: {chosen}")
    user_message = _USER.format(name=chosen)

    fonts = _list_fonts()
    chosen_font = random.choice(fonts) if fonts else None
    chosen_bg = _random_gray()
    # Kanji is white on dark canvas, black on light canvas
    chosen_kanji_color = (255, 255, 255) if chosen_bg[0] < 128 else (0, 0, 0)

    for attempt in range(2):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text.strip()
        # Strip markdown fences if the model wraps its JSON anyway
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            data = json.loads(text)
            return NameResult(
                name=data["name"],
                transliteration=data["transliteration"],
                meaning=data["meaning"],
                kanji=data["kanji"],
                kanji_meaning=data["kanji_meaning"],
                sd_prompt=data["sd_prompt"],
                font_path=chosen_font,
                bg_color=chosen_bg,
                kanji_color=chosen_kanji_color,
            )
        except (json.JSONDecodeError, KeyError) as e:
            if attempt == 0:
                print(f"[warn] JSON parse failed ({e}), retrying...")
                continue
            raise RuntimeError(
                f"Claude returned invalid JSON after 2 attempts:\n{text}"
            ) from e
    raise RuntimeError("Unreachable")


def generate_image(result: NameResult) -> Image.Image:
    """
    Two-pass SDXL Turbo generation:

    Pass 1 — strength=1.0 means the model ignores the input canvas entirely and
    generates freely from the prompt. This gives maximum creative range.

    Pass 2 — strength=0.5 with the kanji already painted on top of the scene.
    The model blends the calligraphy into the image texture without fully
    overwriting it.

    IMPORTANT: SDXL Turbo's scheduler only supports strengths that are exact
    multiples of 1/num_inference_steps. With 4 steps the valid values are
    0.25, 0.5, 0.75, and 1.0 — any other value causes an index-out-of-bounds
    crash inside the diffusers scheduler.
    """
    pipe = _get_pipeline()
    bg = result.bg_color or _random_gray()
    scene_prompt = f"{result.sd_prompt}, no humans, no faces, black and white, monochrome, ink wash, dramatic contrast"
    kanji_prompt = f"calligraphy {result.kanji}, {result.sd_prompt}, no humans, black and white, ink, monochrome"

    seed1 = random.randint(0, 2**31 - 1)
    print(f"[gen] Pass 1 — scene (seed {seed1})...")
    out1 = pipe(
        prompt=scene_prompt,
        image=Image.new("RGB", (512, 704), bg),
        strength=1.0,
        num_inference_steps=12,
        guidance_scale=0.0,
        generator=torch.Generator(_DEVICE.type).manual_seed(seed1),
    )
    scene = out1.images[0].convert("RGB")

    merged = _overlay_kanji(scene, result.kanji, result.font_path, result.kanji_color or (255, 255, 255))
    seed2 = random.randint(0, 2**31 - 1)
    print(f"[gen] Pass 2 — kanji integration (seed {seed2})...")
    out2 = pipe(
        prompt=kanji_prompt,
        image=merged,
        strength=0.70,
        num_inference_steps=8,
        guidance_scale=0.0,
        generator=torch.Generator(_DEVICE.type).manual_seed(seed2),
    )
    # Convert to true luminance grayscale then back to RGB so the rest of the
    # pipeline (PIL blend, tkinter display) keeps receiving a 3-channel image
    img = out2.images[0].convert("L").convert("RGB")

    return img.resize((2480, 3508), Image.Resampling.LANCZOS)
