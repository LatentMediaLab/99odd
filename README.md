# 99odd

99odd — the `99` resembles a lowercase _g_, so the title reads as _god_. Flipped upside down, it remains legible. It also references the 99 names of Allah (Asmaul Husna), the complete list of divine attributes in Islamic tradition.

This program is a key part of an interactive art installation that explores the infinite sloppiness of both God and AI.

## How it works

1. **Double clap** → a name is chosen randomly from the 99 names of Allah
2. **Claude Haiku** (speaking as Allah in first person) selects exactly 2 Japanese kanji as a visual signature for that name and writes a short image prompt
3. **SDXL Turbo** runs two passes:
   - Pass 1 — generates a full scene from the prompt (black and white, ink wash style)
   - Pass 2 — overlays the kanji on the scene and blends them in
4. The final image crossfades onto the display and stays until the next double clap

## Requirements

- Python 3.11+
- A microphone (for clap detection)
- An Anthropic API key
- ~7 GB disk space for the SDXL Turbo model (downloaded automatically on first run)
- macOS (MPS acceleration); CPU fallback works on other platforms

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `venv/venv` file (the project reads environment variables from there):

```
ANTHROPIC_API_KEY=sk-ant-...
```

Optional display settings in `venv/venv`:

```
FULLSCREEN=1        # 0 = windowed (default), 1 = fullscreen
DISPLAY_INDEX=1     # which monitor to use (0 = primary, 1 = second, etc.)
```

## Running

```bash
python main.py
```

On first run the SDXL Turbo model (~7 GB) will be downloaded from Hugging Face. Subsequent runs load it from cache.

The terminal will print available audio input devices on startup. If clap detection uses the wrong microphone, set `CLAP_INPUT` at the top of `clap.py` to the correct device index.

## Controls

| Action      | Effect                                             |
| ----------- | -------------------------------------------------- |
| Double clap | Generate a new name / interrupt current generation |
| `ESC`       | Quit                                               |

## Project structure

```
main.py          — state machine and main loop
generate.py      — Claude API call and SDXL Turbo image generation
display.py       — tkinter fullscreen window with crossfade
clap.py          — microphone-based double clap detector
fonts/           — Japanese font collection for kanji overlay
```

## Fonts

The kanji overlay picks randomly from the fonts in the `fonts/` directory. Any `.ttf`, `.otf`, or `.ttc` file placed there will be included. The current set is a mix of brush, mincho, gothic, and decorative Japanese typefaces.

## AI Disclosure

Base code and documentation written with [Claude Code](https://claude.ai/code) (Anthropic).

This installation requires two AI systems to run:

- [Claude Haiku](https://www.anthropic.com/claude) — generates kanji selection and image prompts
- [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) — runs locally for image generation
