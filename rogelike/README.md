# Roguelike

A roguelike game built with Python arcade. 

## Assets

Sprites use the **Urizen 1Bit Tileset** by vurmux (CC0 — no attribution required).

1. Go to https://vurmux.itch.io/urizen-onebit-tileset
2. Click **Download Now** — it's pay what you want, so you can enter $0
3. Extract the downloaded zip and place the contents into an `assets/` folder in this directory

Additional sprites from the **Free Pixel Art Asset Pack (Top-Down Tileset RPG 16x16)** by Anokolisa.

1. Go to https://anokolisa.itch.io/free-pixel-art-asset-pack-topdown-tileset-rpg-16x16-sprites
2. Click **Download Now** — it's pay what you want, so you can enter $0
3. Extract the downloaded zip and place the contents into the `assets/` folder

The `assets/` folder is gitignored, so each developer needs to download it separately.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management. Dependencies are declared in `pyproject.toml` and pinned
in `uv.lock`.

```bash
uv sync
```

That's the whole setup. `uv sync` installs the pinned Python interpreter
(**3.13**, set in `.python-version`) and all dependencies from the lockfile,
reproducing an identical environment on any machine.

To add a dependency use `uv add <pkg>` (which updates `pyproject.toml` and
`uv.lock`) — **not** `uv pip install`, which installs imperatively and gets
wiped the next time `uv sync`/`uv run` reconciles the environment against the
lockfile.

### Why Python 3.13 (not 3.14)

`arcade==3.3.3` depends on `pymunk==6.9.0`, which has no prebuilt wheel for
Python 3.14. On 3.14, uv falls back to compiling pymunk from source, which fails
without Python development headers. Pinning to 3.13 (via `.python-version`)
means every dependency installs as a prebuilt wheel with no compilation.

## Run

Run anything through the project environment with `uv run`, e.g.

```bash
uv run python maze_viewer.py
```

## Reading

Basic BSP Dungeon generation https://www.roguebasin.com/index.php/Basic_BSP_Dungeon_generation
BSP in rust https://bfnightly.bracketproductions.com/chapter_25.html

## Tools

### `explore_tile.py`

Browses the tileset PNG and lets you pick out individual tiles.

- **Left/Right** — page through column groups of the sheet (skips the separator column)
- **Up/Down** — page through rows
- **Click** a tile — copies its source-pixel coordinates as `(x, y, w, h)` to the clipboard, ready to paste into `sprite_viewer.py`

The status bar shows the current view offset, the hovered tile's `(col, row)`, and the last-copied coords.

Clipboard support requires `xclip` (or `wl-copy`/`xsel`):

```bash
sudo apt install xclip
```

### `sprite_viewer.py`

Visually verifies sprites you've extracted. Edit the `SPRITES` dict at the top of the file with `name: (x, y, w, h)` entries (paste from `explore_tile.py`). Run it and each sprite is drawn scaled with its name and coords.
