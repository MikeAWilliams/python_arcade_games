# Roguelike

A roguelike game built with Python arcade. Room generation will eventually be handled by a C++ library called via a C interface.

## Assets

Sprites use the **Urizen 1Bit Tileset** by vurmux (CC0 — no attribution required).

1. Go to https://vurmux.itch.io/urizen-onebit-tileset
2. Click **Download Now** — it's pay what you want, so you can enter $0
3. Extract the downloaded zip and place the contents into an `assets/` folder in this directory

The `assets/` folder is gitignored, so each developer needs to download it separately.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install arcade
```

## Run

```bash
python game.py
```

## Deactivate venv

```bash
deactivate
```
