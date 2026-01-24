# asteroids

My asteroids clone. 

## Description of Main Scripts

* main_arcade.py: runs graphicly. You can play it with the keyboard or pass --ais to have a heuristic AI play.
* main_headless.py: Runs the heuristic AI thorugh a number of games and computes statistics on the score.
* main_genetic.py: Runs runs a genetic algorithm on the heuristic AI.

## Usage

Run any of the main_ scritps with --help for parameters.

```bash
python3 main_arcade.py --help
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
