# Empathic MARL — Minimal Baseline

This repo trains two Q-learning agents in a Prisoner's Dilemma and compares
selfish (alpha=1.0) vs. empathic (alpha<1.0) reward shaping.

## How it works
- Environment: stateless matrix game with actions {C, D}.
- Agents: Q-learning with epsilon-greedy exploration.
- Empathy: shaped reward r' = alpha*r_self + (1-alpha)*r_other.

## Usage
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install matplotlib
python train.py