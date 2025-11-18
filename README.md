# CRPO
The code for CRPO

[![Python](https://img.shields.io/badge/python-3.13.1-blue)](https://www.python.org/downloads/release/python-3131/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![CI](https://img.shields.io/badge/CI-%EC%82%AC%EC%9A%A9%EC%A4%91-black)]()


---

## Overview
**Contrastive Reasoning Prompt Optimization (CRPO)**, a retrieval-augmented framework that explicitly performs contrastive reasoning over prompts of varying quality.
CRPO then applies two complementary strategies: (1) tiered contrastive reasoning, reflecting over high-, medium-, and low-quality exemplars, and (2) multi-metric contrastive reasoning, integrating metricwise best exemplars into an optimized prompt. Without updating model weights, CRPO improves prompt quality by explicitly explaining why prompts succeed or fail.

---

### Features
- **main.py** — Run this file to launch the CRPO main framework.<br>
- **generate_data_service.py** — Use this file to generate each process of each module; Tiered contrastive reasoning and Multi-Metric contrastive reasoning.<br>
- **helpsteer_train.jsonl** - We left 100 lines of origin train data of helpsteer2 for example of RAG system. You can load whole Helpsteer2 train dataset through url: https://huggingface.co/datasets/nvidia/HelpSteer2<br>
- **base_service.py** — Core utilities and default settings for model parameters and execution.<br>
- **prompt.py** — Prompt templates tailored to each component.

---

## Installation

1) Clone the repository<br>
```bash
git clone https://github.com/GitLeo1/better_by_comparison.git
```
---

2) Create & activate a virtual environment<br>
```bash
python -m venv .venv && source .venv/bin/activate
```
Windows: 
```bash
.venv\Scripts\activate
```
---
3) Install dependencies<br>
```bash
pip install -U pip
pip install -r requirements.txt
```
---
4) Add API keys (Optional)<br>
```bash
touch .env
OPENAI_API_KEY=(insert your openai API key)
REPLICATE_API_TOKEN=(insert your replicate API key)
```
---
5) Usage

SAPO - Main Framework<br>

> MAC: 
```bash
python3 main.py
```
> windows:
```bash
python mian.py
```

---