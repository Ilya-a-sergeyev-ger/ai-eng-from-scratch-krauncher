# Running these lessons on Krauncher

This folder lets you run the course's GPU work on rented hardware **without
renting excessive hardware and without doing the "will it fit?" math yourself**.
Krauncher reads the code, sizes the VRAM it needs, and runs it on the cheapest
GPU on the market that fits.

Examples come in pairs — *learn it, then run it for real*:

- **`*_remote.py`** ships the lesson's own file to a GPU **untouched**. Not a
  copy, not a rewrite: the file is read at submit time and executed as-is, so
  what runs remotely is exactly the code you just read in the lesson.
- **the task-named script beside it** (e.g. `lora_finetune.py`) implements the
  lesson's *"Use It"* section at full scale — a real model on real data, which
  is where a real card is actually needed. This one is written by hand: the
  documentation snippets are illustrative and incomplete, so each script says
  in its docstring what it adds beyond them.

You install exactly one package locally (`krauncher`). torch, transformers,
peft and everything else run on the GPU host, not on your laptop.

## Setup

```bash
cd krauncher
cp .env.example .env          # put your CAS_API_KEY in it
pip install krauncher
```

Get a key at https://krauncher.com — new accounts get free credits. Config
lives here (`.env`) and is shared by every example; you run the examples from
this folder so they pick it up.

## Run an example

The example scripts stay next to their lessons; run them from here.

**Learn it** — the LoRA lesson's own `lora.py`, unmodified, on a rented GPU:

```bash
python ../phases/11-llm-engineering/08-fine-tuning-lora/code/lora_remote.py
```

It prints the lesson's eight steps as they execute remotely, then the card it
ran on and the bill. The lesson builds LoRA from scratch on a small synthetic
model, so this costs about a cent and needs no GPU-class hardware — the point
is that your code left untouched still runs somewhere else.

**Run it for real** — the same lesson's *"Use It"* fine-tune, QLoRA on
Llama-3.1-8B over Alpaca:

```bash
python ../phases/11-llm-engineering/08-fine-tuning-lora/code/lora_finetune.py
```

This one needs ~48 GB of VRAM, and Krauncher works out that number from the
code rather than making you guess. It reports which GPU ran it and what it cost.

## Every pair

Each lesson below has both halves. Paths are relative to `phases/`; run the
scripts from this folder, so `python ../phases/<lesson>/code/<script>`.

| Lesson | Learn it | Run it for real | What the full run does |
|---|---|---|---|
| `11-llm-engineering/08-fine-tuning-lora` | `lora_remote.py` | `lora_finetune.py` | QLoRA fine-tune of Llama-3.1-8B on 5,000 Alpaca examples, 3 epochs, 4-bit base with fp16 adapters; returns the training loss, the card and the bill. ~48 GB — the heavy end. Needs `HF_TOKEN`, see below. |
| `04-computer-vision/11-stable-diffusion` | `sd_remote.py` | `sd_inference.py` | Stable Diffusion 1.5 in fp16 over four prompts, DPM-Solver++ at 25 steps; the 512x512 PNGs come back to your disk, with per-image seconds and cost. ~4 GB, measured at 1.81 s per image on an RTX 4060 Ti. |
| `04-computer-vision/17-self-supervised-vision` | `main_remote.py` | `dinov2_linear_probe.py` | Freezes `facebook/dinov2-base`, embeds all 13,000 STL-10 images as 768-dim CLS vectors, then trains a linear probe on top and the same probe on raw pixels: 0.9951 against 0.3276. The light end — 1.16 GB, 97 s and €0.0138 on an RTX 2000 Ada. |

The *learn it* half is the same shape everywhere: it reads the lesson's own
file and ships it, untouched, to be executed remotely. Most of these lessons
demonstrate their idea on tensors a laptop handles, so that half costs about a
cent; Stable Diffusion is the exception — its own code takes the CUDA branch
your laptop skips and paints a real image out there.

### Prerequisite for the LoRA fine-tune

`meta-llama/Llama-3.1-8B` is a **gated** model on the Hugging Face Hub. To run
this specific example you need:

1. Your own HF account with access granted to the model (open its page, click
   *Request access*, accept Meta's license, wait for approval).
2. Your token exported before running:

   ```bash
   export HF_TOKEN=hf_...
   ```

The token is read from the environment (never stored in `.env`) and passed to
the worker E2E-encrypted. Examples built on open models need none of this.

## Beyond the examples — bring your own code

Nothing here is specific to the course. There are two ways to send your own GPU
Python job — training, fine-tune, or inference — and the examples show both.

Wrap a function: move the imports inside a `@client.task` function, return
JSON-serializable results, call it from an async `main`. This is what
`lora_finetune.py` does, and it is what you want when the job is yours to
shape.

Or send a script you do not want to touch at all: `client.run_code()` takes the
file's text and runs it remotely as-is, which is how `lora_remote.py` executes
the lesson's `lora.py` without editing a line of it.

You don't have to write either by hand — Krauncher documents its API for coding
assistants (`llms.txt` on krauncher.com), so you can point your LLM at your
script and ask it to wrap it.

Why run it this way:

- **You don't pick the GPU or do the "will it fit?" math.** Krauncher reads the
  code, sizes the VRAM, and runs it on the cheapest card on the market that fits
  — not a flagship you overpay for, not one that runs out of memory.
- **You see the price before you spend GPU-seconds.** A pre-run estimate gives
  cost and time per GPU, so you decide before you rent — and can tune your
  config (batch size, gradient checkpointing) against a real number instead of
  guessing, then re-check.
- **One package locally, nothing to manage.** torch and the rest run on the
  host; your code and data travel E2E-encrypted; you get back the result, the
  GPU it ran on, and what it cost.
