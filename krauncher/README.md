# Running these lessons on Krauncher

This folder lets you run the course's GPU lessons on rented hardware **without
renting excessive hardware and without doing the "will it fit?" math yourself**.
Each example wraps a lesson's own training code in a Krauncher task; Krauncher
reads the code, sizes the VRAM it needs, and runs it on the cheapest GPU on the
market that fits.

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

The example scripts stay next to their lessons; run them from here:

```bash
python ../phases/11-llm-engineering/08-fine-tuning-lora/code/lora_finetune.py
```

This one reproduces the LoRA lesson's "Use It" fine-tune (QLoRA on
Llama-3.1-8B over Alpaca) and reports which GPU ran it and what it cost.

### Prerequisite for the LoRA example

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

Nothing here is specific to the course. The same wrapper turns any GPU Python
job — your own training, fine-tune, or inference — into a Krauncher task: move
the imports inside a `@client.task` function, return JSON-serializable results,
call it from an async `main`. It's thin and additive (the lesson's `lora.py` is
left untouched next to `lora_finetune.py`), and you don't have to write it by
hand — Krauncher documents its API for coding assistants (`llms.txt` on
krauncher.com), so you can point your LLM at your script and ask it to wrap it.

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
