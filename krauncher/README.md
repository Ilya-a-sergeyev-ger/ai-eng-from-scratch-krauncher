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
