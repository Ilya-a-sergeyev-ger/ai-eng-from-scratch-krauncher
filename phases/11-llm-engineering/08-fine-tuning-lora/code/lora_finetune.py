"""Fine-tune Llama-3.1-8B with QLoRA on Alpaca -- on the cheapest GPU that fits.

This reproduces the "Use It" fine-tune from
phases/11-llm-engineering/08-fine-tuning-lora (docs/en.md, "Use It" section),
made runnable and pointed at Krauncher. Instead of renting a GPU by hand, the
job is shipped to the cheapest card on the market that fits it, and you get back
which GPU actually ran it and what it cost.

Everything you install locally is ONE package: krauncher. torch, transformers,
peft and the rest run on the GPU host (see `pip=[...]` on the decorator and the
imports inside the task), not on your laptop.

Run it from the repo's `krauncher/` folder, which holds the shared config
(.env). The example scripts stay with their lessons; you run them from
`krauncher/` so they pick up that one .env:

    cd krauncher
    cp .env.example .env          # then put your CAS_API_KEY in it
    export HF_TOKEN=hf_...        # the base model is gated; your Hub token
    pip install krauncher
    python ../phases/11-llm-engineering/08-fine-tuning-lora/code/lora_finetune.py
"""

import asyncio
import os

# ─────────────────────────── Krauncher wrapper ───────────────────────────
from krauncher import KrauncherClient, TaskError

client = KrauncherClient()  # reads CAS_API_KEY / .env from the folder you run in


# Empirical VRAM (measured): this exact config -- Llama-3.1-8B in 4-bit,
# batch=4, seq=512, no gradient checkpointing -- needs a ~48 GB card (ran on an
# RTX 6000 Ada). batch*seq = 2048 exceeds the bitsandbytes fused-4bit limit
# (1536), so the base weights are dequantized to bf16 and held for the backward
# pass (~the whole model in bf16). To fit a smaller card, lower batch so
# batch*seq <= 1536 (e.g. batch=2) or enable gradient_checkpointing.
@client.task(
    vram_gb=None,  # None => Krauncher reads the code, sizes VRAM, picks cheapest
    pip=["peft", "datasets"],  # torch/transformers/bitsandbytes are on the worker
    timeout=14400,
    # Only the (public) dataset is pre-fetched via the bridge. The gated base
    # model is downloaded inside the function with an HF token, because the data
    # bridge does not carry per-request credentials.
    data_urls=["hf://datasets/tatsu-lab/alpaca"],
    dataset_size=22,  # alpaca ~22 MB
    disk_gb=40,
    stream_stderr=True,
)
def finetune(hf_token: str = ""):
    # Why the imports are inside this function, not at the top of the file:
    #
    # This function does NOT run on your machine. Krauncher serializes it and
    # ships it to a rented GPU, where it runs in a fresh Python process that
    # shares nothing with this file. Your file-level imports, variables and
    # globals do not travel with it -- "here" (your laptop) and "there" (the
    # GPU) are two separate worlds, connected only by the arguments going in
    # and the value coming back. So the remote process has to import
    # torch/transformers/peft for itself; whatever it imports that isn't already
    # on the worker, Krauncher installs automatically.
    print("Task started. Importing torch / transformers / peft...", flush=True)
    # One handler for the whole task. With E2E on, the worker's stderr and
    # traceback do not reach the client; stdout is relayed, so on any failure
    # print the full traceback to stdout before re-raising -- otherwise the
    # client just sees "task failed" with no cause.
    try:
        import torch
        # Imported for its side effect: transformers needs bitsandbytes for
        # 4-bit (QLoRA) but imports it only transitively, which the auto-installer
        # can't see. Importing it here makes Krauncher install it on the worker.
        import bitsandbytes  # noqa: F401
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            DataCollatorForLanguageModeling, Trainer, TrainerCallback,
            TrainingArguments,
        )
        from peft import LoraConfig, TaskType, get_peft_model
        from datasets import load_dataset

        class _ProgressBar(TrainerCallback):
            """Per-epoch in-place progress bar: \\r overwrites on every step,
            \\n only at epoch end -- one line per epoch, no scroll.

            On logging steps we append a \\n to flush the relay pipeline
            (which buffers output that lacks \\n), then \\033[A to move the
            cursor back up so the bar line is reused rather than scrolled."""

            BAR = 25

            def on_train_begin(self, args, state, control, **kwargs):
                self.total_epochs = int(args.num_train_epochs)
                self.steps_per_epoch = max(
                    1, state.max_steps // self.total_epochs)

            def _last_loss(self, state):
                return next((h["loss"] for h in reversed(state.log_history)
                             if "loss" in h), None)

            def on_step_end(self, args, state, control, **kwargs):
                total = self.total_epochs
                cur = min(total,
                          (state.global_step - 1) // self.steps_per_epoch + 1)
                done = (state.global_step - 1) % self.steps_per_epoch + 1
                filled = int(self.BAR * done / self.steps_per_epoch)
                bar = "█" * filled + "░" * (self.BAR - filled)
                msg = (f"\rEpoch {cur}/{total} [{bar}] "
                       f"{done:3d}/{self.steps_per_epoch}")
                loss = self._last_loss(state)
                if loss is not None:
                    msg += f"  loss={loss:.4f}"
                # Every logging_steps, append \\n to flush the relay pipeline,
                # then \\033[A to return the cursor to the bar line so the
                # terminal doesn't scroll.
                if state.global_step % args.logging_steps == 0:
                    msg += "\n\033[A"
                print(msg, end="", flush=True)

            def on_epoch_end(self, args, state, control, **kwargs):
                total = self.total_epochs
                ep = int(round(state.epoch))
                loss = self._last_loss(state)
                bar = "█" * self.BAR
                msg = (f"\rEpoch {ep}/{total} [{bar}] "
                       f"{self.steps_per_epoch}/{self.steps_per_epoch}")
                if loss is not None:
                    msg += f"  loss={loss:.4f}"
                print(msg, flush=True)  # default end="\\n"

        # The gated base model is downloaded from the Hub at runtime using the
        # token passed in as an argument (read from HF_TOKEN on the client). The
        # public dataset was pre-fetched into /data by the bridge.
        MODEL_ID = "meta-llama/Llama-3.1-8B"
        dataset_path = "/data/tatsu-lab__alpaca"

        # ─────────── lesson "Use It" code: QLoRA (4-bit base + fp16 LoRA) ───────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading base model in 4-bit...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto",
            token=hf_token or None,
        )
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
            lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        ))
        model.print_trainable_parameters()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token or None)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading + tokenizing dataset...", flush=True)
        dataset = load_dataset(dataset_path)["train"].select(range(5000))
        dataset = dataset.map(
            lambda ex: tokenizer(ex["text"], truncation=True,
                                 max_length=512, padding="max_length"),
            remove_columns=dataset.column_names,
        )

        print("Starting LoRA training...", flush=True)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="/tmp/lora-llama",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                optim="paged_adamw_8bit",
                disable_tqdm=True,  # our _ProgressBar renders instead
            ),
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[_ProgressBar()],
        )
        # Drop the Trainer's default PrinterCallback (installed because
        # disable_tqdm=True) so it doesn't dump raw log dicts over the bar.
        from transformers.trainer_callback import PrinterCallback
        trainer.remove_callback(PrinterCallback)
        trainer.train()
        # ────────── end lesson "Use It" code ──────────

        # Return only JSON-serializable data -- the model and tensors cannot
        # cross back from the GPU, only plain values do.
        return {"train_loss": round(trainer.state.log_history[-1]["train_loss"], 4)}
    except Exception:
        import traceback
        print("TASK FAILED:\n" + traceback.format_exc(), flush=True)
        raise


def _print_progress(msg):
    """Relay the worker's stdout/stderr verbatim (GPU metrics skipped)."""
    if msg.get("type") not in ("stdout", "stderr"):
        return
    text = (msg.get("data") or {}).get("text") or ""
    print(text, end="", flush=True)


async def main():
    if not client.api_key:
        print("Set CAS_API_KEY in .env (copy .env.example). "
              "New accounts get free credits.")
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Warning: HF_TOKEN not set; the gated base model download will "
              "fail on the worker. Run `export HF_TOKEN=hf_...` first.")

    # Calling the wrapped function submits the job; Krauncher picks the cheapest
    # GPU that fits (vram_gb=None) and starts it there. The token rides along as
    # an argument (E2E-encrypted to the worker), never touching the broker.
    handle = await finetune(hf_token=hf_token)
    print(f"Submitted {handle.task_id} -- waiting for the cheapest GPU...",
          flush=True)

    try:
        result = await handle.wait(on_log=_print_progress, timeout=14700)
    except TaskError as e:
        print("\n--- task failed on the GPU ---")
        print(getattr(e, "remote_traceback", None) or e)
        return

    # The card and the bill are already in Krauncher's own closing line.
    print(f"\nLoss: {result.output['train_loss']}")


if __name__ == "__main__":
    asyncio.run(main())
# ───────────────────────── end Krauncher wrapper ─────────────────────────
