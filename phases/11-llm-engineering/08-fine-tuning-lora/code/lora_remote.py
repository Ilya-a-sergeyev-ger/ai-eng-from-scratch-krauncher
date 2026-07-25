"""Run this lesson's own `lora.py` on a rented GPU -- unmodified.

The lesson file sitting next to this one is not copied, not edited and not
imported. It is read as text at submit time and shipped to the worker, where
it runs as a script: its `if __name__ == "__main__"` demo (LoRA injection,
rank comparison, quantization, two swappable adapters) executes there, and its
stdout is relayed back here line by line.

Everything you install locally is ONE package: krauncher. torch runs on the
GPU host, not on your laptop.

Run it from the repo's `krauncher/` folder, which holds the shared config
(.env). The example scripts stay with their lessons; you run them from
`krauncher/` so they pick up that one .env:

    cd krauncher
    cp .env.example .env          # then put your CAS_API_KEY in it
    pip install krauncher
    python ../phases/11-llm-engineering/08-fine-tuning-lora/code/lora_remote.py
"""

import asyncio
from pathlib import Path

from krauncher import KrauncherClient, TaskError

client = KrauncherClient()  # reads CAS_API_KEY / .env from the folder you run in

# Resolved from this file, not from the working directory -- you run from
# `krauncher/`, the lesson file lives here.
LESSON = Path(__file__).parent / "lora.py"


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

    # run_code takes the file's text as the body of a generated task function.
    # The lesson's `__main__` block still fires: the worker writes the whole
    # thing to a script and runs it, so `__name__` is "__main__" over there.
    # No vram_gb => Krauncher reads the code, sizes the VRAM it needs and picks
    # the cheapest card that fits. torch is already on the worker image.
    handle = await client.run_code(
        LESSON.read_text(),
        # The lesson's own closing metric. lenient => if the name is not set
        # (lesson edited upstream), the task still succeeds without it.
        outputs=["adapter_diff"],
        lenient_outputs=True,
        timeout=1800,
        stream_stderr=True,
    )
    print(f"Submitted {handle.task_id} -- waiting for the cheapest GPU...",
          flush=True)

    try:
        result = await handle.wait(on_log=_print_progress, timeout=2100)
    except TaskError as e:
        print("\n--- task failed on the GPU ---")
        print(getattr(e, "remote_traceback", None) or e)
        return

    print("\n--- done ---")
    print(f"Ran on: {result.actual_gpu}")
    print(f"Output: {result.output}")
    print(f"Cost:   {result.total_charged_ku} KU "
          f"({result.total_charged_local} {result.billing_currency})")


if __name__ == "__main__":
    asyncio.run(main())
