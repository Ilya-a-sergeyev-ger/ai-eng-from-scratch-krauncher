"""Run this lesson's own `main.py` on a rented GPU -- unmodified.

The lesson file sitting next to this one is not copied, not edited and not
imported. It is read as text at submit time and shipped to the worker, where
it runs as a script: its `if __name__ == "__main__"` demo (InfoNCE on aligned
vs random pairs, the MAE mask, the DINO centring buffer) executes there, and
its stdout is relayed back here line by line.

Unlike the Stable Diffusion lesson, this one's code has no CUDA branch -- it is
plain torch on random tensors and runs identically on a laptop. Sending it out
demonstrates the plumbing, not a speed-up: the same unmodified file, executed
somewhere else, for about a cent. The real card is needed by the other half of
the pair, `dinov2_linear_probe.py`, which puts 13,000 images through a frozen
ViT.

Everything you install locally is ONE package: krauncher. torch runs on the
GPU host, not on your laptop.

Run it from the repo's `krauncher/` folder, which holds the shared config
(.env). The example scripts stay with their lessons; you run them from
`krauncher/` so they pick up that one .env:

    cd krauncher
    cp .env.example .env          # then put your CAS_API_KEY in it
    pip install krauncher
    python ../phases/04-computer-vision/17-self-supervised-vision/code/main_remote.py
"""

import asyncio
from pathlib import Path

from krauncher import KrauncherClient, TaskError

client = KrauncherClient()  # reads CAS_API_KEY / .env from the folder you run in

# Resolved from this file, not from the working directory -- you run from
# `krauncher/`, the lesson file lives here.
LESSON = Path(__file__).parent / "main.py"


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
        timeout=900,
        stream_stderr=True,
    )
    print(f"Submitted {handle.task_id} -- waiting for the cheapest GPU...",
          flush=True)

    # Nothing to print afterwards: the lesson's own stdout is relayed above,
    # and the card and the bill are in Krauncher's own closing line.
    try:
        await handle.wait(on_log=_print_progress, timeout=1200)
    except TaskError as e:
        print("\n--- task failed on the GPU ---")
        print(getattr(e, "remote_traceback", None) or e)
        return


if __name__ == "__main__":
    asyncio.run(main())
