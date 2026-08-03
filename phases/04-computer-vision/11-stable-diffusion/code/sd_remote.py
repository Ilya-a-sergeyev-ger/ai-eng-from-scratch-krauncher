"""Run this lesson's own `main.py` on a rented GPU -- unmodified.

The lesson file next to this one is not copied, not edited and not imported.
It is read as text at submit time and shipped to the worker, where it runs as
a script.

This lesson is one of the few whose own code does real GPU work: its
`text_to_image_stub` skips the generation when `torch.cuda.is_available()` is
false, which is why it prints a summary on your laptop and actually paints a
512x512 image out there. Stable Diffusion 1.5 in fp16 needs about 4 GB, so
Krauncher lands it on a cheap card -- compare the bill with the QLoRA fine-tune
in phase 11, which needs a ~48 GB one.

Two things worth knowing:

* The lesson saves its image to `~/sd_demo.png`, and you get it back. A task
  runs with HOME set to its own working directory, so writing to `~` is
  writing where `artifacts=True` collects from -- the lesson's own path works
  untouched, which is the point.
* The lesson asks for `runwayml/stable-diffusion-v1-5`. Runway withdrew that
  repository, but the Hub still redirects it to the maintained mirror, so the
  call resolves. `sd_inference.py` names the mirror directly.

Run it from the repo's `krauncher/` folder, which holds the shared config:

    cd krauncher
    cp .env.example .env          # then put your CAS_API_KEY in it
    pip install krauncher
    python ../phases/04-computer-vision/11-stable-diffusion/code/sd_remote.py
"""

import asyncio
import tempfile
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

    # run_code takes the file's text as the body of a generated task function;
    # the lesson's `if __name__ == "__main__"` block still fires, because the
    # worker writes the whole thing to a script and runs it.
    #
    # The auto-installer reads the lesson's imports and finds `diffusers` on its
    # own. transformers and accelerate are pulled in *by* diffusers rather than
    # imported by name, so they are requested explicitly here -- a task option,
    # not an edit to the lesson.
    handle = await client.run_code(
        LESSON.read_text(),
        pip=["diffusers", "transformers", "accelerate"],
        timeout=1800,
        disk_gb=30,  # SD 1.5 weights land in the Hub cache
        artifacts=True,  # the lesson writes to ~, which is its workspace
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

    # The card and the bill are already in Krauncher's own closing line.
    print()
    out_dir = Path(tempfile.mkdtemp(prefix="sd-lesson-"))
    count = result.download(str(out_dir))
    print(f"Files:  {result.files or 'none'}"
          + (f" -> {out_dir}" if count else ""))


if __name__ == "__main__":
    asyncio.run(main())
