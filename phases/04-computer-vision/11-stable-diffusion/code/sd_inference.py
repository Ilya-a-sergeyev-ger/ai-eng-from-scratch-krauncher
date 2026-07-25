"""Generate a batch of images with Stable Diffusion 1.5 -- on the cheapest GPU that fits.

This is the lesson's "Build It" pipeline (step 1, text-to-image, plus the
step 2 scheduler swap) run as a real job rather than a single demonstration
call: a batch of prompts, the images handed back to your machine, and the cost
per image at the end. SD 1.5 in fp16 needs roughly 4 GB, so this lands on a
cheap card -- the same wrapper sends the phase 11 QLoRA fine-tune to a ~48 GB
one, and the bills are worth comparing.

What this adds to the lesson's snippets, which are illustrative rather than
runnable:

* the model id names the maintained mirror. The lesson asks for
  `runwayml/stable-diffusion-v1-5`; Runway withdrew that repository and the Hub
  currently redirects it, which is a redirect this script does not rely on.
* the images come back. The lesson's `image.save("dog.png")` writes into a
  sandbox that is destroyed with the task; `artifacts=True` returns whatever
  the task wrote beside itself, as raw bytes.
* per-image timing and cost, which is the number you actually decide on.
* the safety checker is left out, which skips an extra model download.

Everything you install locally is ONE package: krauncher. torch and diffusers
run on the GPU host.

    cd krauncher
    cp .env.example .env          # then put your CAS_API_KEY in it
    pip install krauncher
    python ../phases/04-computer-vision/11-stable-diffusion/code/sd_inference.py
"""

import asyncio
import tempfile
from pathlib import Path

from krauncher import KrauncherClient, TaskError

client = KrauncherClient()  # reads CAS_API_KEY / .env from the folder you run in

PROMPTS = [
    "a dog riding a skateboard in tokyo, studio ghibli style",
    "a village square at dawn, studio ghibli style",
    "an old lighthouse in a storm, oil painting",
    "a tram on a rainy street at night, cinematic lighting",
]


@client.task(
    vram_gb=None,  # None => Krauncher reads the code, sizes VRAM, picks cheapest
    pip=["diffusers", "transformers", "accelerate"],  # torch is on the worker
    timeout=2400,
    disk_gb=30,  # the weights land in the Hub cache
    artifacts=True,  # whatever the task writes beside itself comes back
    stream_stderr=True,
)
def generate(prompts=None, steps=25, guidance=7.5, seed=42):
    # Why the imports are inside this function: it does not run on your machine.
    # Krauncher ships it to a rented GPU, where it runs in a fresh process that
    # shares nothing with this file -- file-level imports do not travel with it.
    import time

    try:
        import torch
        from diffusers import (
            DPMSolverMultistepScheduler, StableDiffusionPipeline,
        )

        MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        t0 = time.time()
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
        ).to("cuda")
        # Scheduler state is decoupled from the U-Net weights (lesson step 2):
        # DPM-Solver++ at 25 steps matches DDIM at 50.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        load_sec = time.time() - t0
        print(f"pipeline ready in {load_sec:.1f}s", flush=True)

        seconds = []
        for i, prompt in enumerate(prompts, 1):
            t = time.time()
            image = pipe(
                prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=torch.Generator("cuda").manual_seed(seed + i),
            ).images[0]
            seconds.append(round(time.time() - t, 2))

            # Written beside the task, exactly as the lesson writes it locally.
            image.save(f"{i:02d}.png")
            print(f"[{i}/{len(prompts)}] {seconds[-1]:>5.2f}s  {prompt[:48]}",
                  flush=True)

        return {"seconds": seconds, "load_sec": round(load_sec, 1)}
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

    handle = await generate(prompts=PROMPTS)
    print(f"Submitted {handle.task_id} -- waiting for the cheapest GPU...",
          flush=True)

    try:
        result = await handle.wait(on_log=_print_progress, timeout=2700)
    except TaskError as e:
        print("\n--- task failed on the GPU ---")
        print(getattr(e, "remote_traceback", None) or e)
        return

    out_dir = Path(tempfile.mkdtemp(prefix="sd-"))
    result.download(str(out_dir))
    for name, prompt in zip(result.files, PROMPTS):
        print(f"  {out_dir / name}  {prompt}")

    seconds = result.output["seconds"]
    n = len(seconds)
    print("\n--- done ---")
    print(f"Ran on:    {result.actual_gpu}")
    print(f"Pipeline:  {result.output['load_sec']}s to load, "
          f"{sum(seconds) / n:.2f}s per image over {n}")
    print(f"Cost:      {result.total_charged_ku} KU "
          f"({result.total_charged_local} {result.billing_currency})"
          f" -- {result.total_charged_local / n:.4f} "
          f"{result.billing_currency} per image")


if __name__ == "__main__":
    asyncio.run(main())
