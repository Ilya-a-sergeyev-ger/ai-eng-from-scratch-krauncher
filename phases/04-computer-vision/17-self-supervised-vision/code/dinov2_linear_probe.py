"""Linear-probe a frozen DINOv2 on STL-10 -- on the cheapest GPU that fits.

This is the lesson's "Use It" section run as a real job: `facebook/dinov2-base`
is frozen, every STL-10 image becomes a 768-dim CLS embedding, and a single
linear layer is trained on top. That is the standard way self-supervised
features are judged -- if the features are good, a linear classifier is enough,
and STL-10 (5,000 labelled train images, 8,000 test, 96x96) is the benchmark
the SSL literature uses for exactly this.

The number to read is the gap between two probes trained the same way: one on
DINOv2 embeddings, one on raw downscaled pixels. Both use the same linear
layer, the same optimizer and the same labels, so the difference is what the
self-supervised pretraining bought.

What this adds to the lesson's snippet, which is illustrative rather than
runnable:

* a dataset. The snippet embeds one `pil_image`; a probe needs all 13,000, in
  batches, in fp16, with the model on the GPU.
* the probe itself. The snippet stops at the embedding and says "fine-tuning
  rarely needs more than a linear head" -- this trains that head and reports
  test accuracy.
* the pixel baseline, so the accuracy is judgeable rather than just high.
* features are standardized (train mean/std) before the probe, which is what
  makes a linear head converge in seconds rather than sulking.

The model and dataset arrive through the data bridge (`data_urls`), so the
worker does not hit the Hub at run time. Both are open -- no token, no access
request.

This is the light end of the ladder: a ViT-B/14 forward pass needs a couple of
GB, against the ~48 GB of the QLoRA fine-tune in phase 11. Same wrapper, same
`vram_gb=None`, two orders of magnitude apart -- and the bills are worth
comparing.

Measured: an RTX 2000 Ada (16 GB, runpod), 97 s of execution — 60.7 s to embed
all 13,000 images (214 img/s), 2.2 s and 2.0 s for the two probes — peaking at
1.16 GB of VRAM. The probe on DINOv2 features reaches 0.9951 test accuracy, the
same probe on raw pixels 0.3276 (its best is 0.3551 at epoch 20, after which it
overfits 5,000 images; both are trained identically, which is the comparison).
Total €0.0138: €0.0065 of GPU time and €0.0073 of dispatch fee.

Everything you install locally is ONE package: krauncher. torch, transformers
and datasets run on the GPU host.

    cd krauncher
    cp .env.example .env          # then put your CAS_API_KEY in it
    pip install krauncher
    python ../phases/04-computer-vision/17-self-supervised-vision/code/dinov2_linear_probe.py
"""

import asyncio

from krauncher import KrauncherClient, TaskError

client = KrauncherClient()  # reads CAS_API_KEY / .env from the folder you run in

HF_MODEL = "hf://models/facebook/dinov2-base"
HF_DATASET = "hf://datasets/tanganke/stl10"


# Empirical VRAM (measured): 1.16 GB peak — batch 64 of 224x224 through
# ViT-B/14 in fp16, plus the 13,000 x 768 features and the pixel baseline held
# on the card for the probes. Anything with a couple of GB fits; the analyzer
# landed it on a 16 GB RTX 2000 Ada, the cheapest card available at the time.
@client.task(
    vram_gb=None,  # None => Krauncher reads the code, sizes VRAM, picks cheapest
    pip=["datasets"],  # torch/transformers are on the worker
    timeout=3600,
    # Pre-fetched into /data before the task starts: ~350 MB of weights and
    # ~230 MB of parquet. Both repos are public, so the bridge handles them.
    data_urls=[HF_MODEL, HF_DATASET],
    dataset_size=930,
    disk_gb=20,
    stream_stderr=True,
)
def linear_probe(batch_size=64, epochs=100, lr=1e-3, pixel_side=32):
    # Why the imports are inside this function: it does not run on your machine.
    # Krauncher ships it to a rented GPU, where it runs in a fresh process that
    # shares nothing with this file -- file-level imports do not travel with it.
    print("Task started. Importing torch / transformers / datasets...", flush=True)
    # One handler for the whole task. With E2E on, the worker's stderr and
    # traceback do not reach the client; stdout is relayed, so on any failure
    # print the full traceback to stdout before re-raising.
    try:
        import time

        import numpy as np
        import torch
        import torch.nn.functional as F
        from datasets import load_dataset
        from transformers import AutoImageProcessor, AutoModel

        MODEL_PATH = "/data/facebook__dinov2-base"
        DATA_PATH = "/data/tanganke__stl10"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)

        try:
            ds = load_dataset(DATA_PATH)
        except Exception:
            # Snapshot without a loadable config: read the parquet files direct.
            ds = load_dataset("parquet", data_files={
                "train": f"{DATA_PATH}/data/train-*.parquet",
                "test": f"{DATA_PATH}/data/test-*.parquet",
            })
        n_train, n_test = len(ds["train"]), len(ds["test"])
        print(f"STL-10: {n_train} train / {n_test} test", flush=True)

        # ─────────── lesson "Use It" code: frozen DINOv2, CLS embeddings ───────────
        t0 = time.time()
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH)
        # Cast after loading rather than via a from_pretrained dtype kwarg,
        # whose name moved between transformers versions.
        model = model.to(device=device, dtype=torch.float16).eval()
        print(f"DINOv2 loaded in {time.time() - t0:.1f}s", flush=True)

        @torch.no_grad()
        def embed(split):
            """CLS token per image -- the lesson's `last_hidden_state[:, 0]`."""
            data, out, t = ds[split], [], time.time()
            for start in range(0, len(data), batch_size):
                images = data[start:start + batch_size]["image"]
                inputs = processor(images=[im.convert("RGB") for im in images],
                                   return_tensors="pt")
                inputs = {k: v.to(device=device, dtype=torch.float16)
                          for k, v in inputs.items()}
                cls = model(**inputs).last_hidden_state[:, 0]
                out.append(cls.float().cpu())
                done = min(start + batch_size, len(data))
                print(f"\r  {split}: {done}/{len(data)} embedded", end="",
                      flush=True)
            print(f"  ({time.time() - t:.1f}s)", flush=True)
            return torch.cat(out)
        # ────────── end lesson "Use It" code ──────────

        t_embed = time.time()
        x_train, x_test = embed("train"), embed("test")
        embed_sec = time.time() - t_embed
        y_train = torch.tensor(ds["train"]["label"])
        y_test = torch.tensor(ds["test"]["label"])
        dim = x_train.shape[1]
        print(f"Embeddings: {tuple(x_train.shape)} in {embed_sec:.1f}s "
              f"({(n_train + n_test) / embed_sec:.0f} img/s)", flush=True)

        def probe(xtr, xte, name):
            """Train one linear layer on frozen features; return test accuracy."""
            # Standardize on train statistics -- the probe is a single linear
            # layer, so the scale of its input is the whole conditioning story.
            mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
            xtr, xte = ((xtr - mu) / sd).to(device), ((xte - mu) / sd).to(device)
            ytr, yte = y_train.to(device), y_test.to(device)

            head = torch.nn.Linear(xtr.shape[1], 10).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
            t = time.time()
            for epoch in range(1, epochs + 1):
                perm = torch.randperm(len(xtr), device=device)
                for start in range(0, len(xtr), 256):
                    idx = perm[start:start + 256]
                    loss = F.cross_entropy(head(xtr[idx]), ytr[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if epoch % 20 == 0 or epoch == 1:
                    with torch.no_grad():
                        acc = (head(xte).argmax(1) == yte).float().mean().item()
                    print(f"  [{name}] epoch {epoch:3d}/{epochs}  "
                          f"loss={loss.item():.4f}  test_acc={acc:.4f}",
                          flush=True)
            with torch.no_grad():
                acc = (head(xte).argmax(1) == yte).float().mean().item()
            print(f"  [{name}] {acc:.4f} after {time.time() - t:.1f}s", flush=True)
            return acc

        print("Training the probe on DINOv2 features...", flush=True)
        probe_acc = probe(x_train, x_test, "dinov2")

        # Baseline: the same head on raw pixels, downscaled so the two probes
        # see comparable input dimensions (32*32*3 = 3072 vs 768). Nothing here
        # is pretrained -- it is what 5,000 labels buy without an encoder.
        print("Training the same probe on raw pixels...", flush=True)

        def pixels(split):
            data, out = ds[split], []
            for start in range(0, len(data), 256):
                images = data[start:start + 256]["image"]
                out.append(torch.stack([
                    torch.from_numpy(np.asarray(
                        im.convert("RGB").resize((pixel_side, pixel_side))
                    ).copy()).float().flatten() / 255.0
                    for im in images
                ]))
            return torch.cat(out)

        pixel_acc = probe(pixels("train"), pixels("test"), "pixels")

        # Return only JSON-serializable data -- tensors do not cross back.
        return {
            "probe_acc": round(probe_acc, 4),
            "pixel_acc": round(pixel_acc, 4),
            "embed_sec": round(embed_sec, 1),
            "n_train": n_train,
            "n_test": n_test,
            "dim": dim,
        }
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

    handle = await linear_probe()
    print(f"Submitted {handle.task_id} -- waiting for the cheapest GPU...",
          flush=True)

    try:
        result = await handle.wait(on_log=_print_progress, timeout=3900)
    except TaskError as e:
        print("\n--- task failed on the GPU ---")
        print(getattr(e, "remote_traceback", None) or e)
        return

    out = result.output
    # The card and the bill are already in Krauncher's own closing line.
    print(f"\nEmbedded:  {out['n_train'] + out['n_test']} images -> "
          f"{out['dim']}-dim in {out['embed_sec']}s")
    print(f"Linear probe on DINOv2:  {out['probe_acc']:.4f}")
    print(f"Linear probe on pixels:  {out['pixel_acc']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
