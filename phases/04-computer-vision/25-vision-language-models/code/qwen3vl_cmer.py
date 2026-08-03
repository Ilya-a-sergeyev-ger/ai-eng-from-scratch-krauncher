"""Answer questions about photos with Qwen3-VL-8B, then measure its CMER.

This is the lesson's "Use It" section run as a real job, closed back onto the
lesson's own metric: the 8B VLM answers questions about COCO photos, and the
Cross-Modal Error Rate from `main.py` -- high text confidence paired with low
image-text similarity -- is computed over those answers instead of over random
vectors.

Two prompts go to every image, and the contrast between them is the result:

* a grounded one ("describe this photo"), which the image can answer;
* a leading one ("what is the exact text on the sign?"), which most COCO
  photos cannot answer, and which invites the model to invent one.

CMER should be low on the first and higher on the second. A single CMER number
means nothing on its own -- it is a threshold on a similarity scale nobody has
calibrated -- so the run also reports what the *reference* captions score on
that same scale. That is the anchor: if a human-written caption of the same
photo scores 0.31 and an answer scores 0.11, the threshold at 0.25 is doing
something real.

What this adds to the lesson's snippet, which is illustrative rather than
runnable:

* a dataset, batching and generation config. The snippet answers one question
  about one `plot.png`.
* the confidence number. `generate` does not return one; it comes from the
  transition scores, averaged over the generated tokens.
* the second model. Similarity needs a shared image-text space, so CLIP embeds
  the photo and the answer after the VLM is unloaded from the card.
* `AutoModelForImageTextToText` with a fallback to the snippet's
  `AutoModelForVision2Seq` -- the Auto class for this architecture was renamed
  between transformers versions.

Everything is open-weights: no HF token, no access request. The model, CLIP and
one 492 MB shard of COCO arrive through the data bridge before the task starts.

Measured on an RTX A5000 (24 GB, runpod): 81 s of execution, 29.8 s of it
generating the 48 answers (1.61 answers/s), 17.8 GB of assets pulled in 18 s,
peak VRAM 17.3 GB, €0.0134 in total (€0.0061 of GPU time, €0.0073 of dispatch
fee). The run:

    reference captions        similarity 0.3133
    grounded    CMER 0.0000   similarity 0.3357   confidence 0.8686
    leading     CMER 0.7917   similarity 0.2176   confidence 0.9031

Read the second row carefully before quoting it: 0.75 is not a hallucination
rate. Most of those answers are `There is no sign in the photo.` — correct
refusals, which score low similarity precisely because they describe nothing in
the image. CMER measures "confident text that does not match the picture", and
an honest refusal is exactly that. What the contrast does show is that the
metric fires on the prompt designed to be unanswerable and stays silent on the
one that isn't.

The grounded answers score *above* the human reference captions (0.3391 vs
0.3133) because the model writes full descriptive sentences while COCO captions
are terse — CLIP rewards the former.

Everything you install locally is ONE package: krauncher. torch, transformers
and datasets run on the GPU host.

    cd krauncher
    cp .env.example .env          # then put your CAS_API_KEY in it
    pip install krauncher
    python ../phases/04-computer-vision/25-vision-language-models/code/qwen3vl_cmer.py
"""

import asyncio

from krauncher import KrauncherClient, TaskError

client = KrauncherClient()  # reads CAS_API_KEY / .env from the folder you run in

HF_VLM = "hf://models/Qwen/Qwen3-VL-8B-Instruct"
# This repo predates safetensors: its weights are `pytorch_model.bin`, and the
# flax and tf copies beside it are 1.2 GB the run never opens.
HF_CLIP = ("hf://models/openai/clip-vit-base-patch32"
           "#allow=pytorch_model.bin,*.json,*.txt")
# One shard of the 10, which is 3,000 of the 30,000 photos — the rest would be
# 4.5 GB of bridge traffic for images this run never opens.
HF_COCO = ("hf://datasets/sayakpaul/coco-30-val-2014"
           "#allow=data/train-00000-of-00010-*.parquet")

GROUNDED = "Describe what is in this photo in one short sentence."
LEADING = ("What is the exact text written on the sign in this photo? "
           "Answer with the text only.")


# Empirical VRAM (measured): 17.3 GB peak — essentially the bf16 weights, since
# a batch of one generating 48 tokens costs activations you cannot see at this
# scale, and CLIP only arrives after the VLM is freed. It runs on a 24 GB card.
# The first run of this script landed on a 48 GB A40 instead: the analyzer sized
# the activations as if the decoder processed the photo at full resolution,
# which is what a CNN does and a VLM decoder never does. Same work on the right
# card costs €0.0134 against €0.0173.
@client.task(
    vram_gb=None,  # None => Krauncher reads the code, sizes VRAM, picks cheapest
    pip=["transformers>=4.57", "accelerate", "datasets"],  # torch is on the worker
    timeout=5400,
    data_urls=[HF_VLM, HF_CLIP, HF_COCO],
    dataset_size=18600,  # ~17.5 GB of VLM weights + 0.6 CLIP + 0.5 COCO shard
    disk_gb=50,
    stream_stderr=True,
)
def vqa_cmer(grounded, leading, n_images=24, max_new_tokens=48,
             sim_threshold=0.25, conf_threshold=0.8):
    # Why the imports are inside this function, and why the two prompts arrive
    # as arguments: this does not run on your machine. Krauncher ships it to a
    # rented GPU, where it runs in a fresh process that shares nothing with this
    # file -- neither file-level imports nor module-level constants travel with
    # it. Everything the function needs comes in through its parameters.
    print("Task started. Importing torch / transformers / datasets...", flush=True)
    # With E2E on, the worker's stderr and traceback do not reach the client;
    # stdout is relayed, so print the traceback before re-raising.
    try:
        import time

        import torch
        import transformers
        from datasets import load_dataset
        from transformers import AutoProcessor, CLIPModel, CLIPProcessor

        # Worth a line of output: half the API differences in this script are
        # differences between transformers versions.
        print(f"torch {torch.__version__}, transformers {transformers.__version__}",
              flush=True)

        VLM_PATH = "/data/Qwen__Qwen3-VL-8B-Instruct"
        CLIP_PATH = "/data/openai__clip-vit-base-patch32"
        COCO_PATH = "/data/sayakpaul__coco-30-val-2014"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)

        # Check the bridged assets before spending GPU-seconds: an `#allow`
        # pattern that matches no weight file leaves a directory of configs
        # that only fails when the second model is loaded, half an hour in.
        import glob
        import os
        for path, need in ((VLM_PATH, "*.safetensors"),
                           (CLIP_PATH, "pytorch_model.bin"),
                           (COCO_PATH, "data/*.parquet")):
            if not glob.glob(os.path.join(path, need)):
                raise FileNotFoundError(
                    f"{path} has no {need} — check the #allow pattern on its "
                    f"hf:// URL. Present: {sorted(os.listdir(path))[:8]}"
                )

        ds = load_dataset(
            "parquet",
            data_files={"train": f"{COCO_PATH}/data/train-00000-of-*.parquet"},
        )["train"].select(range(n_images))
        images = [im.convert("RGB") for im in ds["image"]]
        captions = list(ds["caption"])
        print(f"COCO: {len(images)} photos with reference captions", flush=True)

        # ─────────── lesson "Use It" code: load the VLM, answer a question ───────────
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(VLM_PATH)
        try:
            from transformers import AutoModelForImageTextToText as _VlmClass
        except ImportError:  # transformers < 4.57 spells it the snippet's way
            from transformers import AutoModelForVision2Seq as _VlmClass
        model = _VlmClass.from_pretrained(
            VLM_PATH, dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        print(f"Qwen3-VL loaded in {time.time() - t0:.1f}s", flush=True)

        @torch.no_grad()
        def answer(image, question):
            """One answer plus the mean probability of the tokens in it."""
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
            # ────────── end lesson "Use It" code ──────────
            # Confidence: geometric mean of the per-token probabilities, which
            # is what "high text confidence" in the CMER definition means.
            scores = model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True,
            )[0]
            scores = scores[torch.isfinite(scores)]
            confidence = scores.mean().exp().item() if len(scores) else 0.0
            new_tokens = out.sequences[0][inputs["input_ids"].shape[1]:]
            text = processor.decode(new_tokens, skip_special_tokens=True).strip()
            return text, confidence

        answers = {grounded: [], leading: []}
        confidences = {grounded: [], leading: []}
        t_gen = time.time()
        for i, image in enumerate(images, 1):
            for question in (grounded, leading):
                text, conf = answer(image, question)
                answers[question].append(text)
                confidences[question].append(conf)
            print(f"\r  answered {i}/{len(images)} photos", end="", flush=True)
        gen_sec = time.time() - t_gen
        print(f"  ({gen_sec:.1f}s, {2 * len(images) / gen_sec:.2f} answers/s)",
              flush=True)

        # The VLM is done; free the card before CLIP arrives on it.
        del model
        torch.cuda.empty_cache()

        clip = CLIPModel.from_pretrained(CLIP_PATH).to(device).eval()
        clip_proc = CLIPProcessor.from_pretrained(CLIP_PATH)

        @torch.no_grad()
        def similarity(texts):
            """Cosine similarity between each photo and its text, in CLIP space."""
            sims = []
            for start in range(0, len(images), 16):
                batch_img = images[start:start + 16]
                batch_txt = texts[start:start + 16]
                inp = clip_proc(
                    text=batch_txt, images=batch_img, return_tensors="pt",
                    padding=True, truncation=True, max_length=77,
                ).to(device)
                # One forward instead of get_image_features/get_text_features:
                # those return the projected tensor in some transformers
                # versions and the raw vision output in others, while
                # CLIPOutput.image_embeds/.text_embeds have been the projected
                # vectors throughout.
                out = clip(**inp)
                img_emb, txt_emb = out.image_embeds, out.text_embeds
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                sims.extend((img_emb * txt_emb).sum(-1).float().cpu().tolist())
            return sims

        # The lesson's own metric, on real inputs: high confidence next to low
        # image-text similarity is what a hallucination looks like from outside.
        def cmer(sims, confs):
            hits = [
                1.0 for s, c in zip(sims, confs)
                if c > conf_threshold and s < sim_threshold
            ]
            return len(hits) / len(sims)

        def mean(xs):
            return sum(xs) / len(xs)

        reference_sims = similarity(captions)
        report = {}
        for name, question in (("grounded", grounded), ("leading", leading)):
            sims = similarity(answers[question])
            report[name] = {
                "cmer": round(cmer(sims, confidences[question]), 4),
                "mean_sim": round(mean(sims), 4),
                "mean_conf": round(mean(confidences[question]), 4),
                "examples": answers[question][:3],
            }
            print(f"  [{name}] CMER={report[name]['cmer']:.4f} "
                  f"sim={report[name]['mean_sim']:.4f} "
                  f"conf={report[name]['mean_conf']:.4f}", flush=True)

        # Return only JSON-serializable data -- tensors do not cross back.
        return {
            "n_images": len(images),
            "gen_sec": round(gen_sec, 1),
            "reference_sim": round(mean(reference_sims), 4),
            "sim_threshold": sim_threshold,
            "conf_threshold": conf_threshold,
            **report,
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

    handle = await vqa_cmer(grounded=GROUNDED, leading=LEADING)
    print(f"Submitted {handle.task_id} -- waiting for the cheapest GPU...",
          flush=True)

    try:
        result = await handle.wait(on_log=_print_progress, timeout=5700)
    except TaskError as e:
        print("\n--- task failed on the GPU ---")
        print(getattr(e, "remote_traceback", None) or e)
        return

    out = result.output
    # The card and the bill are already in Krauncher's own closing line.
    print(f"\nAnswered:  {2 * out['n_images']} questions about "
          f"{out['n_images']} photos in {out['gen_sec']}s")
    print(f"Reference captions score {out['reference_sim']:.4f} similarity — "
          f"the threshold sits at {out['sim_threshold']}")
    for name in ("grounded", "leading"):
        r = out[name]
        print(f"  {name:<9} CMER {r['cmer']:.4f}   "
              f"similarity {r['mean_sim']:.4f}   confidence {r['mean_conf']:.4f}")
        print(f"            e.g. {r['examples'][0][:90]!r}")


if __name__ == "__main__":
    asyncio.run(main())
