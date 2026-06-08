"""
Qwen3-TTS Easy Finetuning — Modal deployment
============================================

Run the full Qwen3-TTS finetuning pipeline on serverless GPUs with
[Modal](https://modal.com). This script mirrors the project's own Dockerfile
(same PyTorch 2.5.1 / CUDA 12.4 base image, same pinned dependencies and
flash-attn wheel) and drives the existing `src/cli.py` so behaviour matches the
upstream Docker/CLI workflow exactly.

The pipeline stages (see README) are:

    prepare   ->  split + ASR + tokenize the raw audio for a speaker
    embed     ->  build speaker_emb.safetensors (REQUIRED before training)
    train     ->  supervised fine-tuning
    infer     ->  synthesize speech from a trained checkpoint

All persistent state lives in Modal Volumes so it survives between runs:

    qwen3-tts-models        ->  /workspace/models        (HF/ModelScope cache + weights)
    qwen3-tts-raw           ->  /workspace/raw-dataset    (your uploaded source audio)
    qwen3-tts-final         ->  /workspace/final-dataset  (processed clips, jsonl, embeddings)
    qwen3-tts-logs          ->  /workspace/logs           (tokenized codes, tensorboard)
    qwen3-tts-output        ->  /workspace/output         (training checkpoints)

--------------------------------------------------------------------------------
Quick start
--------------------------------------------------------------------------------

1. Install + authenticate Modal once:

       pip install modal
       modal setup

2. End-to-end (upload local wavs -> prepare -> embed -> train) in one command:

       modal run modal_app.py \
           --input-dir ./raw-dataset/my_speaker \
           --speaker-name my_speaker \
           --experiment-name exp1 \
           --epochs 3

3. Synthesize with the trained checkpoint (saves the wav locally):

       modal run modal_app.py::synth \
           --checkpoint output/exp1/checkpoint-epoch-2 \
           --speaker my_speaker \
           --text "Hello world, this is my custom voice." \
           --local-out ./my_voice.wav

4. Pull a checkpoint down to your machine when you're happy with it:

       modal volume get qwen3-tts-output /exp1 ./exp1-checkpoints

You can also call individual stages directly, e.g.:

       modal run modal_app.py::prepare --speaker-name my_speaker --experiment-name exp1 --input-dir raw-dataset/my_speaker
       modal run modal_app.py::embed   --speaker-name my_speaker --init-model Qwen/Qwen3-TTS-12Hz-0.6B-Base
       modal run modal_app.py::train   --speaker-name my_speaker --experiment-name exp1 --epochs 3
"""

import os
import subprocess

import modal

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

APP_NAME = "qwen3-tts-finetuning"

# GPUs. The 0.6B models fit comfortably on an A10G; bump TRAIN_GPU to
# "A100-80GB" for the 1.7B variants or larger batch sizes. See the README's
# hardware notes (>=16GB VRAM, 24GB recommended for 1.7B).
TRAIN_GPU = os.environ.get("QWEN_TTS_TRAIN_GPU", "A100-40GB")
PREP_GPU = os.environ.get("QWEN_TTS_PREP_GPU", "A10G")
INFER_GPU = os.environ.get("QWEN_TTS_INFER_GPU", "A10G")

# Up to Modal's 24h per-call ceiling for long training runs.
TRAIN_TIMEOUT = 24 * 60 * 60
PREP_TIMEOUT = 6 * 60 * 60
INFER_TIMEOUT = 30 * 60

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

# --------------------------------------------------------------------------- #
# Image — kept in lockstep with the repository Dockerfile                      #
# --------------------------------------------------------------------------- #

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python=None,  # use the image's own Python 3.11
    )
    .apt_install("git", "libsndfile1", "ffmpeg", "sox", "wget", "curl")
    .pip_install_from_requirements("requirements.txt")
    # qwen-tts / qwen-asr are installed --no-deps exactly like the Dockerfile so
    # they don't drag in conflicting transformers/torch pins.
    .run_commands(
        "pip install --no-cache-dir qwen-tts==0.1.1 qwen-asr==0.0.6 --no-deps",
        f"pip install --no-cache-dir {FLASH_ATTN_WHEEL} "
        f"|| pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation",
    )
    .env(
        {
            # get_project_root() in src/model_repository.py returns /workspace
            # when IS_DOCKER is set, anchoring every project-relative path there.
            "IS_DOCKER": "1",
            "PYTHONPATH": "/workspace/src",
            "PYTHONUNBUFFERED": "1",
            "USE_HF": "1",
            "HF_HOME": "/workspace/models/huggingface",
            "MODELSCOPE_CACHE": "/workspace/models/modelscope",
            # Avoid noisy tokenizers fork warnings during dataloading.
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .workdir("/workspace")
    # Ship the project source. Mounted at runtime, which is all the CLI needs.
    .add_local_dir("src", "/workspace/src")
)

app = modal.App(APP_NAME, image=image)

# --------------------------------------------------------------------------- #
# Persistent storage                                                          #
# --------------------------------------------------------------------------- #

models_vol = modal.Volume.from_name("qwen3-tts-models", create_if_missing=True)
raw_vol = modal.Volume.from_name("qwen3-tts-raw", create_if_missing=True)
final_vol = modal.Volume.from_name("qwen3-tts-final", create_if_missing=True)
logs_vol = modal.Volume.from_name("qwen3-tts-logs", create_if_missing=True)
output_vol = modal.Volume.from_name("qwen3-tts-output", create_if_missing=True)

VOLUMES = {
    "/workspace/models": models_vol,
    "/workspace/raw-dataset": raw_vol,
    "/workspace/final-dataset": final_vol,
    "/workspace/logs": logs_vol,
    "/workspace/output": output_vol,
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _run_cli(*cli_args: str) -> None:
    """Invoke src/cli.py as a subprocess, streaming its output to Modal logs.

    Running the CLI (rather than re-importing its internals) keeps this script
    faithful to the documented `python src/cli.py ...` workflow and to any
    future changes in the project's command handlers.
    """
    cmd = ["python", "/workspace/src/cli.py", *[str(a) for a in cli_args]]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    # Inherit stdout/stderr so progress bars and logs stream live.
    subprocess.run(cmd, cwd="/workspace", check=True)


def _commit_all() -> None:
    """Flush every volume so freshly written artifacts are durable."""
    for vol in {id(v): v for v in VOLUMES.values()}.values():
        vol.commit()


# --------------------------------------------------------------------------- #
# Pipeline stages                                                             #
# --------------------------------------------------------------------------- #


@app.function(gpu=PREP_GPU, volumes=VOLUMES, timeout=PREP_TIMEOUT)
def prepare(
    speaker_name: str,
    experiment_name: str,
    input_dir: str,
    ref_audio: str | None = None,
    asr_model: str = "Qwen/Qwen3-ASR-1.7B",
    batch_size: int = 16,
    model_source: str = "HuggingFace",
    threads: int = 6,
    skip_split: bool = False,
) -> None:
    """Steps 1-3: split + resample, ASR transcribe/clean, and tokenize audio.

    `input_dir` is project-relative (e.g. ``raw-dataset/my_speaker``) and must
    already contain the speaker's .wav files in the raw volume.
    """
    args = [
        "prepare",
        "--input_dir", input_dir,
        "--speaker_name", speaker_name,
        "--experiment_name", experiment_name,
        "--asr_model", asr_model,
        "--batch_size", batch_size,
        "--model_source", model_source,
        "--threads", threads,
        "--gpu", "cuda:0",
    ]
    if ref_audio:
        args += ["--ref_audio", ref_audio]
    if skip_split:
        args += ["--skip_split"]
    _run_cli(*args)
    _commit_all()


@app.function(gpu=PREP_GPU, volumes=VOLUMES, timeout=PREP_TIMEOUT)
def embed(
    speaker_name: str,
    init_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_source: str = "HuggingFace",
    mode: str = "ref",
    ref: str | None = None,
) -> None:
    """Generate speaker_emb.safetensors. REQUIRED before training (Base or
    CustomVoice). `speaker_name` may be comma-separated for multi-speaker."""
    args = [
        "embed",
        "--speaker_name", speaker_name,
        "--init_model", init_model,
        "--model_source", model_source,
        "--mode", mode,
    ]
    if ref:
        args += ["--ref", ref]
    _run_cli(*args)
    _commit_all()


@app.function(gpu=TRAIN_GPU, volumes=VOLUMES, timeout=TRAIN_TIMEOUT)
def train(
    speaker_name: str,
    experiment_name: str,
    init_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_source: str = "HuggingFace",
    batch_size: int = 2,
    lr: float = 1e-7,
    epochs: int = 2,
    grad_acc: int = 4,
    save_strategy: str = "both",
    save_steps: int = 200,
    keep_last_n_checkpoints: int = 3,
    resume_from_checkpoint: str = "latest",
    use_accelerator: bool = False,
) -> None:
    """Run supervised fine-tuning. Expects `prepare` (codes) and `embed`
    (speaker embeddings) to have run for this experiment/speaker already."""
    args = [
        "train",
        "--experiment_name", experiment_name,
        "--speaker_name", speaker_name,
        "--init_model", init_model,
        "--model_source", model_source,
        "--batch_size", batch_size,
        "--lr", lr,
        "--epochs", epochs,
        "--grad_acc", grad_acc,
        "--save_strategy", save_strategy,
        "--save_steps", save_steps,
        "--keep_last_n_checkpoints", keep_last_n_checkpoints,
        "--resume_from_checkpoint", resume_from_checkpoint,
        "--gpu", "cuda:0",
    ]
    if use_accelerator:
        args += ["--use_accelerator"]
    _run_cli(*args)
    _commit_all()


@app.function(gpu=INFER_GPU, volumes=VOLUMES, timeout=INFER_TIMEOUT)
def infer(
    checkpoint: str,
    text: str,
    speaker: str = "my_speaker",
    language: str = "English",
    instruct: str | None = None,
) -> bytes:
    """Synthesize speech from a trained checkpoint and return the WAV bytes.

    `checkpoint` is project-relative (e.g. ``output/exp1/checkpoint-epoch-1``).
    Use the `synth` local entrypoint to save the returned wav to disk.
    """
    remote_out = "/workspace/output/_modal_infer.wav"
    _run_cli(
        "infer",
        "--checkpoint", checkpoint,
        "--speaker", speaker,
        "--language", language,
        "--text", text,
        "--output", remote_out,
        "--gpu", "cuda:0",
        *(["--instruct", instruct] if instruct else []),
    )
    with open(remote_out, "rb") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Local entrypoints                                                           #
# --------------------------------------------------------------------------- #


@app.local_entrypoint()
def main(
    speaker_name: str,
    experiment_name: str,
    input_dir: str,
    init_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_source: str = "HuggingFace",
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 1e-7,
    grad_acc: int = 4,
    ref_audio: str | None = None,
    skip_prepare: bool = False,
    skip_embed: bool = False,
):
    """End-to-end driver: upload local audio, then prepare -> embed -> train.

    If `input_dir` is a path on your machine, its .wav files are uploaded into
    the raw-dataset volume under the speaker name. If it's already a project-
    relative path inside the volume, pass --skip-prepare or just reuse it.
    """
    project_input_dir = f"raw-dataset/{speaker_name}"

    # Upload local source audio into the raw volume (only if a local dir given).
    if os.path.isdir(input_dir):
        print(f"Uploading audio from {input_dir} -> raw volume:{project_input_dir}")
        with raw_vol.batch_upload(force=True) as batch:
            batch.put_directory(input_dir, speaker_name)
        remote_input = project_input_dir
    else:
        # Treat as an already-present project-relative path.
        remote_input = input_dir

    if not skip_prepare:
        print("=== Stage 1/3: prepare (split + ASR + tokenize) ===")
        prepare.remote(
            speaker_name=speaker_name,
            experiment_name=experiment_name,
            input_dir=remote_input,
            ref_audio=ref_audio,
            model_source=model_source,
        )

    if not skip_embed:
        print("=== Stage 2/3: embed (speaker embeddings) ===")
        embed.remote(
            speaker_name=speaker_name,
            init_model=init_model,
            model_source=model_source,
        )

    print("=== Stage 3/3: train ===")
    train.remote(
        speaker_name=speaker_name,
        experiment_name=experiment_name,
        init_model=init_model,
        model_source=model_source,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        grad_acc=grad_acc,
    )

    print(
        "\n✅ Done. Fetch checkpoints with:\n"
        f"   modal volume get qwen3-tts-output /{experiment_name} ./{experiment_name}-checkpoints\n"
        "Then synthesize, e.g.:\n"
        f"   modal run modal_app.py::synth --checkpoint output/{experiment_name}/checkpoint-epoch-{max(epochs - 1, 0)} "
        f"--speaker {speaker_name} --text \"Hello world\" --local-out ./voice.wav"
    )


@app.local_entrypoint()
def synth(
    checkpoint: str,
    text: str,
    speaker: str = "my_speaker",
    language: str = "English",
    instruct: str | None = None,
    local_out: str = "qwen3_tts_output.wav",
):
    """Convenience entrypoint to synthesize and save a wav locally."""
    wav = infer.remote(
        checkpoint=checkpoint,
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
    )
    with open(local_out, "wb") as f:
        f.write(wav)
    print(f"✅ Saved {len(wav)} bytes -> {local_out}")
