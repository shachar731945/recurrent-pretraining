#!/usr/bin/env python3
"""Find safe task-specific batch sizes and update the evaluation runner."""

from __future__ import annotations

import gc
import json
import os
import re
import time
import ast
from datetime import datetime, timezone
from pathlib import Path


# ----------------------------- Configuration -----------------------------

MODEL_ID = "tomg-group-umd/huginn-0125"
HF_HOME = "/home/shaharav/vinkerg/shahar_avni_folder/HF_HOME"
DEVICE = "cuda:0"
DTYPE = "bfloat16"
RECURRENCE = 64
CANDIDATE_BATCH_SIZES = [128, 64, 32, 16, 8, 4, 2, 1]
PROBE_EXAMPLES = 20
MAX_MEMORY_FRACTION = 0.90
SEED = 42
LOCAL_FILES_ONLY = True
CACHE_REQUESTS = True
RESUME = True
RUNNER_FILE = Path(__file__).with_name("run_recurrence_evaluations.py")
OUTPUT_FILE = Path(__file__).with_name("batch_calibration.json")

SYSTEM_INSTRUCTION = "You are a helpful assistant that can assist users with mathematical reasoning."

TASKS = {
    "arc_challenge": {"num_fewshot": 0},
    "gsm8k_cot": {
        "num_fewshot": 8,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "system_instruction": SYSTEM_INSTRUCTION,
    },
    "openbookqa": {"num_fewshot": 0},
    "hellaswag": {"num_fewshot": 0},
    "winogrande": {"num_fewshot": 0},
    "mmlu_humanities": {
        "calibration_task": "mmlu_professional_law",
        "num_fewshot": 5,
    },
}

# GSM8K is deliberately omitted for now: its long autoregressive answers make
# a realistic r=64 batch probe slow. Its existing safe batch size remains 1.
TASKS_TO_CALIBRATE = [
    "arc_challenge",
    "openbookqa",
    "hellaswag",
    "winogrande",
    "mmlu_humanities",
]

BATCH_BLOCK_START = "# BEGIN CALIBRATED BATCH SIZES"
BATCH_BLOCK_END = "# END CALIBRATED BATCH SIZES"


os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from tqdm.auto import tqdm


def release_gpu_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def probe_batch_size(model: HFLM, task_name: str, settings: dict, batch_size: int) -> dict:
    """Run enough real examples to exercise one full batch at the largest r."""
    probe_task = settings.get("calibration_task", task_name)
    limit = max(PROBE_EXAMPLES, batch_size)
    model._model.config.mean_recurrence = RECURRENCE
    model.batch_size_per_gpu = batch_size
    model.batch_sizes.clear()
    release_gpu_memory()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    try:
        simple_evaluate(
            model=model,
            tasks=[probe_task],
            num_fewshot=settings["num_fewshot"],
            batch_size=batch_size,
            cache_requests=CACHE_REQUESTS,
            limit=limit,
            bootstrap_iters=0,
            log_samples=False,
            apply_chat_template=settings.get("apply_chat_template", False),
            fewshot_as_multiturn=settings.get("fewshot_as_multiturn", False),
            system_instruction=settings.get("system_instruction"),
            random_seed=SEED,
            numpy_random_seed=SEED,
            torch_random_seed=SEED,
            fewshot_random_seed=SEED,
        )
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        total_bytes = torch.cuda.get_device_properties(torch.device(DEVICE)).total_memory
        fraction = peak_bytes / total_bytes
        return {
            "batch_size": batch_size,
            "status": "safe" if fraction <= MAX_MEMORY_FRACTION else "too_close_to_limit",
            "peak_gib": peak_bytes / 1024**3,
            "memory_fraction": fraction,
            "elapsed_seconds": time.perf_counter() - started,
            "examples": limit,
            "probe_task": probe_task,
        }
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not isinstance(error, torch.OutOfMemoryError) and "out of memory" not in str(error).lower():
            raise
        return {
            "batch_size": batch_size,
            "status": "out_of_memory",
            "elapsed_seconds": time.perf_counter() - started,
            "examples": limit,
            "probe_task": probe_task,
        }
    finally:
        release_gpu_memory()


def runner_batch_sizes() -> dict[str, int]:
    """Read the currently configured sizes so uncalibrated tasks are retained."""
    source = RUNNER_FILE.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(BATCH_BLOCK_START)}\s*TASK_BATCH_SIZES\s*=\s*(\{{.*?\}})\s*{re.escape(BATCH_BLOCK_END)}",
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise RuntimeError("Could not read TASK_BATCH_SIZES from the evaluation runner")
    return ast.literal_eval(match.group(1))


def update_runner(batch_sizes: dict[str, int]) -> None:
    """Replace only the marked batch-size configuration block."""
    source = RUNNER_FILE.read_text(encoding="utf-8")
    mapping = json.dumps(batch_sizes, indent=4)
    replacement = f"{BATCH_BLOCK_START}\nTASK_BATCH_SIZES = {mapping}\n{BATCH_BLOCK_END}"
    pattern = re.compile(
        rf"{re.escape(BATCH_BLOCK_START)}.*?{re.escape(BATCH_BLOCK_END)}",
        flags=re.DOTALL,
    )
    updated, count = pattern.subn(replacement, source)
    if count != 1:
        raise RuntimeError(f"Expected one calibrated batch-size block, found {count}")
    RUNNER_FILE.write_text(updated, encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for batch-size calibration")

    print(f"Loading {MODEL_ID} on {DEVICE} ...", flush=True)
    model = HFLM(
        pretrained=MODEL_ID,
        device=DEVICE,
        dtype=DTYPE,
        batch_size=1,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model._model.eval()

    report = {
        "configuration": {
            "model_id": MODEL_ID,
            "device": DEVICE,
            "recurrence": RECURRENCE,
            "candidate_batch_sizes": CANDIDATE_BATCH_SIZES,
            "probe_examples": PROBE_EXAMPLES,
            "max_memory_fraction": MAX_MEMORY_FRACTION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "tasks": {},
    }
    if RESUME and OUTPUT_FILE.exists():
        report = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        if report["configuration"].get("recurrence") != RECURRENCE:
            raise ValueError("Cannot resume a calibration with a different recurrence")

    chosen = {
        task_name: task_result["chosen_batch_size"]
        for task_name, task_result in report["tasks"].items()
    }

    for task_name in TASKS_TO_CALIBRATE:
        settings = TASKS[task_name]
        if task_name in chosen:
            print(f"{task_name}: already calibrated at batch {chosen[task_name]}; skipping")
            continue
        print(f"\n=== Calibrating {task_name} at r={RECURRENCE} ===", flush=True)
        attempts = []
        for batch_size in tqdm(CANDIDATE_BATCH_SIZES, desc=task_name, unit="batch"):
            attempt = probe_batch_size(model, task_name, settings, batch_size)
            attempts.append(attempt)
            print(
                f"batch={batch_size}: {attempt['status']}"
                + (f", peak={attempt['peak_gib']:.1f} GiB" if "peak_gib" in attempt else ""),
                flush=True,
            )
            if attempt["status"] == "safe":
                chosen[task_name] = batch_size
                break

        if task_name not in chosen:
            raise RuntimeError(f"No safe batch size found for {task_name}")
        report["tasks"][task_name] = {"chosen_batch_size": chosen[task_name], "attempts": attempts}
        OUTPUT_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    update_runner(runner_batch_sizes() | chosen)
    report["updated_runner"] = str(RUNNER_FILE)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nUpdated {RUNNER_FILE}")
    print(f"Calibration details saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
