#!/usr/bin/env python3
"""Evaluate Huginn-0125 on six benchmarks at several recurrence depths."""

from __future__ import annotations

import json
import os
import time
import gc
import traceback
from datetime import datetime, timezone
from pathlib import Path


# ----------------------------- Configuration -----------------------------

MODEL_ID = "tomg-group-umd/huginn-0125"
HF_HOME = "/home/shaharav/vinkerg/shahar_avni_folder/HF_HOME"
DEVICE = "cuda:0"
DTYPE = "bfloat16"
BATCH_SIZE = 4
MODEL_MAX_LENGTH = 4096
LOCAL_FILES_ONLY = True
CACHE_REQUESTS = True
RESUME = True
# Retry only the unfinished MMLU depths. With RESUME=True, the existing
# r=1/4/8/16 records in results.json are left untouched.
RECURRENCE_VALUES = [32, 64]
SEED = 42
LIMIT = None  # Set to a small integer, such as 20, for a quick trial run.
BOOTSTRAP_ITERS = 1000
OUTPUT_FILE = Path(__file__).with_name("results.json")

# BEGIN CALIBRATED BATCH SIZES
TASK_BATCH_SIZES = {
    "arc_challenge": 128,
    "gsm8k_cot": 1,
    "openbookqa": 128,
    "hellaswag": 128,
    "winogrande": 128,
    "mmlu_humanities": 4
}
# END CALIBRATED BATCH SIZES

SYSTEM_INSTRUCTION = "You are a helpful assistant that can assist users with mathematical reasoning."

EVALUATIONS = {
    # Figure 1 benchmarks
    "arc_challenge": {
        "num_fewshot": 0,
        "metric": "acc_norm,none",
    },
    "gsm8k_cot": {
        "num_fewshot": 8,
        "metric": "exact_match,flexible-extract",
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "system_instruction": SYSTEM_INSTRUCTION,
    },
    "openbookqa": {
        "num_fewshot": 0,
        "metric": "acc_norm,none",
    },
    # Additional non-math/non-science multiple-choice benchmarks
    "hellaswag": {
        "num_fewshot": 0,
        "metric": "acc_norm,none",
    },
    "winogrande": {
        "num_fewshot": 0,
        "metric": "acc,none",
    },
    "mmlu_humanities": {
        "num_fewshot": 5,
        "metric": "acc,none",
    },
}

# Run only the missing benchmark. Existing results remain in results.json.
TASKS_TO_RUN = ["mmlu_humanities"]


# Set the shared Hugging Face cache before importing HF-dependent packages.
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from tqdm.auto import tqdm


def save_results(payload: dict) -> None:
    """Write a checkpoint after every task/depth pair."""
    OUTPUT_FILE.write_text(
        json.dumps(payload, indent=2, default=handle_non_serializable) + "\n",
        encoding="utf-8",
    )


def get_primary_score(report: dict, task_name: str, metric_name: str) -> float | None:
    """Read the configured metric from either a task or grouped-task result."""
    for section_name in ("groups", "results"):
        task_result = report.get(section_name, {}).get(task_name, {})
        if metric_name in task_result:
            return float(task_result[metric_name])
    return None


def release_gpu_memory() -> None:
    """Release temporary tensors and cached allocator blocks between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available() and DEVICE.startswith("cuda"):
        raise RuntimeError(f"{DEVICE} was requested, but PyTorch cannot see a CUDA GPU")
    unknown_tasks = set(TASKS_TO_RUN) - set(EVALUATIONS)
    if unknown_tasks:
        raise ValueError(f"Unknown task names: {sorted(unknown_tasks)}")

    print(f"Loading {MODEL_ID} from {HF_HOME} on {DEVICE} ...", flush=True)
    model = HFLM(
        pretrained=MODEL_ID,
        device=DEVICE,
        dtype=DTYPE,
        batch_size=BATCH_SIZE,
        max_length=MODEL_MAX_LENGTH,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model._model.eval()

    configuration = {
        "model_id": MODEL_ID,
        "hf_home": HF_HOME,
        "device": DEVICE,
        "dtype": DTYPE,
        "batch_size": BATCH_SIZE,
        "model_max_length": MODEL_MAX_LENGTH,
        "task_batch_sizes": TASK_BATCH_SIZES,
        "local_files_only": LOCAL_FILES_ONLY,
        "cache_requests": CACHE_REQUESTS,
        "recurrence_values": RECURRENCE_VALUES,
        "seed": SEED,
        "limit": LIMIT,
        "bootstrap_iters": BOOTSTRAP_ITERS,
        "evaluations": EVALUATIONS,
        "tasks_to_run": TASKS_TO_RUN,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    output = {
        "configuration": configuration,
        "runs": [],
        "failures": [],
    }

    if RESUME and OUTPUT_FILE.exists():
        output = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        previous = output["configuration"]
        original_started_at = previous.get("started_at")
        for key in ("model_id", "seed", "limit"):
            if previous.get(key) != configuration[key]:
                raise ValueError(
                    f"Cannot resume: {key} changed from {previous.get(key)!r} to {configuration[key]!r}. "
                    "Set RESUME = False to start a new results file."
                )
        output["configuration"].update(configuration)
        if original_started_at is not None:
            output["configuration"]["started_at"] = original_started_at
        output["configuration"].pop("finished_at", None)
        output["configuration"]["resumed_at"] = datetime.now(timezone.utc).isoformat()
        output.setdefault("failures", [])

    completed = {(run["task"], run["recurrence"]) for run in output["runs"]}
    save_results(output)

    for task_name in TASKS_TO_RUN:
        task_config = EVALUATIONS[task_name]
        print(f"\n=== {task_name} ===", flush=True)
        progress = tqdm(RECURRENCE_VALUES, desc=task_name, unit="depth")

        for recurrence in progress:
            if (task_name, recurrence) in completed:
                progress.write(f"{task_name}, r={recurrence}: already complete; skipping")
                continue

            task_batch_size = TASK_BATCH_SIZES.get(task_name, BATCH_SIZE)
            progress.set_postfix(r=recurrence)
            model._model.config.mean_recurrence = recurrence
            model.batch_size_per_gpu = task_batch_size
            model.batch_sizes.clear()
            release_gpu_memory()
            started = time.perf_counter()

            try:
                report = simple_evaluate(
                    model=model,
                    tasks=[task_name],
                    num_fewshot=task_config["num_fewshot"],
                    batch_size=task_batch_size,
                    cache_requests=CACHE_REQUESTS,
                    limit=LIMIT,
                    bootstrap_iters=BOOTSTRAP_ITERS,
                    log_samples=False,
                    apply_chat_template=task_config.get("apply_chat_template", False),
                    fewshot_as_multiturn=task_config.get("fewshot_as_multiturn", False),
                    system_instruction=task_config.get("system_instruction"),
                    random_seed=SEED,
                    numpy_random_seed=SEED,
                    torch_random_seed=SEED,
                    fewshot_random_seed=SEED,
                )
                if report is None:
                    raise RuntimeError("lm-eval returned no report")
            except Exception as error:
                # Replace a previous failure for this exact retry rather than
                # accumulating duplicate failure records in results.json.
                output["failures"] = [
                    failure
                    for failure in output["failures"]
                    if (failure["task"], failure["recurrence"]) != (task_name, recurrence)
                ]
                output["failures"].append(
                    {
                        "task": task_name,
                        "recurrence": recurrence,
                        "batch_size": task_batch_size,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                save_results(output)
                release_gpu_memory()
                progress.write(f"{task_name}, r={recurrence}: failed; skipping remaining depths for this task")
                break

            elapsed = time.perf_counter() - started
            score = get_primary_score(report, task_name, task_config["metric"])
            output["runs"].append(
                {
                    "task": task_name,
                    "recurrence": recurrence,
                    "primary_metric": task_config["metric"],
                    "primary_score": score,
                    "batch_size": task_batch_size,
                    "elapsed_seconds": elapsed,
                    "results": report.get("results", {}),
                    "groups": report.get("groups", {}),
                }
            )
            output["failures"] = [
                failure
                for failure in output["failures"]
                if (failure["task"], failure["recurrence"]) != (task_name, recurrence)
            ]
            save_results(output)
            completed.add((task_name, recurrence))
            release_gpu_memory()
            progress.write(f"{task_name}, r={recurrence}: score={score}, time={elapsed:.1f}s")

    output["configuration"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_results(output)
    print(f"\nFinished. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
