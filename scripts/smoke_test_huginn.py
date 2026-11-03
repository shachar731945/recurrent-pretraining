#!/usr/bin/env python3
"""Load Huginn-0125 and compare deterministic generations across recurrence depths."""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path


# Set the cache location before importing Transformers/Hugging Face Hub.  Files
# already present here are reused automatically on subsequent runs.
DEFAULT_HF_HOME = Path("/home/shaharav/vinkerg/shahar_avni_folder/HF_HOME")
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


MODEL_ID = "tomg-group-umd/huginn-0125"
DEFAULT_RECURRENCES = "1,2,4,8,16,32,64"


def parse_recurrences(value: str) -> list[int]:
    """Parse comma-separated depths or an inclusive range such as ``1-64``."""
    value = value.strip()
    if "-" in value and "," not in value:
        start_text, end_text = value.split("-", maxsplit=1)
        values = list(range(int(start_text), int(end_text) + 1))
    else:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]

    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("recurrence values must be positive integers")
    return values


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="How's it going?")
    parser.add_argument(
        "--recurrences",
        type=parse_recurrences,
        default=parse_recurrences(DEFAULT_RECURRENCES),
        help=(
            "Comma-separated recurrence depths (default: 1,2,4,8,16,32,64), "
            "or an inclusive range such as 1-64"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disable downloads and fail if the model is not already cached",
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"Model:   {args.model_id}")
    print(f"Device:  {device} ({dtype})")
    if device.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(device)}")
    else:
        print("Warning: CPU inference for this 3.5B model will be slow and memory intensive.")

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model_load_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    if importlib.util.find_spec("accelerate") is not None:
        model_load_kwargs["device_map"] = str(device)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_load_kwargs)
    if importlib.util.find_spec("accelerate") is None:
        print("Optional package 'accelerate' not found; moving the loaded model to the device.")
        model.to(device)
    model.eval()
    print(f"Loaded model in {time.perf_counter() - load_started:.1f}s")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(chat_text, return_tensors="pt", add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        return_dict_in_generate=True,
        eos_token_id=65505,
        bos_token_id=65504,
        pad_token_id=65509,
        stop_strings=["<|end_text|>", "<|end_turn|>"],
    )

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Recurrence depths: {args.recurrences}\n")
    for recurrence in args.recurrences:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                generation_config=generation_config,
                num_steps=recurrence,
                tokenizer=tokenizer,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        generated_ids = output.sequences[0, input_ids.shape[1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"r={recurrence:>2} ({elapsed:7.2f}s): {response}", flush=True)


if __name__ == "__main__":
    main()
