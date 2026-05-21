from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file


def _split_patterns(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[|,]", value or "") if part.strip()]


def _matches(name: str, patterns: Iterable[str]) -> bool:
    patterns = list(patterns)
    return not patterns or any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _lora_name(weight_name: str, prefix: str) -> str:
    name = weight_name
    if name.endswith(".weight"):
        name = name[:-7]
    name = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return f"{prefix}_{name}"


def _factorize_delta(delta: torch.Tensor, rank: int, alpha: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    if delta.ndim != 2:
        raise ValueError(f"only 2D weights are supported, got shape={tuple(delta.shape)}")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    max_rank = min(delta.shape)
    used_rank = min(rank, max_rank)

    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    scale = alpha / float(used_rank)
    down = vh[:used_rank, :].contiguous()
    up = (u[:, :used_rank] * s[:used_rank].unsqueeze(0) / scale).contiguous()

    recon = (up @ down) * scale
    denom = torch.linalg.norm(delta.float()).item()
    if denom == 0.0:
        rel_error = 0.0
    else:
        rel_error = torch.linalg.norm(delta.float() - recon).item() / denom
    return down, up, rel_error


def extract_lora(
    base_path: str | Path,
    tuned_path: str | Path,
    output_path: str | Path,
    *,
    rank: int,
    alpha: float,
    target_patterns: Iterable[str],
    prefix: str = "lora_unet",
) -> dict[str, float]:
    base = load_file(str(base_path), device="cpu")
    tuned = load_file(str(tuned_path), device="cpu")

    output: dict[str, torch.Tensor] = {}
    errors: dict[str, float] = {}
    patterns = list(target_patterns)

    for name, tuned_tensor in tuned.items():
        if name not in base:
            continue
        if not name.endswith(".weight"):
            continue
        if tuned_tensor.ndim != 2 or base[name].ndim != 2:
            continue
        if tuned_tensor.shape != base[name].shape:
            continue
        if not _matches(name, patterns):
            continue

        delta = tuned_tensor.float() - base[name].float()
        if torch.count_nonzero(delta).item() == 0:
            continue

        down, up, rel_error = _factorize_delta(delta, rank, alpha)
        lora_name = _lora_name(name, prefix)
        output[f"{lora_name}.lora_down.weight"] = down
        output[f"{lora_name}.lora_up.weight"] = up
        output[f"{lora_name}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
        errors[name] = rel_error

    if not output:
        raise RuntimeError("no matching 2D weight deltas were found")

    metadata = {
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps({
            "algo": "lora",
            "factor": 8,
            "preset": "anima_full",
            "source": "full_weight_diff_svd",
        }),
        "anima_extract_lora_errors": json.dumps(errors, sort_keys=True),
    }
    save_file(output, str(output_path), metadata=metadata)
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a LoRA approximation from full fine-tuned weights.")
    parser.add_argument("--base", required=True, help="Base model safetensors path")
    parser.add_argument("--tuned", required=True, help="Full fine-tuned model safetensors path")
    parser.add_argument("--output", required=True, help="Output LoRA safetensors path")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha; defaults to rank")
    parser.add_argument(
        "--target-pattern",
        default="*q_proj.weight|*k_proj.weight|*v_proj.weight|*output_proj.weight|*mlp.layer1.weight|*mlp.layer2.weight",
        help="fnmatch patterns separated by | or ,",
    )
    parser.add_argument("--prefix", default="lora_unet", help="LoRA key prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    alpha = float(args.rank if args.alpha is None else args.alpha)
    errors = extract_lora(
        args.base,
        args.tuned,
        args.output,
        rank=args.rank,
        alpha=alpha,
        target_patterns=_split_patterns(args.target_pattern),
        prefix=args.prefix,
    )
    mean_error = sum(errors.values()) / len(errors)
    print(f"extracted {len(errors)} layers to {args.output}; mean_relative_error={mean_error:.6g}")
    for name, error in sorted(errors.items()):
        print(f"{name}: relative_error={error:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
