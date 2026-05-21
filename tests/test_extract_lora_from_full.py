from __future__ import annotations

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools.extract_lora_from_full import extract_lora


def test_extract_lora_from_rank_one_delta(tmp_path) -> None:
    base_path = tmp_path / "base.safetensors"
    tuned_path = tmp_path / "tuned.safetensors"
    output_path = tmp_path / "extracted.safetensors"

    base_weight = torch.zeros(4, 4)
    up = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    down = torch.tensor([[1.0, -1.0, 0.5, 2.0]])
    delta = up @ down

    save_file({"blocks.0.q_proj.weight": base_weight}, str(base_path))
    save_file({"blocks.0.q_proj.weight": base_weight + delta}, str(tuned_path))

    errors = extract_lora(
        base_path,
        tuned_path,
        output_path,
        rank=1,
        alpha=1.0,
        target_patterns=["*q_proj.weight"],
    )

    assert errors["blocks.0.q_proj.weight"] < 1e-5
    tensors = load_file(str(output_path), device="cpu")
    assert set(tensors) == {
        "lora_unet_blocks_0_q_proj.alpha",
        "lora_unet_blocks_0_q_proj.lora_down.weight",
        "lora_unet_blocks_0_q_proj.lora_up.weight",
    }
    reconstructed = tensors["lora_unet_blocks_0_q_proj.lora_up.weight"] @ tensors["lora_unet_blocks_0_q_proj.lora_down.weight"]
    assert torch.allclose(reconstructed, delta, atol=1e-5)

    with safe_open(str(output_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    ss_args = json.loads(metadata["ss_network_args"])
    assert ss_args["algo"] == "lora"
    assert ss_args["source"] == "full_weight_diff_svd"


def test_extract_lora_filters_non_matching_weights(tmp_path) -> None:
    base_path = tmp_path / "base.safetensors"
    tuned_path = tmp_path / "tuned.safetensors"
    output_path = tmp_path / "extracted.safetensors"

    save_file({
        "blocks.0.q_proj.weight": torch.zeros(2, 2),
        "blocks.0.v_proj.weight": torch.zeros(2, 2),
    }, str(base_path))
    save_file({
        "blocks.0.q_proj.weight": torch.ones(2, 2),
        "blocks.0.v_proj.weight": torch.ones(2, 2),
    }, str(tuned_path))

    errors = extract_lora(
        base_path,
        tuned_path,
        output_path,
        rank=1,
        alpha=1.0,
        target_patterns=["*v_proj.weight"],
    )

    assert set(errors) == {"blocks.0.v_proj.weight"}
    assert set(load_file(str(output_path), device="cpu")) == {
        "lora_unet_blocks_0_v_proj.alpha",
        "lora_unet_blocks_0_v_proj.lora_down.weight",
        "lora_unet_blocks_0_v_proj.lora_up.weight",
    }
