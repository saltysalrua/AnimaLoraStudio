from __future__ import annotations

import subprocess
import sys

LYCORIS_REQUIREMENT = (
    "lycoris-lora @ "
    "git+https://github.com/KohakuBlueleaf/LyCORIS.git"
    "@ac63d7fa2d89347e4313aa2a4236a328d6bbd006"
)


def has_tlora() -> bool:
    try:
        from lycoris.modules.tlora import TLoraModule, compute_timestep_mask, set_timestep_mask
        return (
            getattr(TLoraModule, "name", None) == "tlora"
            and callable(compute_timestep_mask)
            and callable(set_timestep_mask)
        )
    except Exception:
        return False


def main() -> int:
    if has_tlora():
        return 0

    print("[studio] LyCORIS T-LoRA support missing; installing pinned upstream LyCORIS...")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        LYCORIS_REQUIREMENT,
    ]
    subprocess.check_call(cmd)
    if not has_tlora():
        raise RuntimeError("LyCORIS installed but T-LoRA support is still unavailable")
    print("[studio] LyCORIS T-LoRA support ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
