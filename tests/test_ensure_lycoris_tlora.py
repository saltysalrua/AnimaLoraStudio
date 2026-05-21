from __future__ import annotations

import builtins
import subprocess
import sys
import types

from tools import ensure_lycoris_tlora


def test_has_tlora_false_when_module_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lycoris.modules.tlora":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ensure_lycoris_tlora.has_tlora() is False


def test_has_tlora_true_with_expected_api(monkeypatch) -> None:
    mod = types.ModuleType("lycoris.modules.tlora")

    class TLoraModule:
        name = "tlora"

    mod.TLoraModule = TLoraModule
    mod.compute_timestep_mask = lambda *args, **kwargs: None
    mod.set_timestep_mask = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "lycoris.modules.tlora", mod)

    assert ensure_lycoris_tlora.has_tlora() is True


def test_main_installs_when_tlora_missing(monkeypatch) -> None:
    calls: list[list[str]] = []
    states = iter([False, True])
    monkeypatch.setattr(ensure_lycoris_tlora, "has_tlora", lambda: next(states))
    monkeypatch.setattr(subprocess, "check_call", lambda cmd: calls.append(cmd))

    assert ensure_lycoris_tlora.main() == 0
    assert calls == [[
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        ensure_lycoris_tlora.LYCORIS_REQUIREMENT,
    ]]


def test_main_skips_install_when_tlora_present(monkeypatch) -> None:
    monkeypatch.setattr(ensure_lycoris_tlora, "has_tlora", lambda: True)
    monkeypatch.setattr(subprocess, "check_call", lambda cmd: (_ for _ in ()).throw(AssertionError(cmd)))

    assert ensure_lycoris_tlora.main() == 0
