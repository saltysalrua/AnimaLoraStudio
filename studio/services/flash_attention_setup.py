"""Flash Attention wheel 查找与安装。

逻辑对齐 install.sh / install.bat：
- 检测 Python / CUDA / PyTorch 版本 → 生成 pattern
- 从 GitHub Releases (mjun0812/flash-attention-prebuild-wheels) 查匹配 wheel
- pip install <url>（同步，可能几分钟）
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import urllib.request
from typing import Any, Optional


FA_RELEASES_URL = (
    "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases"
)


def detect_env() -> dict[str, Any]:
    """检测当前 Python / CUDA / PyTorch / 平台，返回 pattern 等信息。"""
    vi = sys.version_info
    python_tag = f"cp{vi.major}{vi.minor}"

    syst = platform.system().lower()
    mach = platform.machine().lower()
    if syst == "linux" and mach == "x86_64":
        plat = "linux_x86_64"
    elif syst == "windows" and mach in ("amd64", "x86_64"):
        plat = "win_amd64"
    else:
        plat = None

    cuda_tag: Optional[str] = None
    cuda_ver: Optional[str] = None
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout)
            if m:
                cuda_ver = f"{m.group(1)}.{m.group(2)}"
                cuda_tag = f"cu{m.group(1)}{m.group(2)}"
    except Exception:
        pass

    torch_tag: Optional[str] = None
    torch_ver: Optional[str] = None
    try:
        import torch  # type: ignore
        v = torch.__version__.split("+")[0].split(".")
        torch_tag = f"torch{v[0]}.{v[1]}"
        torch_ver = torch.__version__
    except ImportError:
        pass

    pattern: Optional[str] = None
    if cuda_tag and torch_tag and plat:
        pattern = f"{cuda_tag}{torch_tag}-{python_tag}-{python_tag}-{plat}"

    return {
        "python_tag": python_tag,
        "cuda_tag": cuda_tag,
        "cuda_ver": cuda_ver,
        "torch_tag": torch_tag,
        "torch_ver": torch_ver,
        "platform": plat,
        "pattern": pattern,
    }


def find_wheel(pattern: str) -> Optional[str]:
    """从 GitHub Releases 查找包含 pattern 的第一个 wheel URL。"""
    try:
        req = urllib.request.Request(
            FA_RELEASES_URL,
            headers={"User-Agent": "AnimaLoraStudio"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        for release in data:
            for asset in release.get("assets", []):
                if pattern in asset["name"]:
                    return asset["browser_download_url"]
    except Exception:
        pass
    return None


def current_status() -> dict[str, Any]:
    """当前 flash_attn 安装状态（包名 / 版本）。"""
    try:
        version = importlib.metadata.version("flash_attn")
        return {"installed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def install(url: Optional[str] = None) -> dict[str, Any]:
    """安装 flash_attn wheel。url=None 则自动查 GitHub。

    同步 pip install，可能需要几分钟；前端按钮必须带 loading 状态。
    flash_attn 是 C extension，pip 重装后必须重启进程才能切换。
    """
    env = detect_env()

    if url is None:
        pattern = env.get("pattern")
        if not pattern:
            raise RuntimeError(
                "无法确定环境 pattern（缺 CUDA / PyTorch 或不支持的平台）"
            )
        url = find_wheel(pattern)
        if not url:
            raise RuntimeError(
                f"未找到匹配 wheel（pattern: {pattern}）\n"
                "请前往 https://github.com/mjun0812/flash-attention-prebuild-wheels/releases "
                "手动粘贴 URL"
            )

    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", url],
        capture_output=True,
        text=True,
    )
    stdout = r.stdout + r.stderr
    tail = "\n".join(stdout.splitlines()[-40:])

    if r.returncode != 0:
        raise RuntimeError(f"pip install 失败:\n{tail}")

    try:
        importlib.invalidate_caches()
        version = importlib.metadata.version("flash_attn")
    except Exception:
        version = None

    return {
        "installed": True,
        "version": version,
        "url": url,
        "stdout_tail": tail,
        "restart_required": True,
    }
