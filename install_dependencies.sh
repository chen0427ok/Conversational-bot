#!/usr/bin/env bash
set -e
set -o pipefail

echo "🚀 開始安裝依賴..."

python -m pip install --upgrade pip

# ---------- 科學計算 & 音訊處理 ----------
pip install --retries 3 --default-timeout 120 \
    numpy==1.22.4 \
    scipy==1.7.3 \
    librosa==0.9.2 \
    scikit-learn==1.0.2 \
    soundfile==0.12.1

# ---------- Resemblyzer 語者嵌入 ----------
pip install --retries 3 --default-timeout 120 resemblyzer==0.1.4

# ---------- 若無 PyTorch 則安裝 CPU 版 ----------
python - <<'PY'
import importlib.util, subprocess, sys, textwrap
if importlib.util.find_spec("torch") is None:
    print("🧩 未偵測到 PyTorch，安裝 cpu 版 2.1.0")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "torch==2.1.0+cpu", "-f",
        "https://download.pytorch.org/whl/torch_stable.html"
    ])
else:
    import torch; print(f"✅ 已偵測到 PyTorch {torch.__version__}，跳過安裝")
PY

echo "✅ 依賴安裝完成！"
