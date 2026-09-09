#!/usr/bin/env bash
set -euo pipefail
cd "$HOME"

echo "=== fetch bazelisk ==="
mkdir -p "$HOME/bin"
if [ ! -x "$HOME/bin/bazel" ]; then
  curl -fsSL -o "$HOME/bin/bazel" \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
  chmod +x "$HOME/bin/bazel"
fi
export PATH="$HOME/bin:$PATH"
echo "bazel(isk): $(command -v bazel)"

echo "=== clone visqol ==="
if [ ! -d "$HOME/visqol/.git" ]; then
  git clone --depth 1 https://github.com/google/visqol.git "$HOME/visqol" 2>&1 | tail -3
fi
cd "$HOME/visqol"
echo "bazelversion: $(cat .bazelversion 2>/dev/null || echo none)"
echo "head: $(git log --oneline -1)"

echo "=== build :visqol CLI (opt) ==="
# system python3 on PATH is a user 3.14 (no distutils); use stock 3.10 which has it.
# Only the include PATH string is needed for the C++ CLI target (no py-binding compile).
export PYTHON_BIN_PATH=/usr/bin/python3.10
bazel build -c opt \
  --repo_env=PYTHON_BIN_PATH=/usr/bin/python3.10 \
  --action_env=PYTHON_BIN_PATH=/usr/bin/python3.10 \
  //:visqol 2>&1 | tail -40

echo "=== binary ==="
ls -la "$HOME/visqol/bazel-bin/visqol" 2>&1 || echo "BINARY NOT FOUND"
echo "BUILD_DONE_OK"
