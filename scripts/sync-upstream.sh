#!/usr/bin/env bash
set -euo pipefail

# sync-upstream.sh — 将 butterfly-dream 源文件复制到 hermes-agent 内置插件目录
# 使用方式:
#   ./scripts/sync-upstream.sh              # 复制到默认 hermes-agent 位置
#   HERMES_AGENT=/path/to/hermes-agent ./scripts/sync-upstream.sh  # 自定义路径
#
# 注意: 只复制 src/butterfly_dream/ 下的源文件
#       不会修改 hermes-agent 的 plugin.yaml 或触发 git commit

HERMES_AGENT="${HERMES_AGENT:-$HOME/.hermes/hermes-agent}"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)/src/butterfly_dream"
DST_DIR="$HERMES_AGENT/plugins/memory/butterfly_dream"

if [ ! -d "$SRC_DIR" ]; then
  echo "❌ 找不到源目录: $SRC_DIR"
  echo "  请在 butterfly-dream 仓库根目录执行 scripts/sync-upstream.sh"
  exit 1
fi

if [ ! -d "$DST_DIR" ]; then
  echo "📁 创建目标目录: $DST_DIR"
  mkdir -p "$DST_DIR"
fi

echo "🦋 同步 butterfly-dream → hermes-agent 内置插件"
echo "  源:     $SRC_DIR"
echo "  目标:   $DST_DIR"
echo ""

# 只复制 .py 源文件
PY_FILES=$(find "$SRC_DIR" -maxdepth 1 -name '*.py' | sort)
FILE_COUNT=0
for f in $PY_FILES; do
  filename=$(basename "$f")
  cp "$f" "$DST_DIR/$filename"
  echo "  ✓ $filename"
  FILE_COUNT=$((FILE_COUNT + 1))
done

# 复制 plugin.yaml 如果存在（但保留 hermes-agent 已有的）
if [ -f "$SRC_DIR/../plugin.yaml" ] && [ ! -f "$DST_DIR/plugin.yaml" ]; then
  cp "$SRC_DIR/../plugin.yaml" "$DST_DIR/plugin.yaml"
  echo "  ✓ plugin.yaml (新)"
  FILE_COUNT=$((FILE_COUNT + 1))
fi

echo ""
echo "✅ 同步完成: $FILE_COUNT 个文件已复制"
echo ""
echo "下一步:"
echo "  cd $HERMES_AGENT && git add plugins/memory/butterfly_dream/ && git commit -m \"chore: sync butterfly-dream v<version>\""
