#!/usr/bin/env bash
# 检查点中断恢复实验：kill -9 训练进程 → resume → loss 接续验证
# 产出：/mnt/workspace/runs/sft-activity/resume_report.txt
set -euo pipefail
RUN_DIR=/mnt/workspace/runs/sft-activity
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT=$RUN_DIR/resume_report.txt

mkdir -p "$RUN_DIR"
: > "$OUT"

echo "== [1] 首段训练（60 步后 kill -9）==" | tee -a "$OUT"
nohup python "$SCRIPT_DIR/sft_lora.py" --config "$SCRIPT_DIR/config.yaml" \
  > "$RUN_DIR/first_seg.log" 2>&1 &
TRAIN_PID=$!

# 轮询到 checkpoint 出现（ckpt_every=50 步）后再多跑 10 秒，然后硬杀
for i in $(seq 1 120); do
  if [ -f "$RUN_DIR/checkpoint/trainer_state.pt" ]; then
    sleep 10; break
  fi
  sleep 5
done
kill -9 $TRAIN_PID 2>/dev/null || true
pkill -9 -f "sft_lora.py" 2>/dev/null || true
sleep 5
PRE_STEP=$(python -c "import torch;print(torch.load('$RUN_DIR/checkpoint/trainer_state.pt',map_location='cpu',weights_only=False)['global_step'])")
echo "kill -9 at checkpoint step=$PRE_STEP" | tee -a "$OUT"

echo "== [2] resume 续训 ==" | tee -a "$OUT"
python "$SCRIPT_DIR/sft_lora.py" --config "$SCRIPT_DIR/config.yaml" --resume \
  > "$RUN_DIR/resume_seg.log" 2>&1
grep -m1 "resume-skip" "$RUN_DIR/resume_seg.log" >> "$OUT" || true

echo "== [3] loss 接续性检查 ==" | tee -a "$OUT"
# 首段最后一条日志 vs resume 段第一条日志：loss 应在同一量级（无跳变到初值）
LAST_FIRST=$(grep '"loss"' "$RUN_DIR/train_log.jsonl" | head -n $((PRE_STEP / 5 + 2)) | python -c \
  "import sys,json;ls=[json.loads(l) for l in sys.stdin];print(ls[-1]['loss'] if ls else 'NA')")
FIRST_RESUME=$(python -c "
import json
ls=[json.loads(l) for l in open('$RUN_DIR/train_log.jsonl',encoding='utf-8')]
after=[r for r in ls if r['step']>$PRE_STEP]
print(after[0]['loss'] if after else 'NA')")
echo "pre-kill last loss=$LAST_FIRST | post-resume first loss=$FIRST_RESUME" | tee -a "$OUT"
python - "$LAST_FIRST" "$FIRST_RESUME" <<'EOF' | tee -a "$OUT"
import sys
a, b = float(sys.argv[1]), float(sys.argv[2])
ratio = abs(a - b) / max(a, b, 1e-9)
print(f"loss jump ratio = {ratio:.3f} -> {'PASS (接续)' if ratio < 0.35 else 'CHECK (可能跳变)'}")
EOF

FINAL_STEP=$(python -c "import torch;print(torch.load('$RUN_DIR/checkpoint/trainer_state.pt',map_location='cpu',weights_only=False)['global_step'])")
echo "final step=$FINAL_STEP" | tee -a "$OUT"
echo "== resume 实验完成 ==" | tee -a "$OUT"
