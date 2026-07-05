#!/bin/bash
# run_on_pragya.sh
# Launch main.py on PRAGYA inside a detached tmux session "befgc".
# PRAGYA's login node has NO `screen` (uses tmux) and runs on CPU.
#
# Usage on PRAGYA:
#   bash run_on_pragya.sh              # (re)launch main.py --dataset Cora
#   bash run_on_pragya.sh sweep        # launch sweep.py instead
#
# Then:
#   tmux attach -t befgc               # watch live   (detach: Ctrl-B then D)
#   tmux kill-session -t befgc         # stop the run

SESSION="befgc"
PROJECT="/home/maths/mtech/mt6210961/Static_coarsening/static_coarsening_2/befgc"
CONDA_ENV="fgc_comp"
CONDA_SH="/apps/anaconda3/2025.06/etc/profile.d/conda.sh"

MODE="${2:-main}"
if [ "$1" = "sweep" ]; then MODE="sweep"; fi

if [ "$1" = "inner" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    cd "$PROJECT" || { echo "cd failed"; exec bash; }
    LOG="run_$(date +%Y%m%d_%H%M%S).log"
    if [ "$2" = "sweep" ]; then
        SCRIPT="sweep.py --dataset Cora"
    else
        SCRIPT="main.py --dataset Cora"
    fi
    echo "[run_on_pragya] python=$(which python3)"
    echo "[run_on_pragya] starting $SCRIPT -> $LOG"
    python3 $SCRIPT 2>&1 | tee "$LOG"
    echo "[run_on_pragya] FINISHED (exit ${PIPESTATUS[0]}). Ctrl-B D to detach, or 'exit'."
    exec bash
fi

echo "[run_on_pragya] stopping any existing run + session..."
pkill -f "main.py --dataset" 2>/dev/null
pkill -f "sweep.py --dataset" 2>/dev/null
tmux kill-session -t "$SESSION" 2>/dev/null
sleep 1

tmux new-session -d -s "$SESSION" "bash '$PROJECT/run_on_pragya.sh' inner $1"
echo "[run_on_pragya] launched in tmux session '$SESSION'."
echo ""
echo "Attach to watch live:   tmux attach -t ${SESSION}"
echo "Detach from inside:     Ctrl-B then D"
echo "Kill the run:           tmux kill-session -t ${SESSION}"
