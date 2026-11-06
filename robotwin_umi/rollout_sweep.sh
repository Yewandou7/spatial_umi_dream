#!/bin/bash
# Evaluate checkpoints across tasks with one policy server per SERVER_GPUS entry.
# CKPT_DIR=<run>/checkpoint-<step> evaluates one checkpoint; alternatively OUTPUT_DIR (a training
# output dir) with STEPS selecting its checkpoints (default: all of them).
# TASK_NAME or comma-separated TASK_NAMES (default: the 20 benchmark tasks).
# TEST_NUM is the episode count per task (default 100).
# SERVER_GPUS defaults to 0; SIM_GPUS defaults to 1,2,3.
# PORTS defaults to consecutive ports starting at PORT (8765).
# The RoboTwin-UMI client renders the wrist depth streams (--depth true).
# TOKENIZER_PATH overrides the checkpoint's tokenizer path.
# POLICY_PYTHON and ROBOTWIN_UMI_PYTHON select the two environments.
# Logs and task/checkpoint CSV summaries are written under EVAL_ROOT.

set -euo pipefail

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROBOTWIN_UMI_ROOT=${ROBOTWIN_UMI_ROOT:-}
# Policy folder name inside the RoboTwin-UMI checkout (policy/<name>/); also the eval_result/ subdir.
ROBOTWIN_UMI_POLICY_NAME=${ROBOTWIN_UMI_POLICY_NAME:-SUD}

CKPT_DIR=${CKPT_DIR:-}
if [ -n "$CKPT_DIR" ]; then
    CKPT_DIR=${CKPT_DIR%/}
    if [[ "$(basename "$CKPT_DIR")" != checkpoint-* ]]; then
        echo "ERROR: CKPT_DIR must be a checkpoint-<step> directory: $CKPT_DIR"
        exit 1
    fi
    OUTPUT_DIR=$(dirname "$CKPT_DIR")
    STEPS=$(basename "$CKPT_DIR")
    STEPS=${STEPS#checkpoint-}
fi
OUTPUT_DIR=${OUTPUT_DIR:-}
TASK_NAME=${TASK_NAME:-}
TASK_NAMES=${TASK_NAMES:-}
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: set CKPT_DIR to a checkpoint-<step> directory, or OUTPUT_DIR to a training output dir."
    exit 1
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: OUTPUT_DIR not found: $OUTPUT_DIR"
    exit 1
fi
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
if [ -z "$ROBOTWIN_UMI_ROOT" ]; then
    echo "ERROR: set ROBOTWIN_UMI_ROOT to the RoboTwin-UMI checkout."
    exit 1
fi
if [ ! -d "$ROBOTWIN_UMI_ROOT" ]; then
    echo "ERROR: ROBOTWIN_UMI_ROOT not found: $ROBOTWIN_UMI_ROOT"
    exit 1
fi
ROBOTWIN_UMI_ROOT=$(cd "$ROBOTWIN_UMI_ROOT" && pwd)

TASK_LIST=()
if [ -n "$TASK_NAMES" ]; then
    IFS=',' read -r -a RAW_TASK_LIST <<< "$TASK_NAMES"
    for task in "${RAW_TASK_LIST[@]}"; do
        task="${task//[[:space:]]/}"
        if [ -n "$task" ]; then
            TASK_LIST+=("$task")
        fi
    done
elif [ -n "$TASK_NAME" ]; then
    TASK_LIST=("$TASK_NAME")
else
    # The 20 RoboTwin-UMI benchmark tasks.
    TASK_LIST=(adjust_bottle blocks_ranking_rgb click_alarmclock click_bell grab_roller lift_pot
        move_playingcard_away open_laptop place_a2b_left place_a2b_right place_bread_basket
        place_container_plate place_empty_cup place_object_scale place_object_stand press_stapler
        shake_bottle stack_blocks_two stack_bowls_two turn_switch)
fi
if [ "${#TASK_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no tasks parsed from TASK_NAMES=$TASK_NAMES"
    exit 1
fi
for task in "${TASK_LIST[@]}"; do
    if [ ! -f "$ROBOTWIN_UMI_ROOT/envs/${task}.py" ]; then
        echo "ERROR: RoboTwin-UMI task env not found: $task"
        echo "       Expected file: $ROBOTWIN_UMI_ROOT/envs/${task}.py"
        exit 1
    fi
done

TASK_CONFIG=${TASK_CONFIG:-demo_clean_aloha_umi_fisheye_depth}
TEST_NUM=${TEST_NUM:-100}
SEED=${SEED:-0}
BASE_PORT=${PORT:-8765}
SERVER_GPUS=${SERVER_GPUS:-0}
SIM_GPUS=${SIM_GPUS:-1,2,3}
N_EXEC_STEPS=${N_EXEC_STEPS:-24}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
MODEL_EMBODIMENT_TAG=${MODEL_EMBODIMENT_TAG:-robotwin}
CKPT_SETTING_PREFIX=${CKPT_SETTING_PREFIX:-sud}
EVAL_ROOT=${EVAL_ROOT:-"$OUTPUT_DIR/sim_rollout"}
EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG:-true}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}
EVAL_RUN_TAG=${EVAL_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

POLICY_PYTHON=${POLICY_PYTHON:-"$REPO_ROOT/.venv/bin/python"}
ROBOTWIN_UMI_PYTHON=${ROBOTWIN_UMI_PYTHON:-"$ROBOTWIN_UMI_ROOT/.venv/bin/python"}

if [ ! -x "$POLICY_PYTHON" ]; then
    echo "ERROR: POLICY_PYTHON is not executable: $POLICY_PYTHON"
    exit 1
fi
if [ ! -x "$ROBOTWIN_UMI_PYTHON" ]; then
    echo "ERROR: ROBOTWIN_UMI_PYTHON is not executable: $ROBOTWIN_UMI_PYTHON"
    exit 1
fi

IFS=',' read -r -a SERVER_GPU_LIST <<< "$SERVER_GPUS"
IFS=',' read -r -a SIM_GPU_LIST <<< "$SIM_GPUS"
if [ "${#SERVER_GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: SERVER_GPUS is empty."
    exit 1
fi
if [ "${#SIM_GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: SIM_GPUS is empty."
    exit 1
fi

if [ -n "${PORTS:-}" ]; then
    IFS=',' read -r -a PORT_LIST <<< "$PORTS"
else
    PORT_LIST=()
    for ((i = 0; i < ${#SERVER_GPU_LIST[@]}; i++)); do
        PORT_LIST+=("$((BASE_PORT + i))")
    done
fi
if [ "${#PORT_LIST[@]}" -ne "${#SERVER_GPU_LIST[@]}" ]; then
    echo "ERROR: PORTS count (${#PORT_LIST[@]}) must match SERVER_GPUS count (${#SERVER_GPU_LIST[@]})."
    exit 1
fi

if [ -n "${STEPS:-}" ]; then
    IFS=',' read -r -a STEP_LIST <<< "$STEPS"
else
    mapfile -t STEP_LIST < <(
        find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
            | sed 's/^checkpoint-//' \
            | sort -n
    )
fi
if [ "${#STEP_LIST[@]}" -eq 0 ]; then
    echo "ERROR: No checkpoint-* directories found under $OUTPUT_DIR"
    exit 1
fi

mkdir -p "$EVAL_ROOT"
# Absolute, since the server and simulator workers run from other directories.
EVAL_ROOT=$(cd "$EVAL_ROOT" && pwd)
SUMMARY="$EVAL_ROOT/summary.csv"
CHECKPOINT_SUMMARY="$EVAL_ROOT/checkpoint_summary.csv"
echo "step,task,status,success_rate,total_success,total_tests,eval_result_dir,server_logs,client_logs" > "$SUMMARY"
echo "step,status,macro_success_rate,micro_success_rate,total_success,total_tests,num_tasks,num_ok_tasks" > "$CHECKPOINT_SUMMARY"

server_pids=()
cleanup_servers() {
    for pid in "${server_pids[@]:-}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${server_pids[@]:-}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${server_pids[@]:-}"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    server_pids=()
}
trap cleanup_servers EXIT

wait_for_port() {
    "$POLICY_PYTHON" - "$1" "$2" "${3:-0}" <<'PYPORT'
import socket
import sys
import time

host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.monotonic() + timeout
while True:
    try:
        with socket.create_connection((host, port), timeout=2 if timeout else 1):
            sys.exit(0)
    except OSError:
        if time.monotonic() >= deadline:
            sys.exit(1)
        time.sleep(2)
PYPORT
}

reset_server() {
    local host=$1
    local port=$2
    "$POLICY_PYTHON" - "$host" "$port" <<'PY'
import http.client
import sys

host, port = sys.argv[1], int(sys.argv[2])
conn = http.client.HTTPConnection(host, port, timeout=10)
try:
    conn.request("POST", "/reset", body=b"{}",
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status} {resp.reason} {body[:200]!r}")
except Exception as exc:
    sys.stderr.write(f"reset failed on {host}:{port}: {exc}\n")
    sys.exit(1)
finally:
    conn.close()
PY
}

aggregate_results() {
    local run_dir=$1
    shift
    "$POLICY_PYTHON" - "$run_dir" "$@" <<'PY'
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
run_dir.mkdir(parents=True, exist_ok=True)
worker_test_nums = [int(x) for x in sys.argv[2:]]
total_success = 0
total_tests = 0
lines = []

for worker_idx, worker_tests in enumerate(worker_test_nums):
    if worker_tests <= 0:
        continue
    result_path = run_dir / f"_result_worker_{worker_idx}.txt"
    if not result_path.exists():
        lines.append(f"worker_{worker_idx}: missing result file")
        continue
    values = []
    for line in result_path.read_text().splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            pass
    if not values:
        lines.append(f"worker_{worker_idx}: missing success rate")
        continue
    rate = values[-1]
    success = int(round(rate * worker_tests))
    total_success += success
    total_tests += worker_tests
    lines.append(f"worker_{worker_idx}: {success}/{worker_tests} = {rate:.6f}")

overall = (total_success / total_tests) if total_tests else 0.0
(run_dir / "_result.txt").write_text(
    "\n".join([
        "Parallel eval summary",
        "",
        *lines,
        "",
        f"total: {total_success}/{total_tests}",
        f"{overall}",
    ]) + "\n"
)
print(f"{overall},{total_success},{total_tests}")
PY
}

echo "RoboTwin-UMI closed-loop simulator rollout sweep:"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  ROBOTWIN_UMI_ROOT=$ROBOTWIN_UMI_ROOT"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  TASKS=${TASK_LIST[*]} TASK_CONFIG=$TASK_CONFIG TEST_NUM_PER_TASK=$TEST_NUM SEED=$SEED"
echo "  MODEL_EMBODIMENT_TAG=$MODEL_EMBODIMENT_TAG (execution_interface=robotwin)"
echo "  SERVER_GPUS=$SERVER_GPUS PORTS=${PORT_LIST[*]} N_EXEC_STEPS=$N_EXEC_STEPS"
echo "  SIM_GPUS=$SIM_GPUS"
echo "  steps: ${STEP_LIST[*]}"
echo "  EVAL_RUN_TAG=$EVAL_RUN_TAG"
echo "  fixed eval seeds: disabled"

for STEP in "${STEP_LIST[@]}"; do
    MODEL_PATH="$OUTPUT_DIR/checkpoint-$STEP"
    STEP_DIR="$EVAL_ROOT/checkpoint-$STEP"
    CKPT_SETTING="${CKPT_SETTING_PREFIX}_checkpoint-${STEP}"
    EVAL_RUN_NAME="sweep_${EVAL_RUN_TAG}_checkpoint-${STEP}"

    mkdir -p "$STEP_DIR"
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "WARN: skip invalid checkpoint: $MODEL_PATH"
        for TASK in "${TASK_LIST[@]}"; do
            echo "$STEP,$TASK,invalid,,,,,," >> "$SUMMARY"
        done
        echo "$STEP,invalid,,,,,${#TASK_LIST[@]},0" >> "$CHECKPOINT_SUMMARY"
        continue
    fi

    echo
    echo "===== checkpoint-$STEP ====="
    cleanup_servers

    busy_ports=()
    for port in "${PORT_LIST[@]}"; do
        if wait_for_port localhost "$port"; then
            busy_ports+=("http:$port")
        fi
    done
    if [ "${#busy_ports[@]}" -gt 0 ]; then
        echo "ERROR: server port(s) already in use before starting checkpoint-$STEP: ${busy_ports[*]}"
        echo "       Refusing to continue because clients could hit a stale checkpoint server."
        for TASK in "${TASK_LIST[@]}"; do
            echo "$STEP,$TASK,port_busy,,,,,$(IFS=';'; echo "${busy_ports[*]}")," >> "$SUMMARY"
        done
        echo "$STEP,port_busy,,,,,${#TASK_LIST[@]},0" >> "$CHECKPOINT_SUMMARY"
        exit 1
    fi

    server_logs=()
    for ((server_idx = 0; server_idx < ${#SERVER_GPU_LIST[@]}; server_idx++)); do
        gpu_id="${SERVER_GPU_LIST[$server_idx]}"
        port="${PORT_LIST[$server_idx]}"
        server_log="$STEP_DIR/server_${server_idx}_port_${port}.log"
        server_logs+=("$server_log")
        echo "[server $server_idx] gpu=$gpu_id port=$port"
        (
            cd "$REPO_ROOT"
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            server_args=(
                --ckpt "$MODEL_PATH"
                --port "$port"
                --n_exec_steps "$N_EXEC_STEPS"
                --embodiment_tag "$MODEL_EMBODIMENT_TAG"
            )
            if [ -n "$TOKENIZER_PATH" ]; then
                server_args+=(--tokenizer_path "$TOKENIZER_PATH")
            fi
            exec "$POLICY_PYTHON" -u robotwin_umi/policy_server.py \
                "${server_args[@]}"
        ) >"$server_log" 2>&1 &
        server_pids+=("$!")
    done

    server_ready=1
    for port in "${PORT_LIST[@]}"; do
        if ! wait_for_port localhost "$port" "$SERVER_START_TIMEOUT"; then
            server_ready=0
            break
        fi
    done
    if [ "$server_ready" -ne 1 ]; then
        echo "WARN: at least one server did not become ready for checkpoint-$STEP"
        for TASK in "${TASK_LIST[@]}"; do
            echo "$STEP,$TASK,server_failed,,,,,$(IFS=';'; echo "${server_logs[*]}")," >> "$SUMMARY"
        done
        echo "$STEP,server_failed,,,,,${#TASK_LIST[@]},0" >> "$CHECKPOINT_SUMMARY"
        cleanup_servers
        continue
    fi

    checkpoint_stats=()
    for TASK in "${TASK_LIST[@]}"; do
        TASK_STEP_DIR="$STEP_DIR/$TASK"
        # eval_policy.py writes under eval_result/<task>/<policy_name>/...
        RESULT_DIR="$ROBOTWIN_UMI_ROOT/eval_result/$TASK/$ROBOTWIN_UMI_POLICY_NAME/$TASK_CONFIG/$CKPT_SETTING/$EVAL_RUN_NAME"
        mkdir -p "$TASK_STEP_DIR"
        rm -rf "$RESULT_DIR"

        echo
        echo "----- checkpoint-$STEP task=$TASK -----"
        for port in "${PORT_LIST[@]}"; do
            if ! reset_server localhost "$port"; then
                echo "WARN: failed to reset server on port $port before task=$TASK; continuing"
            fi
        done

        worker_count="${#SIM_GPU_LIST[@]}"
        if [ "$worker_count" -gt "$TEST_NUM" ]; then
            worker_count=$TEST_NUM
        fi
        pids=()
        worker_test_nums=()
        client_logs=()
        failed=0

        # All workers share SEED: eval_policy.py builds one pool of the first TEST_NUM expert-valid
        # seeds and worker i rolls out pool[i::worker_count], independent of the number of GPUs.
        for ((worker_idx = 0; worker_idx < worker_count; worker_idx++)); do
            worker_test_num=$(( (TEST_NUM - worker_idx + worker_count - 1) / worker_count ))
            worker_test_nums+=("$worker_test_num")

            sim_gpu="${SIM_GPU_LIST[$worker_idx]}"
            port="${PORT_LIST[$((worker_idx % ${#PORT_LIST[@]}))]}"
            result_file="_result_worker_${worker_idx}.txt"
            client_log="$TASK_STEP_DIR/client_worker_${worker_idx}_gpu_${sim_gpu}_port_${port}.log"
            client_logs+=("$client_log")

            echo "[task $TASK worker $worker_idx] sim_gpu=$sim_gpu port=$port seed=$SEED tests=$worker_test_num"
            (
                cd "$ROBOTWIN_UMI_ROOT"
                export PATH="$ROBOTWIN_UMI_ROOT/.venv/bin:$PATH"
                export CUDA_VISIBLE_DEVICES="$sim_gpu"
                export PYTHONWARNINGS=ignore::UserWarning
                export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-$(id -u)}
                mkdir -p "$XDG_RUNTIME_DIR"
                exec "$ROBOTWIN_UMI_PYTHON" -u script/eval_policy.py \
                    --config "policy/$ROBOTWIN_UMI_POLICY_NAME/deploy_policy.yml" \
                    --overrides \
                    --task_name "$TASK" \
                    --task_config "$TASK_CONFIG" \
                    --ckpt_setting "$CKPT_SETTING" \
                    --seed "$SEED" \
                    --policy_name "$ROBOTWIN_UMI_POLICY_NAME" \
                    --test_num "$TEST_NUM" \
                    --port "$port" \
                    --worker_idx "$worker_idx" \
                    --worker_num "$worker_count" \
                    --result_file "$result_file" \
                    --eval_run_name "$EVAL_RUN_NAME" \
                    --eval_video_log "$EVAL_VIDEO_LOG" \
                    --instruction_type "$INSTRUCTION_TYPE" \
                    --depth true
            ) >"$client_log" 2>&1 &
            pids+=("$!")
        done

        # Workers wait on each other while building the seed pool, so one failure stops them all.
        # (Poll instead of `wait -n`: the policy servers are background jobs of this shell too.)
        remaining=("${pids[@]}")
        while [ "${#remaining[@]}" -gt 0 ]; do
            alive=()
            for pid in "${remaining[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    alive+=("$pid")
                elif ! wait "$pid"; then
                    failed=1
                fi
            done
            remaining=("${alive[@]}")
            if [ "$failed" -ne 0 ]; then
                kill "${remaining[@]}" 2>/dev/null || true
                break
            fi
            if [ "${#remaining[@]}" -gt 0 ]; then
                sleep 5
            fi
        done

        if [ "$failed" -ne 0 ]; then
            echo "WARN: at least one simulator worker failed for checkpoint-$STEP task=$TASK"
            echo "$STEP,$TASK,client_failed,,,,,$(IFS=';'; echo "${server_logs[*]}"),$(IFS=';'; echo "${client_logs[*]}")" >> "$SUMMARY"
            continue
        fi

        aggregate_csv="$(aggregate_results "$RESULT_DIR" "${worker_test_nums[@]}")"
        IFS=',' read -r success_rate total_success total_tests <<< "$aggregate_csv"
        if [ "${total_tests:-0}" -eq 0 ]; then
            echo "WARN: no result files under $RESULT_DIR for checkpoint-$STEP task=$TASK"
            echo "$STEP,$TASK,no_results,,,,$RESULT_DIR,$(IFS=';'; echo "${server_logs[*]}"),$(IFS=';'; echo "${client_logs[*]}")" >> "$SUMMARY"
            continue
        fi
        echo "[result] checkpoint-$STEP task=$TASK success_rate=$success_rate total=$total_success/$total_tests"
        echo "$STEP,$TASK,ok,$success_rate,$total_success,$total_tests,$RESULT_DIR,$(IFS=';'; echo "${server_logs[*]}"),$(IFS=';'; echo "${client_logs[*]}")" >> "$SUMMARY"
        checkpoint_stats+=("$success_rate,$total_success,$total_tests")
    done

    cleanup_servers

    if [ "${#checkpoint_stats[@]}" -eq 0 ]; then
        echo "$STEP,no_ok_tasks,,,,,${#TASK_LIST[@]},0" >> "$CHECKPOINT_SUMMARY"
        continue
    fi

    checkpoint_csv="$("$POLICY_PYTHON" - "${checkpoint_stats[@]}" <<'PY'
import sys

rates = []
total_success = 0
total_tests = 0
for item in sys.argv[1:]:
    rate_s, success_s, tests_s = item.split(",")
    rates.append(float(rate_s))
    total_success += int(success_s)
    total_tests += int(tests_s)

macro = sum(rates) / len(rates)
micro = total_success / total_tests if total_tests else 0.0
print(f"{macro},{micro},{total_success},{total_tests},{len(rates)}")
PY
)"
    IFS=',' read -r macro_success_rate micro_success_rate checkpoint_total_success checkpoint_total_tests checkpoint_ok_tasks <<< "$checkpoint_csv"
    echo "[checkpoint result] checkpoint-$STEP macro=$macro_success_rate micro=$micro_success_rate total=$checkpoint_total_success/$checkpoint_total_tests ok_tasks=$checkpoint_ok_tasks/${#TASK_LIST[@]}"
    echo "$STEP,ok,$macro_success_rate,$micro_success_rate,$checkpoint_total_success,$checkpoint_total_tests,${#TASK_LIST[@]},$checkpoint_ok_tasks" >> "$CHECKPOINT_SUMMARY"
done

cleanup_servers
echo
echo "Sweep finished. Summary: $SUMMARY"
echo "Checkpoint summary: $CHECKPOINT_SUMMARY"
