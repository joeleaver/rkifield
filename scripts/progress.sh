#!/usr/bin/env bash
# progress.sh — RKIField implementation progress tracker
# Parses git log for phase-N: N.X commit prefixes and cross-references
# against IMPLEMENTATION_PLAN.md to show per-phase completion status.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLAN="$REPO_ROOT/docs/IMPLEMENTATION_PLAN.md"

if [ ! -f "$PLAN" ]; then
    echo "Error: IMPLEMENTATION_PLAN.md not found at $PLAN" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse phase names from ## Phase N: Name headings
# ---------------------------------------------------------------------------
declare -A PHASE_NAME
while IFS= read -r line; do
    # Match: ## Phase N: Name  (N can be multi-digit)
    if [[ "$line" =~ ^##\ Phase\ ([0-9]+):\ (.+)$ ]]; then
        n="${BASH_REMATCH[1]}"
        name="${BASH_REMATCH[2]}"
        PHASE_NAME[$n]="$name"
    fi
done < "$PLAN"

# ---------------------------------------------------------------------------
# Parse task counts per phase from lines matching "^N.X. **" pattern
# ---------------------------------------------------------------------------
declare -A PHASE_TOTAL
while IFS= read -r line; do
    # Match lines like: "0.1. **Create Cargo workspace**" or "17.10. **..."
    if [[ "$line" =~ ^([0-9]+)\.([0-9]+)\.\ \*\* ]]; then
        phase="${BASH_REMATCH[1]}"
        PHASE_TOTAL[$phase]=$(( ${PHASE_TOTAL[$phase]:-0} + 1 ))
    fi
done < "$PLAN"

# ---------------------------------------------------------------------------
# Parse completed tasks from git log
# ---------------------------------------------------------------------------
declare -A PHASE_DONE
declare -A DONE_TASKS  # track unique task IDs to avoid double-counting

# Check if we're in a git repo
if ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "Warning: Not a git repository — showing 0 completed tasks." >&2
else
    while IFS= read -r commit_msg; do
        # Match: phase-N: N.X (anywhere in the commit message)
        # Capture all occurrences on the line
        remainder="$commit_msg"
        while [[ "$remainder" =~ phase-([0-9]+):\ ([0-9]+)\.([0-9]+) ]]; do
            phase="${BASH_REMATCH[1]}"
            task_phase="${BASH_REMATCH[2]}"
            task_num="${BASH_REMATCH[3]}"
            task_id="${task_phase}.${task_num}"

            # Only count if phase numbers agree and task is not already counted
            if [ "$phase" = "$task_phase" ] && [ -z "${DONE_TASKS[$task_id]+x}" ]; then
                DONE_TASKS[$task_id]=1
                PHASE_DONE[$phase]=$(( ${PHASE_DONE[$phase]:-0} + 1 ))
            fi

            # Advance past this match to find more in the same line
            remainder="${remainder#*"${BASH_REMATCH[0]}"}"
        done
    done < <(git -C "$REPO_ROOT" log --format="%B" 2>/dev/null)
fi

# ---------------------------------------------------------------------------
# Determine the full list of phases (union of plan + git, sorted)
# ---------------------------------------------------------------------------
ALL_PHASES=()
declare -A SEEN_PHASE
for k in "${!PHASE_TOTAL[@]}" "${!PHASE_DONE[@]}"; do
    if [ -z "${SEEN_PHASE[$k]+x}" ]; then
        SEEN_PHASE[$k]=1
        ALL_PHASES+=("$k")
    fi
done

# Sort numerically
IFS=$'\n' ALL_PHASES=($(printf '%s\n' "${ALL_PHASES[@]}" | sort -n))
unset IFS

# ---------------------------------------------------------------------------
# Build progress bar: [====····] style, width 8 characters
# ---------------------------------------------------------------------------
make_bar() {
    local done=$1
    local total=$2
    local width=8
    local filled=0

    if [ "$total" -gt 0 ]; then
        # Integer arithmetic: filled = done * width / total (rounded down)
        filled=$(( done * width / total ))
    fi
    local empty=$(( width - filled ))

    local bar=""
    local i
    for (( i=0; i<filled; i++ )); do bar="${bar}="; done
    for (( i=0; i<empty; i++ )); do bar="${bar}·"; done
    printf "[%s]" "$bar"
}

# ---------------------------------------------------------------------------
# Print output
# ---------------------------------------------------------------------------
echo "RKIField Implementation Progress"
echo "================================"

total_done=0
total_tasks=0

for phase in "${ALL_PHASES[@]}"; do
    done=${PHASE_DONE[$phase]:-0}
    total=${PHASE_TOTAL[$phase]:-0}
    name=${PHASE_NAME[$phase]:-"(unknown)"}
    bar=$(make_bar "$done" "$total")

    total_done=$(( total_done + done ))
    total_tasks=$(( total_tasks + total ))

    printf "Phase %2d (%-24s): %3d/%-3d tasks  %s\n" \
        "$phase" "$name" "$done" "$total" "$bar"
done

echo ""
echo "--------------------------------"
total_bar=$(make_bar "$total_done" "$total_tasks")
printf "Total:                             %3d/%-3d tasks  %s\n" \
    "$total_done" "$total_tasks" "$total_bar"
