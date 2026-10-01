#!/usr/bin/env bash
# Scores one count from the `[GRAPH-DIAG] phase=final` line that
# ggml_backend_sycl_free prints when GGML_SYCL_GRAPH_DIAG is set.
#
# The scored line is the LAST final line whose `calls=` is >= 1. A temporary
# backend that a model load creates and frees prints a final line with every
# counter at 0 and is never scored. A log with no qualifying line, or a key
# that does not match exactly once, is VOID (exit status 3), never a 0.
#
#   bash scripts/sycl-graph-diag-score.sh <log> <key>     # prints the value
#   source scripts/sycl-graph-diag-score.sh               # diag_score <log> <key>
#
# Each key is matched with a left anchor (line start or a space), so neither a
# longer key that ends with the same name nor the frame counter's `pp=`/`tg=`
# is taken for it.

diag_final() {  # the scored line; exit status 3 when there is none
    awk '/^\[GRAPH-DIAG\] phase=final / {
           if (match($0, / calls=[0-9]+ /) && substr($0, RSTART + 7, RLENGTH - 8) + 0 >= 1) last = $0
         }
         END { if (last == "") exit 3; print last }' "$1"
}

diag_key() {    # diag_key "<line>" <key>: every match, one per line
    # Whole whitespace-delimited tokens: a regex that consumes the trailing
    # space would skip an adjacent repeat ("k=1 k=2") and report one match.
    printf '%s\n' "$1" | awk -v key="$2" '{ for (i = 1; i <= NF; i++) if (index($i, key "=") == 1 && substr($i, length(key) + 2) ~ /^[0-9]+$/) print substr($i, length(key) + 2) }'
}

diag_score() {  # diag_score <log> <key>: the value, or VOID with exit status 3
    local L v
    L=$(diag_final "$1") || { echo VOID; return 3; }
    v=$(diag_key "$L" "$2")
    [ "$(printf '%s' "$v" | grep -c .)" -eq 1 ] || { echo VOID; return 3; }
    printf '%s\n' "$v"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then diag_score "$@"; exit $?; fi
