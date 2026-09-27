#!/usr/bin/env bash
# The -j1 OOM safety net (CLAUDE.md "Running Tests") has two independent
# failure modes, and both ctest calls below are treated SYMMETRICALLY:
# capture the listing, check ctest's own exit status BEFORE trusting any grep
# count (a failed/truncated listing must not read as "found nothing", which
# for an ABSENCE check like leak-counting is indistinguishable from
# "genuinely clean"), and require the listing to be non-empty before scoring
# it (a -L or -LE that silently selects nothing must not certify an empty
# set as satisfying the query).
#
# Half 1 (label net) catches label-stripping: a merge to
# tests/CMakeLists.txt can silently drop the fork-added `cache|mem-handle`
# labels (registration is upstream code), which makes -L select nothing and
# fails OPEN.
#
# Half 2 (sweep) is NOT label-dependent -- it catches the
# `-E '^test-backend-ops$'` anchor going stale if that binary is ever renamed,
# via a deliberately UNANCHORED substring grep against the filtered listing
# (style: check-merge-source-coverage.sh:30-31).
#
# Half 3 (model loaders) derives the set of model-loading tests from the
# registration, not from a hand list. A loader is:
#   - any test that REQUIRES a fixture (all of this tree's fixtures provide a
#     generated or downloaded model);
#   - any fixture SETUP test that runs a binary rather than cmake (the
#     generator, test-llama-archs -o, loads each arch it writes);
#   - any test labelled `model`, upstream's label for tests that load a model
#     named by the environment (LLAMACPP_TEST_MODELFILE), which the fork also
#     puts on sycl-lifecycle-gpu-sequential (models named by its G1 fixture).
#     Labels are matched the way ctest -LE matches them, a regex search per
#     label, with MODEL_LABEL_RE, the same string the sweep's -LE carries. It
#     is anchored because -LE is not: a bare `model` also drops `cross-model`
#     and `model-load`, which are not loaders.
# It then fails naming each one the documented sweep still selects. Fixture
# names are never consulted, so a new or renamed upstream fixture is covered
# without editing this file. A test that loads a model but has no fixture and
# no `model` label is invisible to this half; label it rather than widen the
# guess.
#
# Half 4 (over-exclusion) takes ctest's own -LE-only listing and fails naming
# each test the label exclusion drops that is neither a model loader nor in the
# label net (form 3 runs those). A widened or unanchored -LE silently removes
# tests from the sweep, and nothing else would notice.
#
# SWEEP_LE_DEFAULT and SWEEP_E_DEFAULT are the exclusions CLAUDE.md documents.
# The guard also checks that CLAUDE.md carries both, together, in the two
# places the sweep is written out (Running Tests form 2 and Before Submitting
# PRs step 3), so the checked sweep and the documented one cannot drift apart.
# --sweep-label-exclude and --sweep-exclude override them for a RED run, which
# skips the doc check.
set -euo pipefail
MODEL_LABEL_RE='^model$'
SWEEP_LE_DEFAULT="residency|mem-handle|cache|$MODEL_LABEL_RE"
SWEEP_E_DEFAULT='^(test-backend-ops|test-generate-models|test-recurrent-state-rollback.*|test-save-load-state|test-thread-safety|test-sycl-model-lifecycle-hooks|test-state-restore-fragmented|test-eval-callback)$'
BUILD="build" CTEST="ctest" SWEEP_LE="$SWEEP_LE_DEFAULT" SWEEP_E="$SWEEP_E_DEFAULT"
CLAUDE_MD="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/CLAUDE.md"
while [ $# -gt 0 ]; do case "$1" in
    --build-dir) [ $# -ge 2 ] || { echo "check-ctest-safety-net: --build-dir needs a value" >&2; exit 2; }
                 BUILD="$2"; shift 2;;
    --ctest-cmd) [ $# -ge 2 ] || { echo "check-ctest-safety-net: --ctest-cmd needs a value" >&2; exit 2; }
                 CTEST="$2"; shift 2;;
    --sweep-label-exclude) [ $# -ge 2 ] || { echo "check-ctest-safety-net: --sweep-label-exclude needs a value" >&2; exit 2; }
                 SWEEP_LE="$2"; shift 2;;
    --sweep-exclude) [ $# -ge 2 ] || { echo "check-ctest-safety-net: --sweep-exclude needs a value" >&2; exit 2; }
                 SWEEP_E="$2"; shift 2;;
    --claude-md) [ $# -ge 2 ] || { echo "check-ctest-safety-net: --claude-md needs a value" >&2; exit 2; }
                 CLAUDE_MD="$2"; shift 2;;
    *) echo "check-ctest-safety-net: unknown arg $1" >&2; exit 2;;
esac; done
command -v "$CTEST" >/dev/null || { echo "check-ctest-safety-net: ctest command not found: $CTEST" >&2; exit 2; }
[ -d "$BUILD" ] || { echo "MISSING BUILD DIR: $BUILD -- refusing to pass vacuously" >&2; exit 2; }

label_out=$("$CTEST" --test-dir "$BUILD" -N -L 'cache|mem-handle') \
    || { rc=$?; echo "LABEL LISTING FAILED: ctest exited $rc -- refusing to pass vacuously" >&2; exit 2; }
sel=$(printf '%s\n' "$label_out" | grep -c '^  Test' || true)
if [ "$sel" -lt 1 ]; then echo "LABEL NET EMPTY: -L 'cache|mem-handle' selects $sel tests"; exit 1; fi

# The leak count below is a count of an UNWANTED pattern: on a failed/truncated
# listing, empty stdout would grep to zero matches, i.e. "no leak" -- a false
# PASS on a query that never actually ran. So, unlike sel above, ctest's own
# exit status must be checked BEFORE trusting the grep count.
sweep_out=$("$CTEST" --test-dir "$BUILD" -N -LE "$SWEEP_LE" -E "$SWEEP_E") \
    || { rc=$?; echo "SWEEP LISTING FAILED: ctest exited $rc -- refusing to pass vacuously" >&2; exit 2; }
# Same non-vacuity concern as the label half: an -LE that silently selects
# nothing at all (e.g. a mangled exclusion expression matching everything)
# would leak-count to zero and read as "clean", when in truth no query ran.
sweep_count=$(printf '%s\n' "$sweep_out" | grep -c '^  Test' || true)
if [ "$sweep_count" -lt 1 ]; then echo "SWEEP LISTING EMPTY -- refusing to pass vacuously" >&2; exit 2; fi
leak=$(printf '%s\n' "$sweep_out" | grep -c 'backend-ops' || true)
if [ "$leak" -ne 0 ]; then echo "SWEEP LEAK: filtered sweep still lists backend-ops ($leak lines)"; exit 1; fi

registry_json=$("$CTEST" --test-dir "$BUILD" -N --show-only=json-v1) \
    || { rc=$?; echo "REGISTRY LISTING FAILED: ctest exited $rc -- refusing to pass vacuously" >&2; exit 2; }
sweep_json=$("$CTEST" --test-dir "$BUILD" -N --show-only=json-v1 -LE "$SWEEP_LE" -E "$SWEEP_E") \
    || { rc=$?; echo "SWEEP JSON LISTING FAILED: ctest exited $rc -- refusing to pass vacuously" >&2; exit 2; }
label_kept_json=$("$CTEST" --test-dir "$BUILD" -N --show-only=json-v1 -LE "$SWEEP_LE") \
    || { rc=$?; echo "LABEL-KEPT JSON LISTING FAILED: ctest exited $rc -- refusing to pass vacuously" >&2; exit 2; }
label_net=$(printf '%s\n' "$label_out" | sed -n 's/^  Test *#[0-9]*: //p')
# Exit codes follow the halves above: 1 = the net has a hole, 2 = the query
# could not be trusted. The registry, sweep and -LE-only listings arrive on
# fds 3, 4 and 5, and the label net's test names on fd 6.
loaders=$(python3 -c '
import json, os, re, sys

MODEL_LABEL_RE = sys.argv[1]

def tests(fd, what):
    try:
        with os.fdopen(fd) as f:
            return json.load(f)["tests"]
    except (ValueError, KeyError, TypeError) as e:
        print(f"{what} LISTING UNREADABLE: {e} -- refusing to pass vacuously", file=sys.stderr)
        sys.exit(2)

def props(t):
    return {p["name"]: p["value"] for p in t.get("properties", [])}

def loader_reason(t):
    p = props(t)
    if p.get("FIXTURES_REQUIRED"):
        return "requires fixture " + ",".join(p["FIXTURES_REQUIRED"])
    cmd = t.get("command") or [""]
    if p.get("FIXTURES_SETUP") and os.path.basename(cmd[0]) != "cmake":
        return "sets up fixture " + ",".join(p["FIXTURES_SETUP"]) + " by running " + os.path.basename(cmd[0])
    if any(re.search(MODEL_LABEL_RE, label) for label in p.get("LABELS") or []):
        return "carries label model"
    return None

registry   = tests(3, "REGISTRY")
sweep      = tests(4, "SWEEP")
label_kept = tests(5, "LABEL-KEPT")
with os.fdopen(6) as f:
    label_net = set(f.read().split())
loaders  = {t["name"]: loader_reason(t) for t in registry if loader_reason(t)}
if not loaders:
    print("MODEL-LOADER SET EMPTY: no registered test requires or generates a fixture or carries label model"
          " -- refusing to pass vacuously",
          file=sys.stderr)
    sys.exit(2)
if not sweep:
    print("SWEEP JSON LISTING EMPTY -- refusing to pass vacuously", file=sys.stderr)
    sys.exit(2)
hits = [(t["name"], loaders.get(t["name"]) or loader_reason(t)) for t in sweep
        if t["name"] in loaders or loader_reason(t)]
for name, why in hits:
    print(f"SWEEP RUNS MODEL LOADER: {name} ({why})")
if hits:
    sys.exit(1)
if not label_kept:
    print("LABEL-KEPT JSON LISTING EMPTY -- refusing to pass vacuously", file=sys.stderr)
    sys.exit(2)
kept    = {t["name"] for t in label_kept}
dropped = [t for t in registry if t["name"] not in kept and t["name"] not in loaders and t["name"] not in label_net]
for t in dropped:
    labels = ",".join(props(t).get("LABELS") or [])
    print("SWEEP DROPS NON-LOADER: " + t["name"] + " (labels " + labels + ")")
if dropped:
    sys.exit(1)
print(len(loaders))
' "$MODEL_LABEL_RE" 3<<<"$registry_json" 4<<<"$sweep_json" 5<<<"$label_kept_json" 6<<<"$label_net") \
    || { rc=$?; [ -n "$loaders" ] && printf '%s\n' "$loaders"; exit "$rc"; }

if [ "$SWEEP_LE" = "$SWEEP_LE_DEFAULT" ] && [ "$SWEEP_E" = "$SWEEP_E_DEFAULT" ]; then
    [ -f "$CLAUDE_MD" ] || { echo "MISSING CLAUDE.md: $CLAUDE_MD -- refusing to pass vacuously" >&2; exit 2; }
    sweep_args="-LE '$SWEEP_LE' -E '$SWEEP_E'"
    documented=$(grep -cF -- "$sweep_args" "$CLAUDE_MD" || true)
    if [ "$documented" -lt 2 ]; then
        echo "DOC DRIFT: $CLAUDE_MD carries $sweep_args $documented time(s); want it in form 2 and PR step 3"
        exit 1
    fi
fi

echo "safety net intact: $sel labelled tests; sweep excludes backend-ops and all $loaders model loaders"
