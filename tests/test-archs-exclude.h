#pragma once

// -x/--exclude for test-llama-archs (llama.cpp-fepd): the excluded-arch set, the parser that
// validates a name into it, and the table row an excluded architecture gets.
//
// Why this is a header and not three functions inside test-llama-archs.cpp: the sweep it serves
// cannot be run to check it. test-llama-archs loads models onto a GPU, so it is on this
// project's never-loop list and a single run peaks near 200 GB of 255 GB -- which means the
// only affordable gate is one that drives this logic WITHOUT the sweep. tests/
// test-archs-exclude.cpp is that gate, and it calls the definitions below rather than
// restating them, exactly as test-archs-table.cpp gates the emitter through
// test-archs-table.h. A gate carrying its own copy of the rule stops testing the harness the
// moment either copy drifts.
//
// What the exclusion exists for: every full sweep dies at gemma2 on a 30 s SYCL watchdog
// (llama.cpp-4lnb), so the ~30 MoE architectures declared after LLM_ARCH_GEMMA2 have never
// been measured by anyone, in any run. `-x gemma2` gets past it without pretending the hang is
// fixed.

#include "../src/llama-arch.h"
#include "test-archs-table.h"

#include <algorithm>
#include <string>
#include <vector>

struct archs_exclude {
    // The status text an excluded row carries in BOTH result columns.
    //
    // The word matters more than the colour. Output from this harness is read by grep at least
    // as often as by eye -- `grep -c FAIL`, `grep -c SKIP` -- so the requirement that EXCLUDED
    // be "distinguishable from SKIP and from OK" is really the requirement that no status word
    // be a substring of another, and that is what test-archs-exclude.cpp gates. Colour
    // distinctness is by construction: the table's palette is green=OK, red=FAIL, yellow=SKIP,
    // cyan=BITDIFF, so magenta (35) is the one bright code still free. That is deliberately NOT
    // gated -- asserting it would need a second copy of the palette here, and a stale copy
    // would pass vacuously, which is worse than a property a reviewer can see.
    //
    // An excluded arch must never be merely ABSENT from the table. An absent row is
    // indistinguishable from a row that was never reached, which is the failure mode this
    // whole ticket exists inside: a sweep that died early and a sweep that skipped on purpose
    // produce the same silence, and only one of them means the architecture is fine.
    static constexpr const char * status = "\033[1;35mEXCLUDED\033[0m";

    // Resolve one -x/--exclude argument and append it. Returns false if the name is not a
    // usable architecture, and leaves `archs` untouched when it does -- a rejected name that
    // still grew the set would be a second way to be wrong.
    //
    // "Not usable" folds together two cases on purpose, because llm_arch_from_string() cannot
    // tell them apart: a name that matches nothing, and the literal "(unknown)", which is a
    // real entry in LLM_ARCH_NAMES and maps to LLM_ARCH_UNKNOWN. The sweep skips
    // LLM_ARCH_UNKNOWN before it reaches the exclusion check, so accepting either would be a
    // silent no-op -- an exclude that excludes nothing, letting a run claim coverage of an
    // architecture it never reached.
    static bool parse(const char * name, std::vector<llm_arch> & archs) {
        const llm_arch arch = llm_arch_from_string(name);
        if (arch == LLM_ARCH_UNKNOWN) {
            return false;
        }
        archs.push_back(arch);
        return true;
    }

    // Repeats are harmless and deliberately not de-duplicated: membership is what the sweep
    // asks about, and `-x llama -x llama` means the same thing as `-x llama`.
    static bool contains(const std::vector<llm_arch> & archs, const llm_arch arch) {
        return std::find(archs.begin(), archs.end(), arch) != archs.end();
    }

    // ONE row per excluded architecture, not one per (device, config) cell.
    //
    // The exclusion is applied at architecture granularity -- `-x <arch>` -- and it takes
    // effect before any fixture is built, so no device and no MoE/Dense configuration was ever
    // selected for this arch. Filling those columns with real device names would describe work
    // that did not happen; '-' says the row is about the architecture as a whole.
    static std::string row(const archs_table & table, const char * arch) {
        return table.row(arch, "-", "-", status, "", status);
    }
};
