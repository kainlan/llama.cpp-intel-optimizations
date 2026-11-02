// H5r (zhcn C7f): the fused-op resolution report. Pure host: it links only
// src/llama-fused-resolution.h, so no model, no device and no scheduler are involved.
//
// Cases:
//   (1) empty: an empty record makes no sink call and sets no mark;
//   (2) partial: a record holding the entries before a throw prints and marks exactly those, and a
//       later, fuller record prints only the entries still unmarked;
//   (3) the same resolution reported twice prints once;
//   (4) a changed resolution reprints once, every line marked, and the new resolution is then silent;
//   (5) A, B, B prints A, then B marked, then nothing;
//   (6) the lines of each kind, at upstream's text and levels, in id order;
//   (7) the unwind guard: silent on a normal return, reports once while an exception leaves its
//       scope, and a report that throws is handed to the lost callback without a terminate.

#include "../src/llama-fused-resolution.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            n_failed++;                                                       \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                     \
            fprintf(stderr, "\n");                                            \
        }                                                                     \
    } while (0)

struct sunk_line {
    fused_resolution_level level;
    std::string            text;
};

static std::vector<sunk_line> g_sunk;

static void recording_sink(fused_resolution_level level, const char * text) {
    g_sunk.push_back({ level, text });
}

using marks_t = std::string[FUSED_RESOLUTION_N_ENTRIES];

static bool marks_clear(const marks_t & marks) {
    for (int i = 0; i < FUSED_RESOLUTION_N_ENTRIES; ++i) {
        if (!marks[i].empty()) {
            return false;
        }
    }
    return true;
}

static void set_header(fused_resolution & rec, fused_resolution_entry id, const char * text) {
    rec.entry[id].present = true;
    rec.entry[id].header  = text;
}

static void set_probe(fused_resolution & rec, fused_resolution_entry id, const char * name, bool enabled) {
    auto & e     = rec.entry[id];
    e            = {};
    e.present    = true;
    e.probe_name = name;
    e.enabled    = enabled;
}

static void emit(const fused_resolution & rec, marks_t & marks) {
    fused_resolution_emit(rec, marks, "resolve_fused_ops", recording_sink);
}

static void case_1_empty() {
    marks_t          marks;
    fused_resolution rec;
    g_sunk.clear();
    emit(rec, marks);
    CHECK(g_sunk.empty(), "an empty record made %zu sink calls", g_sunk.size());
    CHECK(marks_clear(marks), "an empty record set a mark");
}

static void case_2_partial() {
    marks_t          marks;
    fused_resolution rec;
    set_header(rec, FUSED_RESOLUTION_ENTRY_GDN_HEADER, "resolving fused Gated Delta Net support:");
    set_probe(rec, FUSED_RESOLUTION_ENTRY_GDN_AR, "fused Gated Delta Net (autoregressive)", true);
    // the chunked probe threw before it completed: absent

    g_sunk.clear();
    emit(rec, marks);
    CHECK(g_sunk.size() == 2, "the partial record printed %zu lines, want 2 (header, ar enabled)", g_sunk.size());
    CHECK(!marks[FUSED_RESOLUTION_ENTRY_GDN_HEADER].empty() && !marks[FUSED_RESOLUTION_ENTRY_GDN_AR].empty(),
          "the printed entries are not marked");
    CHECK(marks[FUSED_RESOLUTION_ENTRY_GDN_CH].empty() && marks[FUSED_RESOLUTION_ENTRY_FA].empty(),
          "an absent entry was marked");

    // the retry resolves further: only the entry still unmarked prints
    set_probe(rec, FUSED_RESOLUTION_ENTRY_GDN_CH, "fused Gated Delta Net (chunked)", true);
    g_sunk.clear();
    emit(rec, marks);
    CHECK(g_sunk.size() == 1, "the fuller record printed %zu lines, want only the chunked probe's", g_sunk.size());
    CHECK(g_sunk.size() == 1 && g_sunk[0].text == "resolve_fused_ops: fused Gated Delta Net (chunked) enabled",
          "wrong line for the chunked probe");
}

static void case_3_same_twice() {
    marks_t          marks;
    fused_resolution rec;
    set_probe(rec, FUSED_RESOLUTION_ENTRY_FA, "Flash Attention", true);
    g_sunk.clear();
    emit(rec, marks);
    emit(rec, marks);
    CHECK(g_sunk.size() == 1, "the same resolution printed %zu times, want once", g_sunk.size());
}

static bool all_marked(size_t from) {
    for (size_t i = from; i < g_sunk.size(); ++i) {
        const std::string & t = g_sunk[i].text;
        const std::string   m = FUSED_RESOLUTION_REPRINT_MARK;
        if (t.size() < m.size() || t.compare(t.size() - m.size(), m.size(), m) != 0) {
            return false;
        }
    }
    return true;
}

static void case_4_changed() {
    marks_t          marks;
    fused_resolution a;
    set_probe(a, FUSED_RESOLUTION_ENTRY_FA, "Flash Attention", true);
    fused_resolution b;
    set_probe(b, FUSED_RESOLUTION_ENTRY_FA, "Flash Attention", false);

    g_sunk.clear();
    emit(a, marks);
    CHECK(g_sunk.size() == 1 && all_marked(0) == false, "the first print is not a reprint");
    const size_t first = g_sunk.size();
    emit(b, marks);
    CHECK(g_sunk.size() == first + 1, "the changed resolution printed %zu lines, want 1", g_sunk.size() - first);
    CHECK(g_sunk.size() > first && all_marked(first), "the reprint is not marked");
    CHECK(g_sunk.size() > first && g_sunk.back().text.find("not supported, set to disabled") != std::string::npos,
          "the reprint is not the new resolution");
    emit(b, marks);
    CHECK(g_sunk.size() == first + 1, "the new resolution printed again");
}

static void case_5_a_b_b() {
    marks_t          marks;
    fused_resolution a;
    set_probe(a, FUSED_RESOLUTION_ENTRY_LID, "Lightning Indexer", true);
    fused_resolution b;
    set_probe(b, FUSED_RESOLUTION_ENTRY_LID, "Lightning Indexer", false);

    g_sunk.clear();
    emit(a, marks);
    emit(b, marks);
    emit(b, marks);
    CHECK(g_sunk.size() == 2, "A, B, B printed %zu lines, want A once and B once", g_sunk.size());
    CHECK(g_sunk.size() == 2 && !all_marked(0) && all_marked(1), "A is plain and B is marked");
}

static void case_6_text() {
    marks_t          marks;
    fused_resolution rec;

    set_probe(rec, FUSED_RESOLUTION_ENTRY_FA, "Flash Attention", false);
    rec.entry[FUSED_RESOLUTION_ENTRY_FA].mismatch           = true;
    rec.entry[FUSED_RESOLUTION_ENTRY_FA].mismatch_layer     = 3;
    rec.entry[FUSED_RESOLUTION_ENTRY_FA].mismatch_layer_dev = "SYCL0";
    rec.entry[FUSED_RESOLUTION_ENTRY_FA].mismatch_probe_dev = "SYCL1";

    set_header(rec, FUSED_RESOLUTION_ENTRY_GDN_HEADER, "resolving fused Gated Delta Net support:");
    set_probe(rec, FUSED_RESOLUTION_ENTRY_GDN_AR, "fused Gated Delta Net (autoregressive)", true);
    rec.entry[FUSED_RESOLUTION_ENTRY_GDN_AR].n_cpu_landings    = 2;
    rec.entry[FUSED_RESOLUTION_ENTRY_GDN_AR].cpu_landing_layer = 7;
    rec.entry[FUSED_RESOLUTION_ENTRY_GDN_AR].cpu_landing_dev   = "CPU";

    g_sunk.clear();
    emit(rec, marks);

    const std::vector<sunk_line> want = {
        { FUSED_RESOLUTION_LEVEL_WARN,
         "resolve_fused_ops: layer 3 is assigned to device SYCL0 but Flash Attention is assigned to device SYCL1 "
          "(usually due to missing support)"                                                               },
        { FUSED_RESOLUTION_LEVEL_WARN, "resolve_fused_ops: Flash Attention not supported, set to disabled" },
        { FUSED_RESOLUTION_LEVEL_INFO, "resolve_fused_ops: resolving fused Gated Delta Net support:"       },
        { FUSED_RESOLUTION_LEVEL_WARN,
         "resolve_fused_ops: fused Gated Delta Net (autoregressive) executes on CPU for 2 layer(s) (e.g. layer 7) -- "
          "the executor follows data placement, not a capability gap"                                      },
        { FUSED_RESOLUTION_LEVEL_INFO, "resolve_fused_ops: fused Gated Delta Net (autoregressive) enabled" },
    };
    CHECK(g_sunk.size() == want.size(), "printed %zu lines, want %zu", g_sunk.size(), want.size());
    for (size_t i = 0; i < want.size() && i < g_sunk.size(); ++i) {
        CHECK(g_sunk[i].level == want[i].level && g_sunk[i].text == want[i].text, "line %zu: got [%d] %s", i,
              (int) g_sunk[i].level, g_sunk[i].text.c_str());
    }
}

struct guard_probe {
    int  reported      = 0;
    int  lost          = 0;
    bool report_throws = false;
};

static void guard_report(void * arg) {
    auto * p = static_cast<guard_probe *>(arg);
    p->reported++;
    if (p->report_throws) {
        throw std::runtime_error("the report threw");
    }
}

static void guard_lost(void * arg) noexcept {
    static_cast<guard_probe *>(arg)->lost++;
}

static void case_7_guard() {
    // a normal return is silent
    {
        guard_probe p;
        {
            fused_resolution_unwind_guard guard(guard_report, guard_lost, &p);
        }
        CHECK(p.reported == 0 && p.lost == 0, "a normal return reported %d, lost %d", p.reported, p.lost);
    }

    // an exception leaving the scope reports once and still propagates
    {
        guard_probe p;
        bool        propagated = false;
        try {
            fused_resolution_unwind_guard guard(guard_report, guard_lost, &p);
            throw std::runtime_error("from inside the scope");
        } catch (const std::runtime_error &) {
            propagated = true;
        }
        CHECK(propagated, "the exception did not propagate");
        CHECK(p.reported == 1 && p.lost == 0, "an unwind reported %d, lost %d, want 1, 0", p.reported, p.lost);
    }

    // a report that throws during unwinding is lost, not a terminate, and the original still propagates
    {
        guard_probe p;
        p.report_throws = true;
        bool propagated = false;
        try {
            fused_resolution_unwind_guard guard(guard_report, guard_lost, &p);
            throw std::runtime_error("original");
        } catch (const std::runtime_error & err) {
            propagated = std::string(err.what()) == "original";
        }
        CHECK(propagated, "the original exception was replaced");
        CHECK(p.reported == 1 && p.lost == 1, "a throwing report: reported %d, lost %d, want 1, 1", p.reported, p.lost);
    }

    // a guard constructed while another exception is already unwinding is silent until a NEW one leaves
    {
        guard_probe p;

        struct nested {
            guard_probe * p;

            ~nested() { fused_resolution_unwind_guard guard(guard_report, guard_lost, p); }
        };

        try {
            nested n{ &p };
            throw std::runtime_error("outer");
        } catch (const std::runtime_error &) {
        }
        CHECK(p.reported == 0, "a guard built during an unwind reported for the exception already in flight");
    }
}

int main() {
    case_1_empty();
    case_2_partial();
    case_3_same_twice();
    case_4_changed();
    case_5_a_b_b();
    case_6_text();
    case_7_guard();

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    fprintf(stderr, "OK: the fused-op resolution report\n");
    return 0;
}
