#pragma once

#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

// The outcome of resolving the auto fused ops (flash attention, gated delta net, lightning
// indexer, the DeepSeek V4 HC ops), and the pure functions that print it.
//
// A MEASURE resolves into a scratch cparams copy, so the lines `resolve_fused_ops` used to
// log at the point of resolution would repeat once per measure and describe tenants nobody
// publishes. The resolution is therefore carried as a record and printed once per resolution,
// at the publish attempt that applies it. These helpers touch no context, scheduler or log:
// tests/test-fused-resolution-report.cpp executes them on the host.
//
//   fused_resolution               the record: ten entries, in the order they resolve;
//   fused_resolution_emit()        renders the present entries and hands each to a sink, with
//                                  a per-entry printed mark so a retry never prints one twice;
//   fused_resolution_unwind_guard  reports a record only while an exception is leaving a scope.

enum fused_resolution_entry : int {
    FUSED_RESOLUTION_ENTRY_FA,          // the flash attention probe (no header)
    FUSED_RESOLUTION_ENTRY_GDN_HEADER,  // "resolving fused Gated Delta Net support:"
    FUSED_RESOLUTION_ENTRY_GDN_AR,      // the autoregressive probe
    FUSED_RESOLUTION_ENTRY_GDN_CH,      // the chunked probe
    FUSED_RESOLUTION_ENTRY_LID_HEADER,  // "resolving fused Lightning Indexer support:"
    FUSED_RESOLUTION_ENTRY_LID,
    FUSED_RESOLUTION_ENTRY_HC_HEADER,   // "resolving fused DeepSeek V4 HC support:"
    FUSED_RESOLUTION_ENTRY_HC_PRE,
    FUSED_RESOLUTION_ENTRY_HC_COMB,
    FUSED_RESOLUTION_ENTRY_HC_POST,
    FUSED_RESOLUTION_N_ENTRIES,
};

enum fused_resolution_level : int {
    FUSED_RESOLUTION_LEVEL_INFO,
    FUSED_RESOLUTION_LEVEL_WARN,
};

// One entry. A header entry carries the group's header text in `header`. A probe entry is the
// probe's whole group: whether it resolved to enabled, the device mismatch that stopped it, and
// the layers it executes on the CPU. Names are copied as strings, so a record holds nothing that
// points into the scheduler a measure built.
struct fused_resolution_entry_data {
    bool        present = false;
    std::string header;

    std::string probe_name;
    bool        enabled        = false;
    bool        mismatch       = false;
    int         mismatch_layer = -1;
    std::string mismatch_layer_dev;
    std::string mismatch_probe_dev;
    uint32_t    n_cpu_landings    = 0;
    int         cpu_landing_layer = -1;
    std::string cpu_landing_dev;
    // CPU landings the layer's device could not have executed: a capability gap, not a placement
    // (llama_fused_cpu_landing_is_placement). Counted apart from n_cpu_landings and printed as its own WARN.
    uint32_t    n_cpu_gaps    = 0;
    int         cpu_gap_layer = -1;
    std::string cpu_gap_dev;
};

struct fused_resolution {
    std::array<fused_resolution_entry_data, FUSED_RESOLUTION_N_ENTRIES> entry;

    bool empty() const {
        for (const auto & e : entry) {
            if (e.present) {
                return false;
            }
        }
        return true;
    }
};

struct fused_resolution_line {
    fused_resolution_level level;
    std::string            text;  // no trailing newline
};

// The lines of one present entry, at upstream's text and levels. `prefix` stands where upstream's
// `func` stood.
inline std::vector<fused_resolution_line> fused_resolution_render(const fused_resolution_entry_data & e,
                                                                  const std::string &                 prefix) {
    std::vector<fused_resolution_line> lines;

    if (!e.header.empty()) {
        lines.push_back({ FUSED_RESOLUTION_LEVEL_INFO, prefix + ": " + e.header });
        return lines;
    }

    if (e.mismatch) {
        lines.push_back({ FUSED_RESOLUTION_LEVEL_WARN,
                          prefix + ": layer " + std::to_string(e.mismatch_layer) + " is assigned to device " +
                              e.mismatch_layer_dev + " but " + e.probe_name + " is assigned to device " +
                              e.mismatch_probe_dev + " (usually due to missing support)" });
    }

    if (!e.enabled) {
        lines.push_back(
            { FUSED_RESOLUTION_LEVEL_WARN, prefix + ": " + e.probe_name + " not supported, set to disabled" });
        return lines;
    }

    if (e.n_cpu_landings > 0) {
        lines.push_back(
            { FUSED_RESOLUTION_LEVEL_WARN, prefix + ": " + e.probe_name + " executes on " + e.cpu_landing_dev +
                                               " for " + std::to_string(e.n_cpu_landings) + " layer(s) (e.g. layer " +
                                               std::to_string(e.cpu_landing_layer) +
                                               ") -- the executor follows data placement, not a capability gap" });
    }
    if (e.n_cpu_gaps > 0) {
        lines.push_back({ FUSED_RESOLUTION_LEVEL_WARN, prefix + ": " + e.probe_name + " executes on CPU for " +
                                                           std::to_string(e.n_cpu_gaps) + " layer(s) (e.g. layer " +
                                                           std::to_string(e.cpu_gap_layer) + ") because " +
                                                           e.cpu_gap_dev + " does not support it" });
    }
    lines.push_back({ FUSED_RESOLUTION_LEVEL_INFO, prefix + ": " + e.probe_name + " enabled" });
    return lines;
}

typedef void (*fused_resolution_sink_fn)(fused_resolution_level level, const char * text);

// The marker a reprinted entry's lines carry.
#define FUSED_RESOLUTION_REPRINT_MARK " (re-resolved)"

// Prints the present entries of `rec`, in id order, through `sink`. `printed[i]` is the text entry i
// last printed (empty: never). An empty mark passes the entry's lines to the sink; a mark equal to
// the rendering passes nothing; a mark that differs passes the lines again, each with the reprint
// marker. A mark is set only after the sink has taken the entry's lines, to the rendering. An absent
// entry is neither printed nor marked, so an empty record marks nothing and a partial one marks what
// it printed. The marks outlive one call: they belong to the context.
inline void fused_resolution_emit(const fused_resolution & rec,
                                  std::string (&printed)[FUSED_RESOLUTION_N_ENTRIES],
                                  const char *             prefix,
                                  fused_resolution_sink_fn sink) {
    for (int i = 0; i < FUSED_RESOLUTION_N_ENTRIES; ++i) {
        const fused_resolution_entry_data & e = rec.entry[i];
        if (!e.present) {
            continue;
        }

        const std::vector<fused_resolution_line> lines = fused_resolution_render(e, prefix);

        std::string rendering;
        for (const auto & line : lines) {
            rendering += line.text;
            rendering += '\n';
        }

        if (printed[i] == rendering) {
            continue;
        }

        const bool reprint = !printed[i].empty();
        for (const auto & line : lines) {
            sink(line.level, reprint ? (line.text + FUSED_RESOLUTION_REPRINT_MARK).c_str() : line.text.c_str());
        }
        printed[i] = rendering;
    }
}

// Calls `report` from its destructor only when an exception has begun to leave the scope since the
// guard was constructed: a normal return is silent, so the lines are printed at the publish attempt
// and nowhere else. A throw out of a destructor during unwinding is std::terminate and the report
// renders strings, so a throwing report is caught and handed to `lost`, which must not throw.
class fused_resolution_unwind_guard {
  public:
    typedef void (*report_fn)(void * arg);
    typedef void (*lost_fn)(void * arg) noexcept;

    fused_resolution_unwind_guard(report_fn report, lost_fn lost, void * arg) :
        report_(report),
        lost_(lost),
        arg_(arg),
        n_at_construction_(std::uncaught_exceptions()) {}

    ~fused_resolution_unwind_guard() {
        if (std::uncaught_exceptions() > n_at_construction_) {
            try {
                report_(arg_);
            } catch (...) {
                lost_(arg_);
            }
        }
    }

    fused_resolution_unwind_guard(const fused_resolution_unwind_guard &)             = delete;
    fused_resolution_unwind_guard & operator=(const fused_resolution_unwind_guard &) = delete;

  private:
    report_fn report_;
    lost_fn   lost_;
    void *    arg_;
    int       n_at_construction_;
};
