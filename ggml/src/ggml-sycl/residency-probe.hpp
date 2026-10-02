// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// The residency probe's pure core (llama.cpp-moua L4 step 3d, llama.cpp-5cim).  Given one input value it answers which
// layers a context's plan leaves in host memory: on each device the KV slots of the layers planned there are fitted
// on the device's shared zone behind the context's tenant head slots (kv_region_fit, the one fit the commit carves
// with), and a layer the fit demotes, or the caller forced, is host-resident.
//
// Host-testable: it names only the public ABI enum (ggml-sycl.h), the cohort table, the fit and the standard library,
// so tests/test-residency-probe-core.cpp builds it with no device.  It never touches the unified cache.
//
// What the core owns:
//
//   * determinism: the answer is a pure function of the input value.  The tenants are canonicalized into
//     (device, cohort, slot_index) order before the head slots are built, so the order they arrive in cannot change
//     a byte of the answer;
//   * demote-only: a tenant's head slot sits in front of the KV, so adding a tenant or growing one can only take
//     room from the layers, never give any back;
//   * a status that is not OK carries no vector (host_resident is empty): a caller that reads the bytes of a
//     refusal reads nothing, never a plausible "all device-resident".
//
// What it does not own: the geometry.  Step 1d's whole job is to fill residency_probe_input::devices[].geometry from
// the live shared-zone census under the group mutex and to hand the planned device of each layer; the core's contract
// does not change with it.  The proc in ggml-sycl.cpp answers GEOMETRY_NOT_WIRED until then.
//
// A head slot is built with names_weight false: the cohort table carries no zone, and the cohorts that exist today
// are served by the zones' KV-taking TLSFs.  Step 1d fills it from the allocator's zone vocabulary when a cohort
// names WEIGHT.

#ifndef GGML_SYCL_RESIDENCY_PROBE_HPP
#define GGML_SYCL_RESIDENCY_PROBE_HPP

#include "context-tenant-measure.hpp"
#include "ggml-sycl.h"
#include "kv-runtime-demotion.hpp"
#include "runtime-context-section.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_sycl {

// One device the probe fits on: its index (the one tenants and layers name) and the shared-zone geometry the fit reads.
struct residency_probe_device {
    int32_t              device = 0;
    shared_zone_geometry geometry;
};

// One layer, by its id (the layer's position in residency_probe_input::layers).  `device` is the device the placement
// plan has the layer on, or -1 when the plan has it on the host.  A layer with no KV (has_kv false) has nothing to
// place and answers 0.
struct residency_probe_layer {
    bool          has_kv        = false;
    kv_slot_group group         = KV_SLOT_FULL;
    size_t        kv_bytes      = 0;
    size_t        sidecar_bytes = 0;
    int32_t       device        = -1;
};

struct residency_probe_input {
    std::vector<residency_probe_device> devices;
    std::vector<residency_probe_layer>  layers;       // index = layer id; the answer has one byte per layer
    std::vector<runtime_context_tenant> tenants;      // any order; a host-tier tenant names no zone and is ignored
    std::vector<uint32_t>               forced_host;  // layers that must be host-resident
    bool                                no_promotion = false;
};

struct residency_probe_result {
    ggml_sycl_residency_probe_status status = GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED;
    std::vector<uint8_t>             host_resident;  // one byte per layer: empty unless status == OK
    std::string                      reason;         // empty exactly when status == OK
};

namespace residency_probe_detail {

inline residency_probe_result refuse(ggml_sycl_residency_probe_status status, const std::string & reason) {
    residency_probe_result r;
    r.status = status;
    r.reason = reason;
    return r;
}

inline const residency_probe_device * find_device(const residency_probe_input & in, int32_t device) {
    for (const residency_probe_device & d : in.devices) {
        if (d.device == device) {
            return &d;
        }
    }
    return nullptr;
}

}  // namespace residency_probe_detail

// The caller's result struct may be written only if the caller declared one this module knows: a pointer, a struct_size
// at least the layout the proc writes (n_layer, and host_resident through the pointer the caller owns), the version it
// understands.  A smaller or older struct is refused, so the proc never writes past what the caller declared; a larger
// struct of this version is read as the layout this module knows (a newer caller bumps the version and is refused).
// The proc asks this before it writes a byte.
inline bool residency_probe_out_declared(const ggml_sycl_residency_probe * out) {
    return out != nullptr && out->struct_size >= sizeof(*out) && out->version == GGML_SYCL_RESIDENCY_PROBE_VERSION;
}

// The largest byte count the core accepts for one layer's KV (kv_bytes + sidecar_bytes) or one tenant's slot: 64 TiB.
// kv_region_fit rounds a size up to its slot block, so a size within one block of SIZE_MAX wraps to a small number and
// the layer would answer "device-resident" on no room at all.  Nothing real is near the bound (no zone holds a slot
// that large), and with the descriptor reader's element cap (65536 layers, 65536 tenants) every sum the fit forms
// from values at the bound stays below 2^63, so none of its arithmetic can wrap.  A count past it is a malformed
// input, INVALID by name, never an answer.
constexpr size_t residency_probe_max_bytes = (size_t) 1 << 46;

inline residency_probe_result residency_probe_core(const residency_probe_input & in) {
    using namespace residency_probe_detail;
    const auto   INVALID = GGML_SYCL_RESIDENCY_PROBE_INVALID;
    const size_t n_layer = in.layers.size();

    // The element counts the byte bound's arithmetic rests on: the descriptor reader's cap.
    if (n_layer > RUNTIME_CONTEXT_DESC_MAX_ELEMENTS || in.tenants.size() > RUNTIME_CONTEXT_DESC_MAX_ELEMENTS) {
        return refuse(INVALID, "more than " + std::to_string(RUNTIME_CONTEXT_DESC_MAX_ELEMENTS) + " layers or tenants");
    }

    // The devices: each named once, each a real device index (the host's -1 names no geometry).
    for (const residency_probe_device & d : in.devices) {
        if (d.device < 0) {
            return refuse(INVALID, "a geometry is named for device " + std::to_string(d.device));
        }
    }
    for (size_t i = 0; i < in.devices.size(); ++i) {
        for (size_t j = i + 1; j < in.devices.size(); ++j) {
            if (in.devices[i].device == in.devices[j].device) {
                return refuse(INVALID, "device " + std::to_string(in.devices[i].device) + " has two geometries");
            }
        }
    }

    // The layers: a layer that has KV has bytes to place and sits on a device the input has a geometry for, or on the
    // host.
    for (size_t l = 0; l < n_layer; ++l) {
        const residency_probe_layer & layer = in.layers[l];
        if (!layer.has_kv) {
            continue;
        }
        if (layer.kv_bytes == 0) {
            return refuse(INVALID, "layer " + std::to_string(l) + " has KV and no bytes");
        }
        if (layer.kv_bytes > residency_probe_max_bytes || layer.sidecar_bytes > residency_probe_max_bytes ||
            layer.kv_bytes + layer.sidecar_bytes > residency_probe_max_bytes) {
            return refuse(INVALID, "layer " + std::to_string(l) + " has more KV bytes than the probe can place");
        }
        if (layer.device >= 0 && find_device(in, layer.device) == nullptr) {
            return refuse(INVALID, "layer " + std::to_string(l) + " is planned on device " +
                                       std::to_string(layer.device) + ", which has no geometry");
        }
        if (layer.device < -1) {
            return refuse(INVALID, "layer " + std::to_string(l) + " has a device below -1");
        }
    }

    // The forced set: sorted, each layer a layer with KV, none twice.
    std::vector<uint32_t> forced = in.forced_host;
    std::sort(forced.begin(), forced.end());
    for (size_t i = 0; i < forced.size(); ++i) {
        if (i > 0 && forced[i] == forced[i - 1]) {
            return refuse(INVALID, "layer " + std::to_string(forced[i]) + " is forced to the host twice");
        }
        if (forced[i] >= n_layer) {
            return refuse(INVALID, "forced layer " + std::to_string(forced[i]) + " is past the model's " +
                                       std::to_string(n_layer) + " layers");
        }
        if (!in.layers[forced[i]].has_kv) {
            return refuse(INVALID, "forced layer " + std::to_string(forced[i]) + " has no KV");
        }
    }

    // The tenants, canonical: (device, cohort, slot_index) order whatever order they came in.
    std::vector<runtime_context_tenant> tenants = in.tenants;
    std::sort(tenants.begin(), tenants.end(), runtime_context_tenant_less);
    for (size_t i = 0; i < tenants.size(); ++i) {
        const runtime_context_tenant &        t    = tenants[i];
        const ggml_sycl_context_cohort_info * info = ggml_sycl_context_cohort_lookup(t.cohort);
        if (info == nullptr) {
            return refuse(INVALID, "tenant of cohort " + std::to_string(t.cohort) + ", which the table does not know");
        }
        if (i > 0 && !runtime_context_tenant_less(tenants[i - 1], t)) {
            return refuse(INVALID, std::string("two tenants at one slot of cohort ") + info->name);
        }
        if (t.slot_bytes == 0) {
            return refuse(INVALID, std::string("tenant of cohort ") + info->name + " has no bytes");
        }
        if (t.slot_bytes > residency_probe_max_bytes) {
            return refuse(INVALID,
                          std::string("tenant of cohort ") + info->name + " has more bytes than the probe can place");
        }
        const bool host_tier = info->tier == GGML_SYCL_CONTEXT_COHORT_TIER_HOST_PINNED;
        if (host_tier ? t.device != -1 : t.device < 0) {
            return refuse(INVALID, std::string("tenant of cohort ") + info->name + " is on device " +
                                       std::to_string(t.device) + ", which is not its tier's");
        }
        if (!host_tier && find_device(in, t.device) == nullptr) {
            return refuse(INVALID, std::string("tenant of cohort ") + info->name + " is on device " +
                                       std::to_string(t.device) + ", which has no geometry");
        }
    }

    // One fit per device, in device order.  A device with nothing to place needs none.
    std::vector<const residency_probe_device *> order;
    for (const residency_probe_device & d : in.devices) {
        order.push_back(&d);
    }
    std::sort(order.begin(), order.end(),
              [](const residency_probe_device * a, const residency_probe_device * b) { return a->device < b->device; });

    residency_probe_result result;
    std::vector<uint8_t>   host(n_layer, 0);
    for (size_t l = 0; l < n_layer; ++l) {
        host[l] = (in.layers[l].has_kv && in.layers[l].device < 0) ? 1 : 0;  // the plan has it on the host
    }
    std::vector<uint32_t> unforced_demoted;
    for (const residency_probe_device * d : order) {
        kv_region_request req;
        for (size_t l = 0; l < n_layer; ++l) {
            const residency_probe_layer & layer = in.layers[l];
            if (layer.has_kv && layer.device == d->device) {
                kv_layer_slot_request s;
                s.layer         = (uint32_t) l;
                s.group         = layer.group;
                s.kv_bytes      = layer.kv_bytes;
                s.sidecar_bytes = layer.sidecar_bytes;
                req.layers.push_back(s);
                if (std::binary_search(forced.begin(), forced.end(), (uint32_t) l)) {
                    req.forced_host.push_back((uint32_t) l);
                }
            }
        }
        for (const runtime_context_tenant & t : tenants) {
            if (t.device != d->device) {
                continue;
            }
            kv_head_slot_request h;
            h.scope        = demand_scope::CONTEXT;
            h.owner        = 1;
            h.cohort       = ggml_sycl_context_cohort_lookup(t.cohort)->name;
            h.index        = t.slot_index;
            h.size         = (size_t) t.slot_bytes;
            h.names_weight = false;
            req.head_slots.push_back(h);
        }
        if (req.layers.empty() && req.head_slots.empty()) {
            continue;
        }
        const kv_region_fit_result fit = kv_region_fit(d->geometry, req);
        if (!fit.fits) {
            std::string what;
            for (size_t i : fit.refused_heads) {
                what += std::string(what.empty() ? "" : ", ") + req.head_slots[i].cohort + " slot " +
                        std::to_string(req.head_slots[i].index) + " (" + std::to_string(req.head_slots[i].size) + " B)";
            }
            return refuse(GGML_SYCL_RESIDENCY_PROBE_HEAD_SLOT_REFUSED,
                          "device " + std::to_string(d->device) + ": the head slot(s) " + what +
                              " fit on no zone even with every KV layer on the host");
        }
        for (const kv_layer_placement & p : fit.layers) {
            if (p.device) {
                continue;
            }
            host[p.layer] = 1;
            if (p.cause != KV_DEMOTE_FORCED_HOST) {
                unforced_demoted.push_back(p.layer);
            }
        }
    }

    // A layer the plan already had on the host is not a promotion's cause; only a demotion the fit made is.
    if (in.no_promotion && !unforced_demoted.empty()) {
        std::sort(unforced_demoted.begin(), unforced_demoted.end());
        return refuse(
            GGML_SYCL_RESIDENCY_PROBE_NO_PROMOTION_VIOLATED,
            "no_promotion is set and layer " + std::to_string(unforced_demoted.front()) +
                (unforced_demoted.size() > 1 ? " (and " + std::to_string(unforced_demoted.size() - 1) + " more)" :
                                               std::string()) +
                " would be demoted without being forced");
    }
    result.status        = GGML_SYCL_RESIDENCY_PROBE_OK;
    result.host_resident = std::move(host);
    return result;
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_RESIDENCY_PROBE_HPP
