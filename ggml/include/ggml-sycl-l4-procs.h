#pragma once

// The proc-address names of the measured-tenant entry points (ggml-sycl.h, "The measured-tenant
// publish, its coverage query and the load-time late check"). The SYCL reg's get_proc_address
// answers each name with the function of the same name, and a caller that resolves them
// (src/llama-context.cpp, through ggml_backend_reg_get_proc_address in every link mode) names
// them only by these macros, so the answer table and the readers cannot drift apart.
//
// A backend that does not define one of them answers null, which every reader treats as
// inert: UNSUPPORTED for the publish, GROWTH for the coverage query, NOT_RECORDED for the
// late check, NOT_ANSWERED for the residency probe.

#define GGML_SYCL_PROC_SET_RUNTIME_CONTEXT_DESC "ggml_backend_sycl_set_runtime_context_desc"
#define GGML_SYCL_PROC_TENANT_COVERAGE          "ggml_backend_sycl_tenant_coverage"
#define GGML_SYCL_PROC_LOAD_LATE_CHECK          "ggml_backend_sycl_load_late_check"
#define GGML_SYCL_PROC_PROBE_RESIDENCY          "ggml_backend_sycl_probe_residency"
