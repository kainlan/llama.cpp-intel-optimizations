#pragma once

// The tenant element and the cohort ids. The element is defined once, in the public backend
// header (ggml-sycl.h), and the ids in ggml-sycl-cohort.h; this header only gathers both for
// the backend's own sources. The cohort table that gives each id its name, tier, scope and
// lifetime is in context-tenant-measure.hpp.

#include "ggml-sycl-cohort.h"
#include "ggml-sycl.h"
