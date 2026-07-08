#include <sycl/sycl.hpp>
#include "common.hpp"
#include "add-id.hpp"
#include "mem-ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool add_id_validate_enabled() {
    const char * env = std::getenv("GGML_SYCL_ADD_ID_VALIDATE");
    return env != nullptr && std::atoi(env) != 0;
}

static int add_id_validate_limit() {
    const char * env = std::getenv("GGML_SYCL_ADD_ID_VALIDATE_LIMIT");
    return env ? std::max(0, std::atoi(env)) : 32;
}

static bool add_id_barrier_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_SYCL_ADD_ID_BARRIER");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

static void add_id_validate_inputs(
    const char * site,
    sycl::queue & q,
    const ggml_sycl::sycl_tensor & dst,
    const float * src0_d,
    const float * src1_d,
    const int32_t * src2_d,
    float * dst_d,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne11,
    size_t nb11,
    size_t nb20,
    size_t nb21) {
    static int emitted = 0;

    if (!add_id_validate_enabled()) {
        return;
    }

    const int limit = add_id_validate_limit();
    if (limit > 0 && emitted >= limit) {
        return;
    }

    const char * name = dst.name() ? dst.name() : "?";
    if (std::strstr(name, "ffn_moe_") == nullptr) {
        return;
    }

    const size_t src1_bytes = dst.src(1).nbytes();
    const size_t src2_bytes = dst.src(2).nbytes();
    std::vector<unsigned char> src1_host(src1_bytes);
    std::vector<unsigned char> src2_host(src2_bytes);

    const int             queue_device = ggml_sycl_get_device_id_from_queue(q);
    ggml_sycl::mem_handle src1_dst     = ggml_sycl::mem_handle::from_direct(
        src1_host.data(), GGML_LAYOUT_AOS, /*on_device=*/false, ggml_sycl::mem_handle::HOST_DEVICE);
    ggml_sycl::mem_handle src1_src = ggml_sycl_memcpy_handle_for_raw_ptr(src1_d, queue_device, GGML_LAYOUT_AOS,
                                                                         /*fallback_on_device=*/true,
                                                                         /*fallback_unknown=*/true);
    ggml_sycl::mem_handle src2_dst = ggml_sycl::mem_handle::from_direct(
        src2_host.data(), GGML_LAYOUT_AOS, /*on_device=*/false, ggml_sycl::mem_handle::HOST_DEVICE);
    ggml_sycl::mem_handle src2_src = ggml_sycl_memcpy_handle_for_raw_ptr(src2_d, queue_device, GGML_LAYOUT_AOS,
                                                                         /*fallback_on_device=*/true,
                                                                         /*fallback_unknown=*/true);
    ggml_sycl::mem_copy(src1_dst, src1_src, src1_bytes, q);
    ggml_sycl::mem_copy(src2_dst, src2_src, src2_bytes, q);

    int64_t id_min = ne11;
    int64_t id_max = -1;
    int64_t id_oob = 0;
    int64_t selected_nan = 0;
    int64_t selected_inf = 0;
    float selected_min = INFINITY;
    float selected_max = -INFINITY;

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const size_t id_off = static_cast<size_t>(i1) * nb20 + static_cast<size_t>(i2) * nb21;
            if (id_off + sizeof(int32_t) > src2_host.size()) {
                ++id_oob;
                continue;
            }

            const int32_t id = *reinterpret_cast<const int32_t *>(src2_host.data() + id_off);
            id_min = std::min<int64_t>(id_min, id);
            id_max = std::max<int64_t>(id_max, id);
            if (id < 0 || id >= ne11) {
                ++id_oob;
                continue;
            }

            const size_t row_off = static_cast<size_t>(id) * nb11;
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const size_t val_off = row_off + static_cast<size_t>(i0) * sizeof(float);
                if (val_off + sizeof(float) > src1_host.size()) {
                    ++id_oob;
                    break;
                }
                const float v = *reinterpret_cast<const float *>(src1_host.data() + val_off);
                if (std::isnan(v)) {
                    ++selected_nan;
                } else if (std::isinf(v)) {
                    ++selected_inf;
                } else {
                    selected_min = std::min(selected_min, v);
                    selected_max = std::max(selected_max, v);
                }
            }
        }
    }

    fprintf(stderr,
            "[ADD_ID_VALIDATE] site=%s tensor=%s ptrs dst=%p src0=%p src1=%p src2=%p "
            "ne=[%lld,%lld,%lld] src1_ne1=%lld "
            "ids_min=%lld ids_max=%lld ids_oob=%lld selected_nan=%lld selected_inf=%lld "
            "selected_min=%g selected_max=%g src1_bytes=%zu src2_bytes=%zu\n",
            site,
            name,
            (void *) dst_d,
            (const void *) src0_d,
            (const void *) src1_d,
            (const void *) src2_d,
            (long long) ne0,
            (long long) ne1,
            (long long) ne2,
            (long long) ne11,
            (long long) id_min,
            (long long) id_max,
            (long long) id_oob,
            (long long) selected_nan,
            (long long) selected_inf,
            selected_min,
            selected_max,
            src1_bytes,
            src2_bytes);
    ++emitted;
}

static void add_id_kernel(
    const float* src0,
    const float* src1,
    const int32_t* src2,
    float* dst,
    int64_t ne0,
    int64_t ne11,
    size_t nb01,
    size_t nb02,
    size_t nb03,
    size_t nb11,
    size_t nb20,
    size_t nb21,
    size_t nb1,
    size_t nb2,
    size_t nb3,
    sycl::nd_item<3> item_ct1) {
  const int64_t i1 = item_ct1.get_group(2);
  const int64_t i2 = item_ct1.get_group(1);
  const int64_t i3 = item_ct1.get_group(0);

  const int i11 =
      *(const int32_t*)((const char*)src2 + i1 * nb20 + i2 * nb21);

  float* dst_row = (float*)((char*)dst + i3 * nb3 + i2 * nb2 + i1 * nb1);
  const float* src0_row =
      (const float*)((const char*)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01);
  const bool id_valid = i11 >= 0 && i11 < ne11;
  const float* src1_row = id_valid ? (const float*)((const char*)src1 + i11 * nb11) : nullptr;

  for (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = id_valid ? src0_row[i0] + src1_row[i0] : src0_row[i0];
  }
}

void ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_sycl::sycl_tensor dst) {
  auto src0 = dst.src(0);
  auto src1 = dst.src(1);
  auto src2 = dst.src(2);

  const int64_t ne0 = dst.ne(0);
  const int64_t ne1 = dst.ne(1);
  const int64_t ne2 = dst.ne(2);
  const int64_t ne3 = dst.ne(3);
  const int64_t ne11 = src1.ne(1);
  const size_t nb00 = src0.nb(0);
  const size_t nb01 = src0.nb(1);
  const size_t nb02 = src0.nb(2);
  const size_t nb03 = src0.nb(3);
  const size_t nb10 = src1.nb(0);
  const size_t nb11 = src1.nb(1);
  const size_t nb20 = src2.nb(0);
  const size_t nb21 = src2.nb(1);
  const size_t nb1 = dst.nb(1);
  const size_t nb2 = dst.nb(2);
  const size_t nb3 = dst.nb(3);

  GGML_ASSERT(dst.type() == GGML_TYPE_F32);
  GGML_ASSERT(src0.type() == GGML_TYPE_F32);
  GGML_ASSERT(src1.type() == GGML_TYPE_F32);
  GGML_ASSERT(src2.type() == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));
  GGML_ASSERT(nb20 == sizeof(int32_t));

  sycl::queue& q = *ctx.stream();

  const float* src0_d = src0.resolve_as<const float>();
  const float* src1_d = src1.resolve_as<const float>();
  const int32_t* src2_d = src2.resolve_as<const int32_t>();
  float* dst_d = dst.resolve_as<float>();

  int threads = std::min((int)ne0, 768);  // cols

  std::vector<sycl::event> deps;
  auto collect_ready_deps = [&](const ggml_sycl::sycl_tensor & tensor) {
      const ggml_tensor * cur = tensor.raw();
      for (int depth = 0; cur != nullptr && depth < GGML_MAX_SRC; ++depth) {
          auto resolved = ggml_sycl_resolve(cur, ctx.device);
          if (resolved.has_ready_event) {
              try {
                  if (ggml_sycl_should_add_dependency(resolved.ready_event)) {
                      deps.push_back(resolved.ready_event);
                  }
              } catch (...) {
              }
          }
          cur = cur->view_src;
      }
  };
  collect_ready_deps(src0);
  collect_ready_deps(src1);
  collect_ready_deps(src2);

  add_id_validate_inputs(
      "before", q, dst, src0_d, src1_d, src2_d, dst_d, ne0, ne1, ne2, ne11, nb11, nb20, nb21);

  sycl::event evt = q.submit([&](sycl::handler & cgh) {
      if (!deps.empty()) {
          cgh.depends_on(deps);
      }
      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(ne3, ne2, ne1) * sycl::range<3>(1, 1, threads),
              sycl::range<3>(1, 1, threads)),
          [=](sycl::nd_item<3> item_ct1) {
            add_id_kernel(
                src0_d,
                src1_d,
                src2_d,
                dst_d,
                ne0,
                ne11,
                nb01,
                nb02,
                nb03,
                nb11,
                nb20,
                nb21,
                nb1,
                nb2,
                nb3,
                item_ct1);
          });
  });
  if (!ggml_sycl_graph_recording_active()) {
      ggml_sycl_set_tensor_ready_event(const_cast<ggml_tensor *>(dst.raw()), ctx.device, evt);
  }
  if (add_id_barrier_enabled() && !ggml_sycl_graph_recording_active()) {
      (void) q.ext_oneapi_submit_barrier({ evt });
  }

  add_id_validate_inputs(
      "after", q, dst, src0_d, src1_d, src2_d, dst_d, ne0, ne1, ne2, ne11, nb11, nb20, nb21);
}
