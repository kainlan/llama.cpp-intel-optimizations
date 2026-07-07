#pragma once

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace sycl_bench {

enum class KernelKind {
    MMVQ,
    MMQ,
    ONEDNN_FP16_GEMM,
    ONEDNN_INT8_GEMM,
    ONEDNN_WOQ_GEMM,
    ONEDNN_MXFP4_GEMM,
    UNIFIED_MATMUL,
    MEMORY_BANDWIDTH,
    MXFP4_DECODE_BANDWIDTH,
    MXFP4_INLINE_DOT,
    MXFP4_SELECTED_READ,
    MXFP4_SELECTED_KMAJOR,
    MXFP4_SELECTED_XMX_DPAS,
    MXFP4_PAIR_GLU,
    MXFP4_LAYER_GLU_DOWN,
    MXFP4_MMV_ID,
    MXFP4_MMV_ID_F32,
    MXFP4_MMV_ID_XMX_TILED,
    MXFP4_DPAS_GROUPED,
    ROOFLINE_COMPUTE,
    DPAS_EXPLORATION,
};

struct KernelInfo {
    const char *     name;
    ggml_layout_mode layout;
    KernelKind       kind;
};

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline const std::vector<KernelInfo> & kernel_list() {
    static const std::vector<KernelInfo> kernels = {
        { "mmvq_aos",                                                  GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_aos_baseline",                                         GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_soa",                                                  GGML_LAYOUT_SOA,       KernelKind::MMVQ                    },
        { "mmvq_soa_baseline",                                         GGML_LAYOUT_SOA,       KernelKind::MMVQ                    },
        { "mmvq_coalesced",                                            GGML_LAYOUT_COALESCED, KernelKind::MMVQ                    },
        { "mmvq_slm_cached",                                           GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_prefetch",                                             GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_wide_load",                                            GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_block_load",                                     GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_slm",                                            GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_tile_8x8",                                         GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_tile_16x16",                                       GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_aos_direct",                                       GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_soa_direct",                                       GGML_LAYOUT_SOA,       KernelKind::MMVQ                    },
        { "mmvq_xmx_double_buffer",                                    GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_dpas_1x16x32",                                   GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_dpas_8x16x32",                                   GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_dpas_chained",                                   GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_tile_64x64",                                       GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_register_accum",                                   GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_multi_wg",                                         GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_persistent",                                       GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_large_tile",                                     GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_persistent",                                     GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_lsc_prefetch",                                   GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_hybrid_adaptive",                                      GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_xmx_fused",                                            GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_coalesced_xmx_aligned",                                GGML_LAYOUT_COALESCED, KernelKind::MMVQ                    },
        { "mmvq_esimd_hybrid",                                         GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_esimd_cooperative",                                    GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_q4_0_specialized",                                     GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_q6_k_specialized",                                     GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmvq_mxfp4_native",                                         GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "mmq_aos",                                                   GGML_LAYOUT_AOS,       KernelKind::MMQ                     },
        { "mmq_soa",                                                   GGML_LAYOUT_SOA,       KernelKind::MMQ                     },
        { "mmq_coalesced",                                             GGML_LAYOUT_COALESCED, KernelKind::MMQ                     },
        { "mmq",                                                       GGML_LAYOUT_AOS,       KernelKind::MMQ                     },
        { "dpas_baseline",                                             GGML_LAYOUT_AOS,       KernelKind::DPAS_EXPLORATION        },
        { "dpas_sweep",                                                GGML_LAYOUT_AOS,       KernelKind::DPAS_EXPLORATION        },
        { "dpas_memory_patterns",                                      GGML_LAYOUT_AOS,       KernelKind::DPAS_EXPLORATION        },
        { "mmvq",                                                      GGML_LAYOUT_AOS,       KernelKind::MMVQ                    },
        { "onednn_fp16_gemm",                                          GGML_LAYOUT_AOS,       KernelKind::ONEDNN_FP16_GEMM        },
        { "onednn_int8_gemm",                                          GGML_LAYOUT_AOS,       KernelKind::ONEDNN_INT8_GEMM        },
        { "onednn_woq_gemm",                                           GGML_LAYOUT_AOS,       KernelKind::ONEDNN_WOQ_GEMM         },
        { "onednn_mxfp4_gemm",                                         GGML_LAYOUT_AOS,       KernelKind::ONEDNN_MXFP4_GEMM       },
        { "onednn_mxfp4_user_gemm",                                    GGML_LAYOUT_AOS,       KernelKind::ONEDNN_MXFP4_GEMM       },
        { "onednn_mxfp4_f32scale_gemm",                                GGML_LAYOUT_AOS,       KernelKind::ONEDNN_MXFP4_GEMM       },
        { "onednn_mxfp4_f32dst_gemm",                                  GGML_LAYOUT_AOS,       KernelKind::ONEDNN_MXFP4_GEMM       },
        { "onednn_mxfp4_f32scale_f32dst_gemm",                         GGML_LAYOUT_AOS,       KernelKind::ONEDNN_MXFP4_GEMM       },
        { "unified_matmul",                                            GGML_LAYOUT_AOS,       KernelKind::UNIFIED_MATMUL          },
        { "memory_bandwidth",                                          GGML_LAYOUT_AOS,       KernelKind::MEMORY_BANDWIDTH        },
        { "mxfp4_decode_aos",                                          GGML_LAYOUT_AOS,       KernelKind::MXFP4_DECODE_BANDWIDTH  },
        { "mxfp4_decode_soa",                                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_DECODE_BANDWIDTH  },
        { "mxfp4_decode_f16_aos",                                      GGML_LAYOUT_AOS,       KernelKind::MXFP4_DECODE_BANDWIDTH  },
        { "mxfp4_decode_f16_soa",                                      GGML_LAYOUT_SOA,       KernelKind::MXFP4_DECODE_BANDWIDTH  },
        { "mxfp4_inline_dot_aos",                                      GGML_LAYOUT_AOS,       KernelKind::MXFP4_INLINE_DOT        },
        { "mxfp4_inline_dot_soa",                                      GGML_LAYOUT_SOA,       KernelKind::MXFP4_INLINE_DOT        },
        { "mxfp4_selected_read_aos",                                   GGML_LAYOUT_AOS,       KernelKind::MXFP4_SELECTED_READ     },
        { "mxfp4_selected_read_soa",                                   GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_READ     },
        { "mxfp4_selected_read_interleave_aos",                        GGML_LAYOUT_AOS,       KernelKind::MXFP4_SELECTED_READ     },
        { "mxfp4_selected_read_interleave_soa",                        GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_READ     },
        { "mxfp4_selected_kmajor_read",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_KMAJOR   },
        { "mxfp4_selected_kmajor_tile_read",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_KMAJOR   },
        { "mxfp4_selected_kmajor_pair_glu",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_KMAJOR   },
        { "mxfp4_selected_xmx_dpas_tile_r8",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_XMX_DPAS },
        { "mxfp4_selected_xmx_dpas_tile_r8_m2",                        GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_XMX_DPAS },
        { "mxfp4_selected_xmx_dpas_tile_r8_tn2",                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_XMX_DPAS },
        { "mxfp4_selected_xmx_dpas_tile_r8_tn4",                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_SELECTED_XMX_DPAS },
        { "mxfp4_pair_glu_soa_r1",                                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r2",                                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4",                                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r8",                                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r16",                                    GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_t4",                                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_t8",                                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_t16",                                 GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_t512",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_t4",                                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_t8",                                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_t16",                                 GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_t512",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_noscale",                             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_noscale",                             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_sparse32",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_sparse32",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_sparse32_bias",                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_sparse32_bias",                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_soa",                                    GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_soa_tn2",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_soa_tn4",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_soa_t4",                                 GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_soa_t8",                                 GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_soa_t16",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r1",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r4",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r8",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_singlecol_r1",                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_singlecol_r2",                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_singlecol_r4",                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_multirhs_n2_r8",                   GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_multirhs_n4_r8",                   GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r8_bias",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_direct_r8_m2",                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_direct_r8_m2_bias",                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_direct_r8_m2_sparse32_bias",       GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8",                        GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8_m2",                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_bias",                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_loadv2",              GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_loadv2_bias",         GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias",       GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias",    GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_bundle4_packed_r8_m2",             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_bundle4_packed_r8_m2_sparse32_bias", GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8_m4",                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8_m4_bias",                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_r8_m4_sparse32_bias",       GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_prefetch_r8_m2",            GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_packed_prefetch_r8_m2_bias",       GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r8_tn2",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r8_tn4",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r8_t4",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r8_t16",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r8_t85",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_r8_t512",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_grouped_r8_t512",                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_grouped_r8_t512_tn2",              GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_grouped_r8_t512_tn4",              GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_grouped_r8_t512_bias",             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_grouped_r8_t512_tn2_bias",         GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_xmx_tiled_grouped_r8_t512_tn4_bias",         GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_cache",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_nocache",                             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_vecq",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_vecq",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_scale96",                             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_scale128",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_scale96",                             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_scale128",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_vecq_scale96",                        GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_vecq_scale128",                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_noscale",                             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_noscale",                             GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_sg16_noscale",                        GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_sg16_noscale",                        GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r1_sg16",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_soa_r4_sg16",                                GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_split_soa_r1_sg16",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_split_soa_r4_sg16",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_split_predecoded_r1_sg16",                   GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_split_predecoded_r4_sg16",                   GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_pair_glu_split_soa_r4",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
        { "mxfp4_layer_glu_down_soa_r1",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r2",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r8",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r16",                              GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r1_t4",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r1_t8",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r1_t16",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_t4",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_t8",                            GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_t16",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_t512",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_cache",                         GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_nocache",                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_sparse32",                      GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_bias",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_vecq",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r1_sg16",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_soa_r4_sg16",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_xmx_tiled_r8_t512",                    GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_xmx_tiled_packed_r8_m2_t512",          GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_xmx_tiled_packed_r8_m4_t512",          GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_xmx_tiled_packed_prefetch_r8_m2_t512", GGML_LAYOUT_SOA,
         KernelKind::MXFP4_LAYER_GLU_DOWN                                                                                         },
        { "mxfp4_layer_glu_down_xmx_tiled_grouped_r8_t512",            GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_layer_glu_down_xmx_tiled_grouped_r8_t512_bias",       GGML_LAYOUT_SOA,       KernelKind::MXFP4_LAYER_GLU_DOWN    },
        { "mxfp4_mmv_id_soa_r1",                                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r2",                                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4",                                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r8",                                       GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r16",                                      GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_t4",                                    GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_t8",                                    GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_t16",                                   GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_t4",                                    GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_t8",                                    GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_t16",                                   GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_noscale",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_noscale",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_sparse32",                              GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_sparse32",                              GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_cache",                                 GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_nocache",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_vecq",                                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_vecq",                                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_scale96",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_scale128",                              GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_scale96",                               GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_scale128",                              GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_vecq_scale96",                          GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_vecq_scale128",                         GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r1_sg16",                                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_soa_r4_sg16",                                  GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_predecoded_r1_sg16",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_predecoded_r4_sg16",                           GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_predecoded_r4_cache_sg16",                     GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID            },
        { "mxfp4_mmv_id_f32_soa_sg16",                                 GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID_F32        },
        { "mxfp4_mmv_id_f32_soa_sg32",                                 GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID_F32        },
        { "mxfp4_mmv_id_f32_soa_noscale_sg16",                         GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID_F32        },
        { "mxfp4_mmv_id_f32_soa_noscale_sg32",                         GGML_LAYOUT_SOA,       KernelKind::MXFP4_MMV_ID_F32        },
        { "mxfp4_mmv_id_xmx_tiled",                                    GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_tn1",                                GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_tn2",                                GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_tn4",                                GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_raw_tn1",                            GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_raw_tn2",                            GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_raw_tn4",                            GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_sparse32_tn1",                       GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_sparse32_tn2",                       GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_sparse32_tn4",                       GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_raw_sparse32_tn1",                   GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_raw_sparse32_tn2",                   GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_raw_sparse32_tn4",                   GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_t512_tn1",                           GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_t512_tn2",                           GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_t512_tn4",                           GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_sparse32_t512_tn1",                  GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_sparse32_t512_tn2",                  GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_xmx_tiled_sparse32_t512_tn4",                  GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_dpas_i8rm_t512",                               GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_dpas_i8rm_sparse32_t512",                      GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_dpas_i8rm",                                    GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_mmv_id_dpas_i8rm_sparse32",                           GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_MMV_ID_XMX_TILED  },
        { "mxfp4_dpas_expanded_raw",                                   GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_expanded_raw_pf4",                               GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_expanded_raw_nt2",                               GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_expanded_raw_nt4",                               GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_raw",                                    GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_raw_nt2",                                GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_raw_nt4",                                GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_scaled",                                 GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_scaled_nt2",                             GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_scaled_nt4",                             GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_scaled_ps",                              GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_scaled_ps_nt2",                          GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_scaled_ps_nt4",                          GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_bytescale",                              GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_bytescale_nt2",                          GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_bytescale_nt4",                          GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_compact_bytescale_xmxlayout",                    GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_scaled_predecoded",                              GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_scaled_predecoded_nt2",                          GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_scaled_predecoded_nt4",                          GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_scaled_predecoded_ps",                           GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_scaled_predecoded_ps_nt2",                       GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "mxfp4_dpas_scaled_predecoded_ps_nt4",                       GGML_LAYOUT_XMX_TILED, KernelKind::MXFP4_DPAS_GROUPED      },
        { "roofline_compute",                                          GGML_LAYOUT_AOS,       KernelKind::ROOFLINE_COMPUTE        },
    };
    return kernels;
}

inline const KernelInfo * find_kernel(std::string_view name) {
    const std::string needle = to_lower(std::string(name));
    for (const auto & kernel : kernel_list()) {
        if (needle == kernel.name) {
            return &kernel;
        }
    }
    return nullptr;
}

inline const char * layout_name(ggml_layout_mode layout) {
    switch (layout) {
        case GGML_LAYOUT_AOS:
            return "AOS";
        case GGML_LAYOUT_SOA:
            return "SOA";
        case GGML_LAYOUT_COALESCED:
            return "COALESCED";
        case GGML_LAYOUT_MXFP4_I8:
            return "MXFP4_I8";
        case GGML_LAYOUT_XMX_TILED:
            return "XMX_TILED";
        case GGML_LAYOUT_XMX_GEMM_TILED:
            return "XMX_GEMM_TILED";
        default:
            return "UNKNOWN";
    }
}

inline bool kernel_supports_layout(ggml_type type, ggml_layout_mode layout) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q6_K:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED;
        case GGML_TYPE_Q4_K:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
            return layout == GGML_LAYOUT_AOS;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
            return layout == GGML_LAYOUT_AOS;
        default:
            return false;
    }
}

}  // namespace sycl_bench
