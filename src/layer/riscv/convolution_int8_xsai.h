// convolution_int8_xsai.h — XSAI AME int8 convolution for the riscv port.
//
// Compile-time guard: NCNN_XSAI_AME (set by the xsai-env rootfs app build,
// which also adds the ame_xsa.h / xsai_alloc.h include paths). Runtime gate:
// env XSAI_NCNN_AME=1 plus the ame_xsa.c tile-spec/shape checks below; any
// gate failure makes forward_int8_xsai return -1 and the caller keeps the
// stock NCNN scalar kernel — the AME path is either right or not installed,
// never silently wrong (skills/xsai-ame-mnn-porting discipline).
//
// The riscv int8 convolution is otherwise a TODO: Convolution_riscv::forward
// unpacks to pack1 fp32 and calls the scalar Convolution::forward_int8
// (src/layer/convolution.cpp). This path computes the identical direct
// convolution as an int8 GEMM on the AME matrix unit:
//
//   C[M,N] = A[M,K] x B[N,K]^T
//     M = outw*outh output pixels, N = num_output, K = channels*maxk
//
// Layout contract with ame_xsa.c::xsai_ame_gemm_tile_kloop:
//   A/B tiles are k_tiles blocks of tile_m x tile_k / tile_n x tile_k int8
//   rows (k zero-padded to the 64-byte tile_k); C is tile_m x tile_n int32
//   with row stride tile_n (NEVER a hardcoded 64).
//   Operands must live in the xsai physical-contiguous pool (xsai_malloc).
//
// AME instructions exist only in ame_xsa.c (compiled with the zames march);
// this file is plain C++ calling through ame_xsa.h.
#ifndef NCNN_CONVOLUTION_INT8_XSAI_H
#define NCNN_CONVOLUTION_INT8_XSAI_H

#include "mat.h"
#include "fused_activation.h"

#include "ame_xsa.h"
#include "xsai_alloc.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

namespace ncnn {

// ---------------------------------------------------------------------------
// capability gate + pool-backed tile scratch (mirrors MNN Int8GemmKernelXSAI)
// ---------------------------------------------------------------------------

// Max K (int8 elements) a single call may pack: 8192 covers every conv in the
// models under test (ic*kx*ky <= 8192). Larger shapes stay on the stock kernel.
static constexpr int kXsaiAmeKMax = 8192;
static constexpr size_t kXsaiAmeTileBytes = 128 * (size_t)kXsaiAmeKMax; // tile_m_max(128) x K

static int8_t* g_xsai_atile = 0;
static int8_t* g_xsai_btile = 0;
static int32_t* g_xsai_ctile = 0;
static bool g_xsai_warmed = false; // qemu-user cold-start: first kloop re-run
static bool g_xsai_prof_installed = false;
static unsigned long long g_xsai_ame_calls = 0;
static unsigned long long g_xsai_ame_cycles = 0;

static void xsai_ame_print_summary()
{
    fprintf(stderr, "[xsai-ncnn] ame calls=%llu cycles=%llu\n", g_xsai_ame_calls, g_xsai_ame_cycles);
}

static bool xsai_ame_tiles_init_ncnn()
{
    if (g_xsai_atile != 0)
        return true;
    g_xsai_atile = (int8_t*)xsai_malloc(kXsaiAmeTileBytes);
    g_xsai_btile = (int8_t*)xsai_malloc(kXsaiAmeTileBytes);
    g_xsai_ctile = (int32_t*)xsai_malloc(128 * 128 * 4);
    if (g_xsai_atile == 0 || g_xsai_btile == 0 || g_xsai_ctile == 0)
    {
        fprintf(stderr, "[xsai-ncnn] fatal: xsai_malloc failed for tiles\n");
        return false;
    }
    // Contiguity is an AMU requirement on real hardware; qemu-user addresses
    // operands through guest virtual addresses, so accept host-test mode.
    if (!xsai_pool_phys_contiguous() && !xsai_alloc_host_test_mode())
    {
        fprintf(stderr, "[xsai-ncnn] fatal: xsai pool not physically contiguous\n");
        return false;
    }
    if (!g_xsai_prof_installed)
    {
        g_xsai_prof_installed = true;
        atexit(xsai_ame_print_summary);
    }
    return true;
}

static bool xsai_conv_int8_ame_on()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* env = getenv("XSAI_NCNN_AME");
        const int env_on = (env != 0 && env[0] == '1' && env[1] == '\0') ? 1 : 0;
        const int spec_ok = (xsai_ame_tile_spec_get()->enabled != 0) ? 1 : 0;
        cached = (env_on && spec_ok && xsai_ame_tiles_init_ncnn()) ? 1 : 0;
        if (cached)
        {
            const xsai_ame_tile_spec* spec = xsai_ame_tile_spec_get();
            fprintf(stderr, "[xsai-ncnn] AME int8 conv enabled (tile %dx%dx%d)\n",
                    spec->tile_m, spec->tile_k, spec->tile_n);
        }
    }
    return cached == 1;
}

static inline signed char xsai_f2i8(float v)
{
    int int32 = (int)roundf(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

// ---------------------------------------------------------------------------
// create_pipeline prepack: B tile blocks for the whole int8 weight matrix.
//
// Stored as [nb][k_tiles][tile_n][tile_k] bytes where nb indexes tile_n-sized
// output-channel blocks (rows beyond num_output zero-filled, K tail
// zero-padded to tile_k-byte rows). forward() memcpy's one block into the pool
// btile per N-block instead of re-gathering the row-major weights per call.
// Returns an empty Mat when a gate rejects the layer.
// tile_k 来自硬件探测（trlenb），**不是常量 64**：tlenb/trlenb 有多种配置，
// 写死会和发射端（msettilek）对不上。
// ---------------------------------------------------------------------------
static Mat xsai_conv_int8_prepack_weight(const Mat& weight_data, int num_output, int K, int tile_n)
{
    const int tile_k = xsai_ame_tile_spec_get()->tile_k;
    if (tile_k <= 0)
        return Mat();
    const int k_tiles = (K + tile_k - 1) / tile_k;
    const int K_pad = k_tiles * tile_k;
    if (K <= 0 || K_pad > kXsaiAmeKMax)
        return Mat();
    if ((long long)num_output * K < (long long)xsai_ame_min_macs())
        return Mat();

    const int nb = (num_output + tile_n - 1) / tile_n;
    Mat packed(nb * k_tiles * tile_n * tile_k, (size_t)1u);
    if (packed.empty())
        return Mat();
    memset((void*)packed.data, 0, packed.total());

    for (int b = 0; b < nb; b++)
    {
        signed char* blk = (signed char*)packed.data + (size_t)b * k_tiles * tile_n * tile_k;
        const int oc0 = b * tile_n;
        const int nn = (num_output - oc0 < tile_n) ? num_output - oc0 : tile_n;
        for (int n = 0; n < nn; n++)
        {
            const signed char* wrow = (const signed char*)weight_data + (size_t)(oc0 + n) * K;
            for (int kt = 0; kt < k_tiles; kt++)
            {
                const int off = kt * tile_k;
                const int len = (K - off < tile_k) ? K - off : tile_k;
                memcpy(blk + (size_t)kt * tile_n * tile_k + (size_t)n * tile_k, wrow + off, len);
            }
        }
    }
    return packed;
}

// ---------------------------------------------------------------------------
// the operator
// ---------------------------------------------------------------------------

// Same front matter as Convolution::forward_int8 (quantize + padding), then
// one AME GEMM per (N-block x M-block) tile pair. Returns -1 when a gate
// rejects the shape so the caller falls back to the stock kernel.
int Convolution_riscv::forward_int8_xsai(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_to_int8(bottom_blob, bottom_blob_unbordered, bottom_blob_int8_scales, opt_g);
        if (bottom_blob_unbordered.empty())
            return -100;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets (same as the base kernel)
    int* space_ofs = (int*)fastMalloc(maxk * sizeof(int));
    if (!space_ofs)
        return -100;
    {
        int p1 = 0;
        int p2 = 0;
        const int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // ---- AME shape gates ----
    const int K = maxk * channels;
    const int k_tiles = (K + 63) / 64;
    const int K_pad = k_tiles * 64;
    if (K_pad == 0 || K_pad > kXsaiAmeKMax)
    {
        fastFree(space_ofs);
        return -1; // too deep for one packed tile; stock kernel handles it
    }
    // Prepacked B must exist (create_pipeline builds it under the same gates;
    // dynamic weights / tiny oc*K layers therefore stay on the stock kernel).
    if (weight_int8_ame_data.empty())
    {
        fastFree(space_ofs);
        return -1;
    }

    const xsai_ame_tile_spec* spec = xsai_ame_tile_spec_get();
    const int tile_m = spec->tile_m;
    const int tile_n = spec->tile_n;
    const int tile_k = spec->tile_k;
    int8_t* atile = g_xsai_atile;
    int8_t* btile = g_xsai_btile;
    int32_t* ctile = g_xsai_ctile;

    const bool use_int8_requantize = int8_scale_term > 100;
    const size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
    {
        fastFree(space_ofs);
        return -100;
    }

    const int M_total = outw * outh;

    // Scratch for one logical A row (K bytes + k zero-pad to K_pad). The AME
    // tile layout splits each logical row across the k_tiles 64-byte blocks
    // (block kt row r at atile + kt*tile_m*tile_k + r*tile_k), so rows are
    // built contiguously first, then redistributed per block.
    signed char* arow_buf = (signed char*)fastMalloc(K_pad);
    if (!arow_buf)
    {
        fastFree(space_ofs);
        return -100;
    }

    // B tile: one memcpy per N-block from the create_pipeline prepack
    // ([nb][k_tiles][tile_n][64] bytes, K tail zero-padded there).
    const signed char* prepacked = (const signed char*)weight_int8_ame_data.data;
    for (int j0 = 0; j0 < num_output; j0 += tile_n)
    {
        const int nn = (num_output - j0 < tile_n ? num_output - j0 : tile_n);
        const int b = j0 / tile_n;
        memcpy(btile, prepacked + (size_t)b * k_tiles * tile_n * 64,
               (size_t)k_tiles * tile_n * 64);

        for (int i0 = 0; i0 < M_total; i0 += tile_m)
        {
            const int mm = (M_total - i0 < tile_m ? M_total - i0 : tile_m);

            // A tile: row r = output pixel i0+r, logical column q*maxk+k.
            memset(atile, 0, (size_t)tile_m * K_pad);
            for (int r = 0; r < mm; r++)
            {
                const int p = i0 + r;
                const int py = p / outw;
                const int px = p - py * outw;
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const signed char* sptr = m.row<const signed char>(py * stride_h) + px * stride_w;
                    signed char* dst = arow_buf + (size_t)q * maxk;
                    for (int k = 0; k < maxk; k++)
                    {
                        dst[k] = sptr[space_ofs[k]];
                    }
                }
                memset(arow_buf + K, 0, (size_t)(K_pad - K));
                for (int kt = 0; kt < k_tiles; kt++)
                {
                    memcpy(atile + (size_t)kt * tile_m * tile_k + (size_t)r * tile_k,
                           arow_buf + (size_t)kt * 64, 64);
                }
            }

            const uint64_t cyc0 = xsai_ame_rdcycle();
            xsai_ame_gemm_tile_kloop(atile, (ptrdiff_t)tile_m * tile_k,
                                     btile, (ptrdiff_t)tile_n * tile_k, k_tiles, ctile,
                                     tile_m, tile_k, tile_n);
            // qemu-user cold-start: the process's first kloop once produced a
            // mismatching result; re-run and keep the second (no-op when warm).
            if (!g_xsai_warmed)
            {
                g_xsai_warmed = true;
                xsai_ame_gemm_tile_kloop(atile, (ptrdiff_t)tile_m * tile_k,
                                         btile, (ptrdiff_t)tile_n * tile_k, k_tiles, ctile,
                                         tile_m, tile_k, tile_n);
            }
            g_xsai_ame_cycles += xsai_ame_rdcycle() - cyc0;
            g_xsai_ame_calls++;

            // epilogue: same arithmetic as the base kernel, reading ctile with
            // the tile_n row stride.
            for (int n = 0; n < nn; n++)
            {
                const int p_oc = j0 + n;
                float scale_in;
                if (weight_data_int8_scales[p_oc] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p_oc]);

                signed char* outc = use_int8_requantize ? (signed char*)top_blob.channel(p_oc) : 0;
                float* outf = use_int8_requantize ? 0 : (float*)(unsigned char*)top_blob.channel(p_oc);

                for (int r = 0; r < mm; r++)
                {
                    const int sum = ctile[(size_t)r * tile_n + n];
                    float sumfp32 = sum * scale_in;
                    if (bias_term)
                        sumfp32 += bias_data[p_oc];
                    sumfp32 = activation_ss(sumfp32, activation_type, activation_params);
                    if (use_int8_requantize)
                    {
                        const float scale_out = top_blob_int8_scales[0];
                        outc[i0 + r] = xsai_f2i8(sumfp32 * scale_out);
                    }
                    else
                    {
                        outf[i0 + r] = sumfp32;
                    }
                }
            }
        }
    }

    fastFree(arow_buf);
    fastFree(space_ofs);
    return 0;
}

} // namespace ncnn

#endif // NCNN_CONVOLUTION_INT8_XSAI_H
