// convolution_bf16_xsai.h — XSAI AME bf16 convolution for the riscv port.
//
// MNN BF16FunctionsRISCV convention ported to NCNN: the layer keeps fp32
// storage end-to-end (bytes == 4 — activations, weights source, bias, output
// stay fp32) and only the AME matmul operands are bf16 (matmulBytes == 2):
// packing truncates fp32 -> bf16 ((uint16_t)(bits >> 16), matching MNN's
// _fp32ToBf16), the AME accumulates in fp32, C is written back as fp32.
//
//   C[M,N] = A[M,K] x B[N,K]^T    (bf16 operands, fp32 accumulator)
//     M = outw*outh output pixels, N = num_output, K = channels*maxk
//
// Layout contract with ame_xsa.c::xsai_ame_bf16_gemm_tile_kloop:
//   one k-block is 32 bf16 elements (64 bytes) per row; k_tiles blocks of
//   tile_m x 32 / tile_n x 32 bf16 (k zero-padded); C is tile_m x tile_n fp32
//   with row stride tile_n. msettilek takes ELEMENTS (32), mla/mlb take
//   BYTES (64).
//
// Gate: env XSAI_NCNN_AME_BF16 == "1" (separate from the int8 env on purpose
// — this path changes float precision, so the official fp32 tests and the
// int8 gates must keep it off) + tile spec + shape limits + create_pipeline
// prepack. Any rejection returns -1 and the caller keeps the stock kernel.
//
// fp16-weight models ride the same path: with NCNN_ZFH off the layer's
// support_fp16_storage is false and NCNN casts fp16 inputs to fp32 upstream
// (fp16 -> fp32 -> bf16, the ggml GGML_AME_F16_BF16 route).
#ifndef NCNN_CONVOLUTION_BF16_XSAI_H
#define NCNN_CONVOLUTION_BF16_XSAI_H

#include "mat.h"
#include "fused_activation.h"

#include "ame_xsa.h"
#include "xsai_alloc.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

namespace ncnn {

// Max K (bf16 elements) a single dispatch may pack: covers conv shapes with
// ic*kx*ky up to 8192; larger K stays on the stock kernel. Buffer per tile =
// tile_m(128) x 8192 x 2B = 2 MiB.
static constexpr int kXsaiAmeBf16KMax = 8192;

static uint16_t* g_xsai_bf16_atile = 0;
static uint16_t* g_xsai_bf16_btile = 0;
static float* g_xsai_bf16_ctile = 0;
static bool g_xsai_bf16_warmed = false;
static unsigned long long g_xsai_bf16_ame_calls = 0;
static unsigned long long g_xsai_bf16_ame_cycles = 0;

static void xsai_bf16_print_summary()
{
    fprintf(stderr, "[xsai-ncnn] bf16 ame calls=%llu cycles=%llu\n",
            g_xsai_bf16_ame_calls, g_xsai_bf16_ame_cycles);
}

static bool xsai_ame_bf16_tiles_init()
{
    if (g_xsai_bf16_atile != 0)
        return true;
    g_xsai_bf16_atile = (uint16_t*)xsai_malloc((size_t)128 * kXsaiAmeBf16KMax * sizeof(uint16_t));
    g_xsai_bf16_btile = (uint16_t*)xsai_malloc((size_t)128 * kXsaiAmeBf16KMax * sizeof(uint16_t));
    g_xsai_bf16_ctile = (float*)xsai_malloc((size_t)128 * 128 * sizeof(float));
    if (g_xsai_bf16_atile == 0 || g_xsai_bf16_btile == 0 || g_xsai_bf16_ctile == 0)
    {
        fprintf(stderr, "[xsai-ncnn] bf16 fatal: xsai_malloc failed for tiles\n");
        return false;
    }
    if (!xsai_pool_phys_contiguous() && !xsai_alloc_host_test_mode())
    {
        fprintf(stderr, "[xsai-ncnn] bf16 fatal: xsai pool not physically contiguous\n");
        return false;
    }
    atexit(xsai_bf16_print_summary);
    return true;
}

static bool xsai_conv_bf16_ame_on()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* env = getenv("XSAI_NCNN_AME_BF16");
        const int env_on = (env != 0 && env[0] == '1' && env[1] == '\0') ? 1 : 0;
        const int spec_ok = (xsai_ame_tile_spec_get()->enabled != 0) ? 1 : 0;
        cached = (env_on && spec_ok && xsai_ame_bf16_tiles_init()) ? 1 : 0;
        if (cached)
        {
            const xsai_ame_tile_spec* spec = xsai_ame_tile_spec_get();
            fprintf(stderr, "[xsai-ncnn] AME bf16 conv enabled (tile %dx32x%d)\n",
                    spec->tile_m, spec->tile_n);
        }
    }
    return cached == 1;
}

// Truncating fp32 -> bf16 (matches MNN _fp32ToBf16 so references compare
// like with like).
static inline uint16_t xsai_f32_to_bf16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// ---------------------------------------------------------------------------
// create_pipeline prepack: B tile blocks for the whole weight matrix.
//
// Stored as [nb][k_tiles][tile_n][32] uint16 where nb indexes tile_n-sized
// output-channel blocks (rows beyond num_output zero-filled). forward()
// memcpy's one block into the pool btile per N-block — no per-call fp32->bf16
// conversion or gather. Returns an empty Mat when a gate rejects the layer.
// ---------------------------------------------------------------------------
static Mat xsai_conv_bf16_prepack_weight(const Mat& weight_data, int num_output, int K, int tile_n)
{
    // k 块宽度 = 探测到的 tile_k（字节）/ 2，不是常量 32：
    // tlenb/trlenb 有多种配置，写死会和发射端 msettilek 对不上。
    const int tke = xsai_ame_tile_spec_get()->tile_k / (int)sizeof(uint16_t);
    if (tke <= 0)
        return Mat();
    const int k_tiles = (K + tke - 1) / tke;
    if (K <= 0 || k_tiles * tke > kXsaiAmeBf16KMax)
        return Mat();
    if ((long long)num_output * K < (long long)xsai_ame_min_macs())
        return Mat();

    const int nb = (num_output + tile_n - 1) / tile_n;
    Mat packed(nb * k_tiles * tile_n * tke, (size_t)2u);
    if (packed.empty())
        return Mat();
    uint16_t* dst = (uint16_t*)packed.data;
    memset(dst, 0, packed.total() * 2);

    const int K_pad = k_tiles * tke;
    for (int b = 0; b < nb; b++)
    {
        uint16_t* blk = dst + (size_t)b * k_tiles * tile_n * tke;
        const int oc0 = b * tile_n;
        const int nn = (num_output - oc0 < tile_n) ? num_output - oc0 : tile_n;
        for (int n = 0; n < nn; n++)
        {
            const float* wrow = (const float*)weight_data + (size_t)(oc0 + n) * K;
            for (int k = 0; k < K; k++)
            {
                blk[(size_t)(k / tke) * tile_n * tke + (size_t)n * tke + (k % tke)] = xsai_f32_to_bf16(wrow[k]);
            }
        }
    }
    (void)K_pad;
    return packed;
}

// ---------------------------------------------------------------------------
// the operator: fp32 in / fp32 out, bf16 AME matmul inside
// ---------------------------------------------------------------------------
int Convolution_riscv::forward_bf16_xsai(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;
    const int maxk = kernel_w * kernel_h;
    const int K = maxk * channels;
    const int tke = xsai_ame_tile_spec_get()->tile_k / (int)sizeof(uint16_t);
    if (tke <= 0)
        return -1;
    const int k_tiles = (K + tke - 1) / tke;
    const int K_pad = k_tiles * tke;

    // gates: prepack must exist (built by create_pipeline under the same
    // spec) and K within one dispatch
    if (weight_bf16_ame_data.empty() || K_pad > kXsaiAmeBf16KMax)
        return -1;

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

    top_blob.create(outw, outh, num_output, 4u, opt.blob_allocator);
    if (top_blob.empty())
    {
        fastFree(space_ofs);
        return -100;
    }

    const xsai_ame_tile_spec* spec = xsai_ame_tile_spec_get();
    const int tile_m = spec->tile_m;
    const int tile_n = spec->tile_n;
    uint16_t* atile = g_xsai_bf16_atile;
    uint16_t* btile = g_xsai_bf16_btile;
    float* ctile = g_xsai_bf16_ctile;

    const int M_total = outw * outh;
    const int nb = weight_bf16_ame_data.w / (k_tiles * tile_n * tke);
    const uint16_t* prepacked = (const uint16_t*)weight_bf16_ame_data.data;

    // per-pixel logical row scratch (bf16), redistributed into 32-element
    // k-blocks afterwards
    uint16_t* arow_buf = (uint16_t*)fastMalloc(K_pad * sizeof(uint16_t));
    if (!arow_buf)
    {
        fastFree(space_ofs);
        return -100;
    }

    for (int b = 0; b < nb; b++)
    {
        const int j0 = b * tile_n;
        const int nn = (num_output - j0 < tile_n) ? num_output - j0 : tile_n;
        memcpy(btile, prepacked + (size_t)b * k_tiles * tile_n * tke,
               (size_t)k_tiles * tile_n * tke * sizeof(uint16_t));

        for (int i0 = 0; i0 < M_total; i0 += tile_m)
        {
            const int mm = (M_total - i0 < tile_m) ? M_total - i0 : tile_m;

            // A tile: row r = output pixel i0+r, logical column q*maxk+k
            memset(atile, 0, (size_t)tile_m * K_pad * sizeof(uint16_t));
            for (int r = 0; r < mm; r++)
            {
                const int p = i0 + r;
                const int py = p / outw;
                const int px = p - py * outw;
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const float* sptr = m.row(py * stride_h) + px * stride_w;
                    uint16_t* dst = arow_buf + (size_t)q * maxk;
                    for (int k = 0; k < maxk; k++)
                    {
                        dst[k] = xsai_f32_to_bf16(sptr[space_ofs[k]]);
                    }
                }
                memset(arow_buf + K, 0, (size_t)(K_pad - K) * sizeof(uint16_t));
                for (int kt = 0; kt < k_tiles; kt++)
                {
                    memcpy(atile + (size_t)kt * tile_m * tke + (size_t)r * tke,
                           arow_buf + (size_t)kt * tke, tke * sizeof(uint16_t));
                }
            }

            const uint64_t cyc0 = xsai_ame_rdcycle();
            xsai_ame_bf16_gemm_tile_kloop(atile, (ptrdiff_t)tile_m * tke * sizeof(uint16_t),
                                          btile, (ptrdiff_t)tile_n * tke * sizeof(uint16_t),
                                          k_tiles, 2, ctile, tile_m, tile_n);
            // qemu-user cold-start insurance (same as the int8 path)
            if (!g_xsai_bf16_warmed)
            {
                g_xsai_bf16_warmed = true;
                xsai_ame_bf16_gemm_tile_kloop(atile, (ptrdiff_t)tile_m * tke * sizeof(uint16_t),
                                              btile, (ptrdiff_t)tile_n * tke * sizeof(uint16_t),
                                              k_tiles, 2, ctile, tile_m, tile_n);
            }
            g_xsai_bf16_ame_cycles += xsai_ame_rdcycle() - cyc0;
            g_xsai_bf16_ame_calls++;

            // epilogue: fp32 accumulator + bias + activation, fp32 output
            for (int n = 0; n < nn; n++)
            {
                const int p_oc = j0 + n;
                float* outf = (float*)(unsigned char*)top_blob.channel(p_oc);
                const float biasv = bias_term ? bias_data[p_oc] : 0.f;
                for (int r = 0; r < mm; r++)
                {
                    float value = ctile[(size_t)r * tile_n + n] + biasv;
                    value = activation_ss(value, activation_type, activation_params);
                    outf[i0 + r] = value;
                }
            }
        }
    }

    fastFree(arow_buf);
    fastFree(space_ofs);
    return 0;
}

} // namespace ncnn

#endif // NCNN_CONVOLUTION_BF16_XSAI_H
