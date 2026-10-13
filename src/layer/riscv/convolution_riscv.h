// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION_RISCV_H
#define LAYER_CONVOLUTION_RISCV_H

#include "convolution.h"

namespace ncnn {

class Convolution_riscv : public Convolution
{
public:
    Convolution_riscv();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ZFH
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

#if NCNN_INT8 && NCNN_XSAI_AME
    // XSAI AME int8 path (convolution_int8_xsai.h). Returns -1 when a
    // capability/shape gate rejects the layer; the caller then keeps the
    // stock scalar kernel. Requires the create_pipeline prepack below.
    int forward_int8_xsai(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

#if NCNN_XSAI_AME
    // XSAI AME bf16 path (convolution_bf16_xsai.h): fp32 in/out, bf16 AME
    // matmul inside (MNN BF16FunctionsRISCV convention). Runtime-gated by
    // env XSAI_NCNN_AME_BF16 (off by default: it changes float precision).
    int forward_bf16_xsai(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    Layer* activation;

    int nT;

    Mat weight_data_tm;
    Mat weight_sgemm_data;
    Mat weight_winograd23_data;
    Mat weight_winograd43_data;
    Mat weight_winograd63_data;

#if NCNN_INT8 && NCNN_XSAI_AME
    // create_pipeline prepack for the AME int8 path:
    // [nb][k_tiles][tile_n][64] bytes of K-padded weight rows.
    Mat weight_int8_ame_data;
#endif
#if NCNN_XSAI_AME
    // create_pipeline prepack for the AME bf16 path:
    // [nb][k_tiles][tile_n][32] uint16 of K-padded bf16 weight rows.
    Mat weight_bf16_ame_data;
#endif

    // fp16
    Mat bias_data_fp16;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_RISCV_H
