#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <internal/datatype.hpp>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>


namespace gptoss {

class MatMulKernelTester {
public:
    MatMulKernelTester() { }

    MatMulKernelTester(const MatMulKernelTester&) = delete;
    MatMulKernelTester(MatMulKernelTester&&) = delete;
    MatMulKernelTester& operator=(const MatMulKernelTester&) = delete;
    MatMulKernelTester& operator=(MatMulKernelTester&&) = delete;

    [[nodiscard]]
    MatMulKernelTester& num_rows(std::uint32_t num_rows) {
        num_rows_ = num_rows;
        return *this;
    }

    std::uint32_t num_rows() const {
        return num_rows_;
    }

    [[nodiscard]]
    MatMulKernelTester& num_cols(std::uint32_t num_cols) {
        num_cols_ = num_cols;
        return *this;
    }

    std::uint32_t num_cols() const {
        return num_cols_;
    }

    [[nodiscard]]
    MatMulKernelTester& num_tokens(std::uint32_t num_tokens) {
        num_tokens_ = num_tokens;
        return *this;
    }

    std::uint32_t num_tokens() const {
        return num_tokens_;
    }

    [[nodiscard]]
    MatMulKernelTester& threadgroup_size(std::size_t threadgroup_size) {
        threadgroup_size_ = threadgroup_size;
        return *this;
    }

    std::size_t threadgroup_size() const {
        return threadgroup_size_;
    }

    void Validate(std::uint32_t vec_size) const {
        ASSERT_NE(num_rows(), 0);
        ASSERT_NE(num_cols(), 0);
        ASSERT_EQ(num_cols() % vec_size, 0);
        ASSERT_NE(num_tokens(), 0);
        ASSERT_NE(threadgroup_size(), 0);
    }

    void TestF32_BF16W() const {
        Validate(/*vec_size=*/4);

        metal::CommandBuffer command_buffer{command_queue_};
        metal::Buffer input_buffer{device_, num_tokens() * num_cols() * sizeof(float)};
        metal::Buffer weight_buffer{device_, num_rows() * num_cols() * sizeof(gptoss_bfloat16)};
        metal::Buffer bias_buffer{device_, num_rows() * sizeof(gptoss_bfloat16)};
        metal::Buffer output_buffer{device_, num_tokens() * num_rows() * sizeof(float)};

        command_buffer.encode_launch_f32_fill_random(
            f32_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/input_buffer,
            /*output_offset=*/0,
            num_tokens() * num_cols(), kSeed, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        command_buffer.encode_launch_bf16_fill_random(
            bf16_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/weight_buffer,
            /*output_offset=*/0,
            num_rows() * num_cols(), kSeed + 1, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        command_buffer.encode_launch_bf16_fill_random(
            bf16_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/bias_buffer,
            /*output_offset=*/0,
            num_rows(), kSeed + 2, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        Check(gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul(
                command_buffer.handle(),
                f32_bf16w_matmul_fn_.handle(),
                /*threadgroup_size=*/threadgroup_size(),
                input_buffer.handle(),
                /*input_offset=*/0,
                weight_buffer.handle(),
                /*weight_offset=*/0,
                bias_buffer.handle(),
                /*bias_offset=*/0,
                output_buffer.handle(),
                /*output_offset=*/0,
                num_tokens(),
                num_cols(),
                num_rows()),
            "gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul");

        command_buffer.commit();
        command_buffer.wait_completion();

        const float* input_ptr = static_cast<const float*>(input_buffer.ptr());
        const gptoss_bfloat16* weight_ptr = static_cast<const gptoss_bfloat16*>(weight_buffer.ptr());
        const gptoss_bfloat16* bias_ptr = static_cast<const gptoss_bfloat16*>(bias_buffer.ptr());
        const float* output_ptr = static_cast<const float*>(output_buffer.ptr());
        for (size_t t = 0; t < num_tokens(); t++) {
            for (size_t r = 0; r < num_rows(); r++) {
                double ref_sum = upcast<double>(bias_ptr[r]);
                for (size_t c = 0; c < num_cols(); c++) {
                    const double ref_weight = upcast<double>(weight_ptr[r * num_cols() + c]);
                    const double input_value = upcast<double>(input_ptr[t * num_cols() + c]);
                    ref_sum = std::fma(input_value, ref_weight, ref_sum);
                }
                ASSERT_NEAR(upcast<double>(output_ptr[t * num_rows() + r]), ref_sum, std::abs(ref_sum) * 1.0e-5)
                    << "token " << t;
            }
        }
    }

private:
    static constexpr std::uint64_t kSeed{UINT64_C(1019827666124465388)};
    static constexpr std::size_t kFillRandomMaxThreadgroups = 10;
    static constexpr float fp4e2m1_to_fp32[16] = {
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };

    metal::Device device_{};
    metal::CommandQueue command_queue_{device_};
    metal::Library library_{device_};
    metal::Function f32_fill_random_fn_{library_, "gptoss_f32_fill_random"};
    metal::Function bf16_fill_random_fn_{library_, "gptoss_bf16_fill_random"};
    metal::Function f32_bf16w_matmul_fn_{library_, "gptoss_f32_bf16w_matmul"};
    std::uint32_t num_tokens_{1};
    std::uint32_t num_rows_{1};
    std::uint32_t num_cols_{32};
    std::size_t threadgroup_size_{32};
};

}  // namespace gptoss
