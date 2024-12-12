
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLAS_LIB "nolib"

#ifdef USE_MKL
#include "mkl.h"
#define BLAS_LIB "mkl"
#endif

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define BLAS_LIB "cublas"
#endif

#ifdef USE_CUBLASXT
#include <cublasXt.h>
#include <cuda_runtime.h>
#define BLAS_LIB "cublasXt"
#endif

#ifdef USE_LIBSCI
#include <cblas.h>
#define BLAS_LIB "libsci"
#endif

#ifdef USE_LIBSCI_ACC
#include <libsci_acc.h>
#define BLAS_LIB "libsci_acc"
#endif


#ifdef USE_CBLAS
#include "cblas.h"
#define BLAS_LIB "cblas"
#endif

#ifdef USE_ESSL
#include "essl.h"
#define BLAS_LIB "essl"
#endif

#define DGEMM_RESTRICT __restrict__



// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "impl/device/device.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;


void golden_matmul(std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& output,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j+ (i * N);
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                c_f += float_tmp;
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}


void matmul_multi_core(vector<bfloat16>& a, vector<bfloat16>& b, vector<bfloat16>& output, bool bcast_batch,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B, Device* device) {

    /*
    * Setup program to execute along with its buffers and kernels to use
    */
    CommandQueue& cq = device->command_queue();
    Program program{};

    /*
    * Multi-Core prep
    */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // From tt_metal/common/constants.hpp
    auto num_output_tiles_total = (M * N) / TILE_HW;

    /*
     * Use a helper function to deduce the splits needed to co-operatively do
     * this matmul.
     */
    auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    /*
    * Extracting Matrix dimensions from input/output vectors
    */
    // C = A*B
    // MN = MK*KN
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    /*
    * Create DRAM Buffers for input and output vectors
    * Writing data from input vectors to source buffers
    */
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = 2 * 32 * 32;

    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    tt_metal::InterleavedBufferConfig dram_config_A{
                    .device= device,
                    .size = dram_buffer_A_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_B{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_C{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_A);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config_B);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_C);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    /*
    * Config of Circular Buffer in the device L1
    * input tiles count is = 2 because it's single tile process, and double-buffer
    */
    uint32_t src0_cb_index = CB::c_in0; //0
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1; // 1
    CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    /*
    * Compile time arguments
    */
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t) output_cb_index, (uint32_t)dst_is_dram};

    /*
    * Create Kernels (Reader, Writer, Compute)
    */
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args_group_1 = {
        1, // B
        1, // Mt
        Kt, // Kt
        num_output_tiles_per_core_group_1 // Nt
    }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

    auto matmul_multi_core_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_1}
    );

    if (!core_group_2.ranges().empty()) {
        vector<uint32_t> compute_args_group_2 = {
            1, // B
            1, // Mt
            Kt, // Kt
            num_output_tiles_per_core_group_2 // Nt
        }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

        auto matmul_multi_core_kernel_group_2_id = tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_2}
        );
    }

    /*
    * Kernels - Runtime arguments
    */
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){

        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program, reader_id, core,
            {src0_addr,
            src1_addr,
            Mt,
            Kt,
            Nt,
            MtKt,
            KtNt,
            B,
            uint32_t(bcast_batch),
            num_tiles_written,
            num_output_tiles_per_core,
            MtNt }
        );
        tt_metal::SetRuntimeArgs(
            program,
            writer_id,
            core,
            {dst_addr,
            num_output_tiles_per_core,
            num_tiles_written }
        );
        num_tiles_written += num_output_tiles_per_core;
    }

    /* Launch program & read in output buffer result into the host vector */
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
}

void matmul_single_core(std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& output, bool bcast_batch,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B, Device* device) {

    /*
    * Setup program to execute along with its buffers and kernels to use
    * Core range is just single core
    */
    CommandQueue& cq = device->command_queue();
    Program program{};
    CoreRange core({0, 0}, {0, 0});

    /*
    * EXtracting Matrix dimensions from input/output vectors
    */
    // C = A*B
    // MN = MK*KN
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    /*
    * Create DRAM Buffers for input and output vectors
    * Writing data from input vectors to source buffers
    */
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = 2 * 32 * 32;

    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    /* DRAM buffer size = input full size */
    /* limiting page_size = single tile size; to allow DRAM channels interleaving */

    tt_metal::InterleavedBufferConfig dram_config_A{
                    .device= device,
                    .size = dram_buffer_A_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_B{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_C{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_A);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config_B);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_C);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    /*
    * Config of Circular Buffer in the device L1
    * input tiles count is = 2 because it's single tile process, and double-buffer
    */
    uint32_t src0_cb_index = CB::c_in0; //0
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1; // 1
    CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    /*
    * Compile time arguments
    */
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    /*
    * Create Kernels (Reader, Writer, Compute)
    */
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        B, // B
        Mt, // Mt
        Kt, // Kt
        Nt // Nt
    };
    auto matmul_single_core_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args}
    );

    /*
    * Kernels - Runtime arguments
    */
    tt_metal::SetRuntimeArgs(
        program, reader_id, core,
        {src0_addr, src1_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, uint32_t(bcast_batch ? 1 : 0)}
    );

    tt_metal::SetRuntimeArgs(
        program, writer_id, core,
        {dst_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
    );

    /* Launch program & read in output buffer result into the host vector */
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
}


///////////////////////////////////////



// ------------------------------------------------------- //
// Function: get_seconds
//
// Vendor may modify this call to provide higher resolution
// timing if required
// ------------------------------------------------------- //
double get_seconds() {
	struct timeval now;
	gettimeofday(&now, NULL);

	const double seconds = (double) now.tv_sec;
	const double usec    = (double) now.tv_usec;

	return seconds + (usec * 1.0e-6);
}

// ------------------------------------------------------- //
// Function: main
//
// Modify only in permitted regions (see comments in the
// function)
// ------------------------------------------------------- //
int main(int argc, char* argv[]) {

	// ------------------------------------------------------- //
	// DO NOT CHANGE CODE BELOW
	// ------------------------------------------------------- //
	try {

	//Accelerator setup

	constexpr int device_id = 0;
	Device *device_0 = CreateDevice(device_id);

	uint32_t N = 256;

	//constexpr uint32_t 
	//size_t N = 256;
	uint32_t repeats = 8;
	size_t block_size = 0;

    float alpha = 1.0;
    float beta  = 1.0;

	if(argc > 1) {
		N = atoi(argv[1]);
		printf("Matrix size input by command line: %u\n", N);

		if(argc > 2) {
			repeats = atoi(argv[2]);

			if(repeats < 4) {
				fprintf(stderr, "Error: repeats must be at least 4, setting is: %d\n", repeats);
				exit(-1);
			}

			printf("Repeat multiply %d times.\n", repeats);

            if(argc > 3) {
                alpha = (float) atof(argv[3]);
                if(argc > 4) {
                    beta = (float) atof(argv[4]);
                    if(argc > 5) block_size = atoi(argv[5]);
                }
            }
		} else {
			printf("Repeat multiply defaulted to %d\n", repeats);
		}
	} else {
		printf("Matrix size defaulted to %u\n", N);
	}

	if(N < 128) {
		printf("Error: N (%u) is less than 128, the matrix is too small.\n", N);
		exit(-1);
	}
    
    const size_t matrixsize = sizeof(float) * N * N;
	if (block_size == 0) block_size = N/2;

    printf("Alpha =    %.2f\n", alpha);
    printf("Beta  =    %.2f\n", beta);
    printf("BlockSize  =    %zu\n", block_size);
	printf("Allocating Matrices...\n");

	float* DGEMM_RESTRICT matrixA = (float*) malloc(matrixsize);
	float* DGEMM_RESTRICT matrixB = (float*) malloc(matrixsize);
	float* DGEMM_RESTRICT matrixC = (float*) malloc(matrixsize);

	printf("Allocation complete, populating with values...");

	size_t i, j, k, r;
    double start, end, time_taken, time_section, time_taken_acc, time_taken_acc_mc;

    start = get_seconds();
    #pragma omp parallel for private(i,j,k)
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
            k=i*N + j;
			matrixA[k] = 2.0;
			matrixB[k] = 0.5;
			matrixC[k] = 0.0; //changed 1.0
		}
	}


	//Create Device data
	uint32_t M1 = N;
	uint32_t N1 = N;
	uint32_t K1 = N;
	constexpr uint32_t B = 1;

	uint32_t M1t = M1 / TILE_HEIGHT;
	uint32_t N1t = N1 / TILE_WIDTH;
	uint32_t K1t = K1 / TILE_WIDTH;

	constexpr uint32_t single_tile_size = 2 * 32 * 32;

	uint32_t dram_buffer_A_size = single_tile_size * M1t * K1t;
	uint32_t dram_buffer_B_size = single_tile_size * K1t * N1t;
	uint32_t dram_buffer_C_size = single_tile_size * M1t * N1t;

	vector<bfloat16> a_matrix(dram_buffer_A_size / sizeof(bfloat16));
	vector<bfloat16> b_matrix(dram_buffer_B_size / sizeof(bfloat16));
	vector<bfloat16> c_matrix(dram_buffer_C_size / sizeof(bfloat16));

	for(i = 0; i< M1 ; i++){
		for(j = 0; j < N1 ; j++){
			k = i * N1 + j;
			a_matrix[k] = bfloat16(matrixA[k]);
			b_matrix[k] = bfloat16(matrixB[k]);
		}
	}
	tilize(a_matrix, M1, K1);
	tilize(b_matrix, K1, N1);

#if 1
#if defined(USE_CUBLAS)
    // Create Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);
	printf("-- CUDA!!\nAllocating and transferring values...");
    double *dMatrixA, *dMatrixB, *dMatrixC;
    cudaMalloc((void **)&dMatrixA, matrixsize);
    cudaMalloc((void **)&dMatrixB, matrixsize);
    cudaMalloc((void **)&dMatrixC, matrixsize);

    cudaMemcpy(dMatrixA, matrixA, matrixsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dMatrixB, matrixB, matrixsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dMatrixC, matrixC, matrixsize, cudaMemcpyHostToDevice);
#endif

#ifdef USE_CUBLASXT
// Create CublasXt Handle and select all available devices.
// You don't want to use explicit device memory here because it needs
// to be distributed across all devices and cudaMalloc only assigns
// to the current device.
    int *devices = NULL;
    cublasXtHandle_t handle;
    int device_count, blockdim;
    cudaGetDeviceCount(&device_count);
    devices = (int *)malloc(sizeof(int) * device_count);
    cublasXtCreate(&handle);
    for (int i=0; i<device_count; i++) devices[i] = i;
    cublasXtDeviceSelect(handle, device_count, devices);
    cublasXtSetPinningMemMode(handle, CUBLASXT_PINNING_ENABLED);
    cublasXtSetBlockDim(handle, block_size);
    cublasXtGetBlockDim(handle, &blockdim);
    printf("CUBLASXT has block dim: %d\n", blockdim);
#endif

    end = get_seconds();
    time_section = (end - start);
    printf(" %g seconds\n", time_section);

	printf("Performing multiplication...\n");
	printf("Using Blas Type: %s\n", BLAS_LIB);
	printf("Iteration #:\n");

	start = get_seconds();

	// ------------------------------------------------------- //
	// VENDOR NOTIFICATION: START MODIFIABLE REGION
	//
	// Vendor is able to change the lines below to call optimized
	// DGEMM or other matrix multiplication routines. Do *NOT*
	// change any lines above this statement.
	// ------------------------------------------------------- //

	double sum = 0;

	// Repeat multiple times
	for(r = 0; r < repeats; r++) {
#if defined(USE_MKL) || defined(USE_CBLAS) || defined(USE_LIBSCI)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, alpha, matrixA, N, matrixB, N, beta, matrixC, N);
#elif defined(USE_CUBLAS)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dMatrixA, N, dMatrixB, N,
                     &beta, dMatrixC, N);
        cudaDeviceSynchronize();
#elif defined(USE_CUBLASXT)
        cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, matrixA, N, matrixB, N,
                     &beta, matrixC, N);
        cudaDeviceSynchronize();
#elif defined(USE_ESSL) || defined(USE_LIBSCI_ACC)
        dgemm('N', 'N',
            N, N, N, alpha, matrixA, N, matrixB, N, beta, matrixC, N);
#else
        #pragma omp parallel for private(sum, j, k)
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				sum = 0;

				for(k = 0; k < N; k++) {
					sum += matrixA[i*N + k] * matrixB[k*N + j];
					//matrixC[i*N + j] += matrixA[i*N + k] * matrixB[k*N + j];
				}
                matrixC[i*N + j] = (sum) + (matrixC[i*N + j]);
				//matrixC[i*N + j] = (alpha * sum) + (beta * matrixC[i*N + j]);
			}
		}
#endif

		if ( r%10 == 0 ) {
			printf("%zu, ", r);
			fflush(stdout); 
		}
	}
	printf("\n");

#if defined(USE_CUBLAS)
    cudaMemcpy(matrixC, dMatrixC, matrixsize, cudaMemcpyDeviceToHost);
#endif

	// ------------------------------------------------------- //
	// VENDOR NOTIFICATION: END MODIFIABLE REGION
	// ------------------------------------------------------- //

	// ------------------------------------------------------- //
	// DO NOT CHANGE CODE BELOW
	// ------------------------------------------------------- //

	end = get_seconds();
    time_taken = (end - start);

	start = get_seconds();
	// Repeat multiple times
	for(r = 0; r < repeats; r++) {
		//Accelerator Execution
		matmul_single_core(a_matrix, b_matrix, c_matrix, false, M1, N1, K1, B, device_0);
	}

	end = get_seconds();
    time_taken_acc = (end - start);

	start = get_seconds();
	// Repeat multiple times
	for(r = 0; r < repeats; r++) {
		//Accelerator Execution
		matmul_multi_core(a_matrix, b_matrix, c_matrix, false, M1, N1, K1, B, device_0);
	}

	end = get_seconds();
    time_taken_acc_mc = (end - start);

#ifdef USE_CUBLAS
    cublasDestroy(handle);
    cudaFree(dMatrixA);
    cudaFree(dMatrixB);
    cudaFree(dMatrixC);
    cudaDeviceSynchronize();
#endif

#ifdef USE_CUBLASXT
    cublasXtDestroy(handle);
    free(devices);
#endif

	printf("Calculating matrix check...");

	float final_sum = 0;
	float final_sum_acc = 0.0f;//On Accelerator
	float count     = 0;
    start = get_seconds();

	#pragma omp parallel for reduction(+:final_sum, count) private(i,j)
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			final_sum += matrixC[i*N + j];
			count += 1.0;
		}
	}
	
	
	untilize(a_matrix, M1, K1);//Untilizing the C matrix

	#pragma omp parallel for reduction(+:final_sum_acc, count) private(i,j)
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			final_sum_acc += c_matrix[i*N + j].to_float();
			count += 1.0;
		}
	}

    end = get_seconds();
    time_section = (end - start);
    printf(" %g seconds\n", time_section);

	double matrix_memory = (3 * matrixsize);

	printf("\n");
	printf("===============================================================\n");

	printf("Final Sum is:         %f\n", (final_sum / (count * repeats)));
	printf("Accelerator Final Sum is:         %f\n", (final_sum_acc / (count)));
	printf("Memory for Matrices:  %.0f MB\n",
		(matrix_memory / (1024 * 1024)));

    double N_dbl = (double) N;

	printf("Multiply time:                    %.6g seconds\n", time_taken);
	printf("Accelerator Multiply time:        %.6g seconds\n", time_taken_acc);
	printf("Accelerator Multiply time(Multicore):        %.6g seconds\n", time_taken_acc_mc);

	// O(N**3) elements each with one add and three multiplies
    	// (alpha, beta and A_i*B_i).
	double flops_computed = (N_dbl * N_dbl * 2.0 * (double)repeats)*(N_dbl+1.0);
    double total_time = ( flops_computed / time_taken) / 1.0e9;
	double total_time_single = ( flops_computed / time_taken_acc) / 1.0e9;
	double total_time_multi = ( flops_computed / time_taken_acc_mc) / 1.0e9;

	printf("FLOPs computed:       %.0g\n", flops_computed);
	printf("GFLOP/s rate:                             %.8g GF/s\n", (total_time));
	printf("GFLOP/s rate(Acc Single Cores)  :         %.8g GF/s\n", (total_time_single));
	printf("GFLOP/s rate(Acc Multiple Cores):         %.8g GF/s\n", (total_time_multi));

	printf("===============================================================\n");
	printf("\n");

	free(matrixA);
	free(matrixB);
	free(matrixC);

	CloseDevice(device_0); //Accelerator Device close

#endif
	}
	catch ( const std::exception &e){
		tt::log_error(tt::LogTest, "Failed with Error\n");
		tt::log_error(tt::LogTest, "{}", e.what());
		throw;
	}
	return 0;
}