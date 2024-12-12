#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char *argv[]){
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    //Define and create buffer with Float data type

    constexpr uint32_t buf_size = 2*1024;

    tt_metal::BufferConfig buffer_config = {
            .device = device,
            .size = buf_size,
            .page_size = buf_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer1 = CreateBuffer(buffer_config);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer2 = CreateBuffer(buffer_config);

    auto src0_coord =  dram_buffer->noc_coordinates();
    auto src1_coord = dram_buffer1->noc_coordinates();
    auto dst_coord = dram_buffer2->noc_coordinates();

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = CB::c_in0;

    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        buf_size,
        {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, buf_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CB::c_in1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        buf_size,
        {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, buf_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );


    constexpr uint32_t dst_cb_index = tt::CB::c_out0;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        buf_size,
        {{dst_cb_index, tt::DataFormat::Float16_b}}).set_page_size(dst_cb_index, buf_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels

    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/lanl_dgemm_sample_calligo/kernels/data/read_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/lanl_dgemm_sample_calligo/kernels/data/write_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/lanl_dgemm_sample_calligo/kernels/compute/add_tiles_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );

    float val[2] = {4.7888999, 6.467876543};
    //float val = 16.0f;
    float val1[2] = {90.0, 3.2345678765};


    EnqueueWriteBuffer(cq, dram_buffer, val, false);
    EnqueueWriteBuffer(cq, dram_buffer1, val1, false);

    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y});
    //SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(),src0_coord.x, src0_coord.y});

    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), dst_coord.x, dst_coord.y});
    //, dram_buffer1->address()
    //SetRuntimeArgs(program, void_data_kernel_noc1, core, {dram_buffer1->address()});
    EnqueueProgram(cq, program, false);

    Finish(cq);

    printf("Hello, device 0, handle the data\n");

    vector<uint32_t> result;
    EnqueueReadBuffer(cq, dram_buffer2, result, true);

    float data_result = *(float*)&result[0];
    float data_result1 = *(float*)&result[1];

    //printf("Execution is done : Result = %u %f %f\n", result[0], data_result, data_result1);
    printf("Execution is done Metalium : Result[0] = %u %f  Expected = %f\n", result[0], data_result, val[0] + val1[0]);
    printf("Execution is done Metalium : Result[1] = %u %f  Expected = %f\n", result[1], data_result1, val[1] + val1[1]);

    CloseDevice(device);
    return 0;
}