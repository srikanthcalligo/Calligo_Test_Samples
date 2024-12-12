
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
//Device imports
# include <host_api.hpp>
# include <impl/device/device.hpp>

using namespace std;
using namespace tt;
using namespace tt::tt_metal;

float sdot ( int n, float dx[], int incx, float dy[], int incy )

//****************************************************************************80
//
//  Purpose:
//
//    SDOT forms the dot product of two float vectors.
//
//  Discussion:
//
//    This routine uses unrolled loops for increments equal to one.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    23 February 2006
//
//  Author:
//
//    C++ version by John Burkardt
//
//  Reference:
//
//    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
//    LINPACK User's Guide,
//    SIAM, 1979,
//    ISBN13: 978-0-898711-72-1,
//    LC: QA214.L56.
//
//    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
//    Basic Linear Algebra Subprograms for Fortran Usage,
//    Algorithm 539,
//    ACM Transactions on Mathematical Software,
//    Volume 5, Number 3, September 1979, pages 308-323.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vectors.
//
//    Input, float DX[*], the first vector.
//
//    Input, int INCX, the increment between successive entries in DX.
//
//    Input, float DY[*], the second vector.
//
//    Input, int INCY, the increment between successive entries in DY.
//
//    Output, float SDOT, the sum of the product of the corresponding
//    entries of DX and DY.
//
{
  float dtemp;
  int i;
  int ix;
  int iy;
  int m;

  dtemp = 0.0;

  if ( n <= 0 )
  {
    return dtemp;
  }
//
//  Code for unequal increments or equal increments
//  not equal to 1.
//
  if ( incx != 1 || incy != 1 )
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( - n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( - n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      dtemp = dtemp + dx[ix] * dy[iy];
      ix = ix + incx;
      iy = iy + incy;
    }
  }
//
//  Code for both increments equal to 1.
//
  else
  {
    m = n % 5;

    for ( i = 0; i < m; i++ )
    {
      dtemp = dtemp + dx[i] * dy[i];
    }

    for ( i = m; i < n; i = i + 5 )
    {
      dtemp = dtemp + dx[i  ] * dy[i  ]
                    + dx[i+1] * dy[i+1]
                    + dx[i+2] * dy[i+2]
                    + dx[i+3] * dy[i+3]
                    + dx[i+4] * dy[i+4];
    }

  }

  return dtemp;
}


int main(int argc, char **argv)

//****************************************************************************80
//
//  Purpose:
//
//    TEST06 demonstrates SDOT.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    22 February 2006
//
//  Author:
//
//    John Burkardt
//
{
# define N 5
# define LDA 10
# define LDB 7
# define LDC 6

  float a[LDA*LDA];
  float b[LDB*LDB];
  float c[LDC*LDC];
  int i;
  int j;
  float sum1;
  float x[N];
  float y[N];

  cout << "\n";
  cout << "  SDOT computes the dot product of vectors.\n";
  cout << "\n";

  for ( i = 0; i < N; i++ )
  {
    x[i] = ( float ) ( i + 1 ) * 0.56624334;
  }

  for ( i = 0; i < N; i++ )
  {
    y[i] =  -( float ) ( i + 1 ) * 0.2345244;
  }
#if 0
  for ( i = 0; i < N; i++ )
  {
    for ( j = 0; j < N; j++ )
    {
      a[i+j*LDA] = ( float ) ( i + 1 + j + 1 );
    }
  }

  for ( i = 0; i < N; i++ )
  {
    for ( j = 0; j < N; j++ )
    {
      b[i+j*LDB] = ( float ) ( ( i + 1 ) - ( j + 1 ) );
    }
  }
#endif
#if 1
  //Run the below call on the device

  constexpr uint32_t device_id = 0;
  Device *device = CreateDevice(device_id);
  CommandQueue& cq = device->command_queue();
  Program program  = CreateProgram();
  constexpr CoreCoord core = {0,0};
  constexpr uint32_t single_tile_size = 2 * 32 * 32;
  uint32_t num_tiles = N / tt::constants::TILE_HEIGHT;

  uint32_t dev_incx = 1;
  uint32_t dev_incy = 1;
  uint32_t n_size = N;

  InterleavedBufferConfig dram_buffer_config{
    .device = device,
    .size = single_tile_size,
    .page_size = single_tile_size,
    .buffer_type = tt::tt_metal::BufferType::DRAM
    };

  std::shared_ptr<Buffer> srcx_data = CreateBuffer(dram_buffer_config);
  std::shared_ptr<Buffer> srcy_data = CreateBuffer(dram_buffer_config);
  std::shared_ptr<Buffer> dist_data = CreateBuffer(dram_buffer_config);

  auto srcx_noc_coords = srcx_data->noc_coordinates();
  auto srcy_noc_coords = srcy_data->noc_coordinates();
  auto dist_noc_coords = dist_data->noc_coordinates();

  uint32_t srcx_xcoord = srcx_noc_coords.x;
  uint32_t srcx_ycoord = srcx_noc_coords.y;
  uint32_t srcy_xcoord = srcy_noc_coords.x;
  uint32_t srcy_ycoord = srcy_noc_coords.y;
  uint32_t dist_xcoord = dist_noc_coords.x;
  uint32_t dist_ycoord = dist_noc_coords.y;  

  /*std::vector<bfloat16> srcx(N);
  std::vector<bfloat16> srcy(N);
  for(int i=0; i< N ; i++){
    srcx[i] = bfloat16(x[i]);
    srcy[i] = bfloat16(y[i]);
  }*/

  //std::vector<uint32_t> srcx_uint32 = pack_bfloat16_vec_into_uint32_vec(srcx);
  //std::vector<uint32_t> srcy_uint32 = pack_bfloat16_vec_into_uint32_vec(srcy);

  std::vector<uint32_t> srcx_uint32(N);
  std::vector<uint32_t> srcy_uint32(N);

  /*
  std::vector<uint32_t> srcx_uint32(2);// = {14, 16};
  float a1 = 14.5678654;
  float b1 = 16.3456776;
  srcx_uint32[0] = *(uint32_t*)&a1;
  srcx_uint32[1] = *(uint32_t*)&b1;
  //std::vector<uint32_t> srcy_uint32 = {1,8};
  std::vector<uint32_t> srcy_uint32(2);// = {14, 16};
  float a11 = 1.7654567;
  float b11 = 8.7654345;
  srcy_uint32[0] = *(uint32_t*)&a11;
  srcy_uint32[1] = *(uint32_t*)&b11;
  
  printf("x= %u\n", srcx_uint32[0]);
  */

  for(int i=0; i< N ; i++){
    srcx_uint32[i] = *(uint32_t *)&x[i];
    //printf("x= %u\n", srcx_uint32[i]);
    srcy_uint32[i] = *(uint32_t *)&y[i];
    //printf("y= %u\n", srcy_uint32[i]);
  }
  //print_vec_of_uint32_as_packed_bfloat16(srcx_uint32, srcx_uint32.size());

  EnqueueWriteBuffer(cq, srcx_data, srcx_uint32, false);
  EnqueueWriteBuffer(cq, srcy_data, srcy_uint32, false);

  /* Use L1 circular buffers to set input buffers */
  constexpr uint32_t src0_cb_index = CB::c_in0;
  CircularBufferConfig cb_src0_config = CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
  CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

  constexpr uint32_t src1_cb_index = CB::c_in1;
  CircularBufferConfig cb_src1_config = CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, single_tile_size);
  CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

  /* Specify data movement kernel for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/lanl_dgemm_sample_calligo/kernels/data/reader_writer_sdot_in_riscv.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(program, binary_reader_kernel_id, core,
        {
            srcx_data->address(),
            srcy_data->address(),
            dist_data->address(),
            srcx_xcoord,
            srcx_ycoord,
            srcy_xcoord,
            srcy_ycoord,
            dist_xcoord,
            dist_ycoord,
            dev_incx,
            dev_incy,
            n_size,
        }
    );

    EnqueueProgram(cq, program, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dist_data, result_vec, true);
    float bb = *(float*)&result_vec[0]; 
    //printf("Result = %d  %u\n", result_vec[0], result_vec[0]);
    CloseDevice(device);

    printf("  (Metalium)Dot product of X and Y is %.7f\n", bb);
//End of Device execution
#endif
  sum1 = sdot ( N, x, 1, y, 1 );

  cout << "\n";
  cout << "  (Host)Dot product of X and Y is " << setprecision(8) << sum1 << "\n";
//
//  To multiply a ROW of a matrix A times a vector X, we need to
//  specify the increment between successive entries of the row of A:
//
#if 0
  sum1 = sdot ( N, a+1+0*LDA, LDA, x, 1 );

  cout << "\n";
  cout << "  Product of row 2 of A and X is " << sum1 << "\n";
//
//  Product of a column of A and a vector is simpler:
//
  sum1 = sdot ( N, a+0+1*LDA, 1, x, 1 );

  cout << "\n";
  cout << "  Product of column 2 of A and X is " << sum1 << "\n";
//
//  Here's how matrix multiplication, c = a*b, could be done
//  with SDOT:
//
  for ( i = 0; i < N; i++ )
  {
    for ( j = 0; j < N; j++ )
    {
      c[i+j*LDC] = sdot ( N, a+i, LDA, b+0+j*LDB, 1 );
    }
  }

  cout << "\n";
  cout << "  Matrix product computed with SDOT:\n";
  cout << "\n";
  for ( i = 0; i < N; i++ )
  {
    for ( j = 0; j < N; j++ )
    {
      cout << "  " << setw(14) << c[i+j*LDC];
    }
    cout << "\n";
  }
#endif
  return 0;
# undef N
# undef LDA
# undef LDB
# undef LDC
}
