Steps to run the Testcases
**************************
Step 0: Clone the tt-metal repo

	git clone https://github.com/tenstorrent/tt-metal.git
	
Step 1 : Clone the Calligo's Test samples repo

	git clone https://github.com/srikanthcalligo/Calligo_Test_Samples.git
	
Step 2 : Copy lanl_dgemm_sample_calligo folder to /home/user/tt-metal/tt_metal/programming_examples directory

Step 3 : Add the following lines to CMakeLists.txt in the above specified directory.

		${CMAKE_CURRENT_SOURCE_DIR}/lanl_dgemm_sample_calligo/dgemm_base_application.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lanl_dgemm_sample_calligo/dgemm_metalium_application.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lanl_dgemm_sample_calligo/add_nums_in_compute.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lanl_dgemm_sample_calligo/sfpu_math_functions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lanl_dgemm_sample_calligo/cblas_sdot_in_riscv.cpp

Step 4 : Do the following steps to build and run the testcases

		cd /home/user/tt-metal
		
		ninja tests -C build
		
	    ./build/programming_examples/binary_names
