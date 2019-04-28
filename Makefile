CUDA_INC = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib -lcudart -lcufft
NVCC = /usr/local/cuda/bin/nvcc

HIP_INC = -I/opt/rocm/include
HIP_LIBS = -L/opt/rocm/lib -lhip_hcc -lrocfft -lrocfft-device
HIPCC = /opt/rocm/bin/hipcc
hipfft3d_test: hipfft3d_test.hip.cpp
	$(HIPCC) $(HIP_INC) $(HIP_LIBS) -o $@ $<

cufft3d_test: cufft3d_test.cu
	$(NVCC) $(CUDA_INC) $(CUDA_LIBS) -o $@ $<

clean:
	rm hipfft3d_test cufft3d_test -f
