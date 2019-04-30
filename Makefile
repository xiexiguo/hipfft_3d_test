CUDA_INC = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib -lcudart -lcufft
NVCC = /usr/local/cuda/bin/nvcc

HIP_INC = -I/opt/rocm/include
HIP_LIBS = -L/opt/rocm/lib -lrocfft
HIPCC = hipcc

FFTW = /home/xiexg/lib/fftw/3.3.8
FFTW_INC = -I$(FFTW)/include
FFTW_LIBS = -L$(FFTW)/lib -lfftw3

hipfft3d_test: hipfft3d_test.o
	$(HIPCC) $(FFTW_LIBS) $(HIP_LIBS) -o $@ $<

hipfft3d_test.o: hipfft3d_test.hip.cpp
	$(HIPCC) -c $(HIP_INC) $(FFTW_INC) -o $@ $<

cufft3d_test: cufft3d_test.cu
	$(NVCC) $(CUDA_INC) $(FFTW_INC) $(CUDA_LIBS) $(FFTW_LIBS) -o $@ $<

clean:
	rm hipfft3d_test cufft3d_test *.o -f
