INC = -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib -lcudart -lcufft
NVCC = /usr/local/cuda/bin/nvcc
cufft3d_test: cufft3d_test.cu
	$(NVCC) $(INC) $(LIBS) -o $@ $<
