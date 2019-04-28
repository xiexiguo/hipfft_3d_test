#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK 1
#define VERBOSE 1

void cuda_error_check (cudaError_t cudaError, int line) {
  size_t      free, total;
  cudaError_t cErr2;

  cErr2 = cudaGetLastError();
  if (cudaError != cudaSuccess || cErr2 != cudaSuccess) {
    printf("CUDA RT Error line %d\n", line);
    printf("CUDA RT1 Error: %s\n", cudaGetErrorString(cudaError));
    printf("CUDA RT2 Error: %s\n", cudaGetErrorString(cErr2));
    cudaMemGetInfo(&free,&total);
    printf("Free: %zu , Total: %zu\n", free, total);
    fflush(stdout);
    exit(-1);
  }
}


void cufft_error_check (cufftResult_t cufftError, int line)
{
  size_t      free, total;
  cudaError_t cErr2;

  cErr2 = cudaGetLastError();
  if (cufftError != CUFFT_SUCCESS || cErr2 != cudaSuccess) {
    printf("CUDA FFT Error line: %d \n", line);
    switch (cufftError) {
      case CUFFT_INVALID_PLAN:   printf("CUDA FFT1 Error (CUFFT_INVALID_PLAN)\n"); break;
      case CUFFT_ALLOC_FAILED:   printf("CUDA FFT1 Error (CUFFT_ALLOC_FAILED)\n"); break;
      case CUFFT_INVALID_VALUE:  printf("CUDA FFT1 Error (CUFFT_INVALID_VALUE)\n"); break;
      case CUFFT_INTERNAL_ERROR: printf("CUDA FFT1 Error (CUFFT_INTERNAL_ERROR)\n"); break;
      case CUFFT_EXEC_FAILED:    printf("CUDA FFT1 Error (CUFFT_EXEC_FAILED)\n"); break;
      case CUFFT_INVALID_SIZE:   printf("CUDA FFT1 Error (CUFFT_INVALID_SIZE)\n"); break;
      default: printf("CUDA FFT1 Error (--unimplemented--) %d %d\n", cufftError, cErr2); break;
    }
    printf("CUDA FFT2 Error %s \n", cudaGetErrorString(cErr2));
    cudaMemGetInfo(&free,&total);
    printf("Free: %zu , Total: %zu\n", free, total);
    fflush(stdout);
    exit(-1);
  }
}


void fftcu_plan3d_z(cufftHandle  &plan, const int *n) 
{

  cufftResult_t cErr;

  if (VERBOSE) printf("FFT 3D (%d-%d-%d)\n", n[0], n[1], n[2]);
  cErr = cufftPlan3d(&plan, n[2], n[1], n[0], CUFFT_Z2Z);
  if (CHECK) cufft_error_check(cErr, __LINE__);
  //cErr = cufftSetStream(plan, cuda_stream);
  //if (CHECK) cufft_error_check(cErr, __LINE__);
}



int main()
{

  // file store data before and after fft
  FILE *f_data_in, *f_data_out;
  int n[3], fsign;
  int lmem;
  double *data_in_h, *data_out_h, *data_fft_h;
  cufftDoubleComplex *data_d;
  cufftHandle   plan;
  cufftResult_t cErr;
  cudaError_t  cuErr;

  // read data before and after fft
  f_data_in = fopen("data_in.dat", "r+");
  f_data_out = fopen("data_out.dat", "r+");

  // read data size
  fscanf(f_data_in,"%d %d %d\n", &n[0], &n[1], &n[2]);
  fscanf(f_data_in,"%d\n", &fsign);
  if(VERBOSE) printf("FFT 3D size: %d %d %d\n", n[0], n[1], n[2]);
  lmem = n[0] * n[1] * n[2];

  data_in_h = (double*) malloc(2*lmem*sizeof(double));
  data_out_h = (double*) malloc(2*lmem*sizeof(double));
  data_fft_h = (double*) malloc(2*lmem*sizeof(double));

  for(int i=0; i< 2*lmem; ++i)
  {
        fscanf(f_data_in,"%lf\n", &data_in_h[i]);
        fscanf(f_data_out,"%lf\n", &data_out_h[i]);
  }
  if(VERBOSE)
  {
      for(int i=0; i<10; ++i) printf("%.16e %.16e\n", data_in_h[i], data_out_h[i]);
  }

  cuErr = cudaMalloc(&data_d, 2*lmem*sizeof(double));
  if(CHECK) cuda_error_check(cuErr, __LINE__);
  cudaMemcpy(data_d, data_in_h, 2*lmem*sizeof(double), cudaMemcpyHostToDevice);

  fftcu_plan3d_z(plan, n);
  if ( fsign < 0  ) {
    cErr = cufftExecZ2Z(plan, data_d, data_d, CUFFT_INVERSE);
    if (CHECK) cufft_error_check(cErr, __LINE__);
  }
  else {
    cErr = cufftExecZ2Z(plan, data_d, data_d, CUFFT_FORWARD);
    if (CHECK) cufft_error_check(cErr, __LINE__);
  }

  cudaMemcpy(data_fft_h, data_d, 2*lmem*sizeof(double), cudaMemcpyDeviceToHost);

  for(int i=0; i<10; ++i) printf("%.16e %.16e %.16e\n", data_in_h[i], data_out_h[i],data_fft_h[i]);
  free(data_in_h);
  free(data_out_h);
  free(data_fft_h);
  fclose(f_data_in);
  fclose(f_data_out);
}
