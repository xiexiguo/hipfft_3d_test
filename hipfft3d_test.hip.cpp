#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hipfft.h>
#include <fftw3.h>

#define CHECK 1
#define VERBOSE 0

using namespace std;

void hip_error_check (hipError_t hipError, int line)
{
  size_t      free, total;
  hipError_t cErr2;

  cErr2 = hipGetLastError();
  if (hipError != hipSuccess || cErr2 != hipSuccess) {
    printf("HIP RT Error line %d\n", line);
    printf("HIP RT1 Error: %s\n", hipGetErrorString(hipError));
    printf("HIP RT2 Error: %s\n", hipGetErrorString(cErr2));
    hipMemGetInfo(&free,&total);
    printf("Free: %zu , Total: %zu\n", free, total);
    fflush(stdout);
    exit(-1);
  }
}


void hipfft_error_check (hipfftResult_t hipfftError, int line)
{
  size_t      free, total;
  hipError_t cErr2;

  cErr2 = hipGetLastError();
  if (hipfftError != HIPFFT_SUCCESS || cErr2 != hipSuccess) {
    printf("HIP FFT Error line: %d \n", line);
    switch (hipfftError) {
      case HIPFFT_INVALID_PLAN:   printf("HIP FFT1 Error (HIPFFT_INVALID_PLAN)\n"); break;
      case HIPFFT_ALLOC_FAILED:   printf("HIP FFT1 Error (HIPFFT_ALLOC_FAILED)\n"); break;
      case HIPFFT_INVALID_VALUE:  printf("HIP FFT1 Error (HIPFFT_INVALID_VALUE)\n"); break;
      case HIPFFT_INTERNAL_ERROR: printf("HIP FFT1 Error (HIPFFT_INTERNAL_ERROR)\n"); break;
      case HIPFFT_EXEC_FAILED:    printf("HIP FFT1 Error (HIPFFT_EXEC_FAILED)\n"); break;
      case HIPFFT_INVALID_SIZE:   printf("HIP FFT1 Error (HIPFFT_INVALID_SIZE)\n"); break;
      default: printf("HIP FFT1 Error (--unimplemented--) %d %d\n", hipfftError, cErr2); break;
    }
    printf("HIP FFT2 Error %s \n", hipGetErrorString(cErr2));
    hipMemGetInfo(&free,&total);
    printf("Free: %zu , Total: %zu\n", free, total);
    fflush(stdout);
    exit(-1);
  }
}


void ffthip_plan3d_z(hipfftHandle  &plan, const int *n) 
{

  hipfftResult_t cErr;

  if (VERBOSE) printf("FFT 3D (%d-%d-%d)\n", n[0], n[1], n[2]);
  cErr = hipfftPlan3d(&plan, n[2], n[1], n[0], HIPFFT_Z2Z);
  if (CHECK) hipfft_error_check(cErr, __LINE__);
  //cErr = hipfftSetStream(plan, hipda_stream);
  //if (CHECK) hipfft_error_check(cErr, __LINE__);
}



int main()
{

  // file store data before and after fft
  FILE *f_data_in, *f_data_out;
  int n[3], fsign;
  int lmem;
  double *data_in_h, *data_out_h, *data_fft_h;
  hipfftDoubleComplex *data_d;
  hipfftHandle   plan;
  hipfftResult_t cErr;
  hipError_t  hipErr;

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

  // read data from file
  for(int i=0; i< 2*lmem; ++i)
  {
        fscanf(f_data_in,"%lf\n", &data_in_h[i]);
        fscanf(f_data_out,"%lf\n", &data_out_h[i]);
  }
  if(VERBOSE)
  {
      for(int i=0; i<10; ++i) printf("%.16e %.16e\n", data_in_h[i], data_out_h[i]);
  }

  // do fft on CPU fftw
  fftw_complex *fftw_cpu_in = (fftw_complex*) malloc(2*lmem*sizeof(double));  
  fftw_complex *fftw_cpu_out = (fftw_complex*) malloc(2*lmem*sizeof(double));  
  double *data_fft_cpu_in = (double*) fftw_cpu_in;
  double *data_fft_cpu_out = (double*) fftw_cpu_out;

  fftw_plan fftw_plan_cpu = fftw_plan_dft_3d(n[0], n[1], n[2], fftw_cpu_in, fftw_cpu_out, fsign, FFTW_EXHAUSTIVE);

  memcpy(fftw_cpu_in, data_in_h, 2*lmem*sizeof(double));

  fftw_execute(fftw_plan_cpu);

  printf("FFT 3d on CPU\n");
  for(int i=0; i<10; ++i) printf("%.16e %.16e %.16e\n", data_fft_cpu_in[i], data_out_h[i], data_fft_cpu_out[i]);  

  fftw_destroy_plan(fftw_plan_cpu);
  free(fftw_cpu_in);
  free(fftw_cpu_out);  

  // do fft on GPU
  hipErr = hipMalloc(&data_d, 2*lmem*sizeof(double));
  if(CHECK) hip_error_check(hipErr, __LINE__);
  hipMemcpy(data_d, data_in_h, 2*lmem*sizeof(double), hipMemcpyHostToDevice);

  ffthip_plan3d_z(plan, n);
  if ( fsign < 0  ) {
    cErr = hipfftExecZ2Z(plan, data_d, data_d, HIPFFT_BACKWARD);
    if (CHECK) hipfft_error_check(cErr, __LINE__);
  }
  else {
    cErr = hipfftExecZ2Z(plan, data_d, data_d, HIPFFT_FORWARD);
    if (CHECK) hipfft_error_check(cErr, __LINE__);
  }

  hipMemcpy(data_fft_h, data_d, 2*lmem*sizeof(double), hipMemcpyDeviceToHost);

  printf("FFT 3d on GPU\n");
  for(int i=0; i<10; ++i) printf("%.16e %.16e %.16e\n", data_in_h[i], data_out_h[i],data_fft_h[i]);
  free(data_in_h);
  free(data_out_h);
  free(data_fft_h);
  fclose(f_data_in);
  fclose(f_data_out);
}
