#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <limits.h>

#define TOTAL_DEGREES 90
#define BINS_PER_DEG 4
#define THREADS_PER_BLK 640

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int NoofSim;

// histograms will be stored in the following
unsigned int *histogramDR, *histogramDD, *histogramRR;

// total no. of bins in the histogram
const int NO_OF_BINS = TOTAL_DEGREES * BINS_PER_DEG;

// value of one radian
const float ONE_RAD = 180.0f / M_PI;

// Kernel to calculate the angular separation
__global__ void calculateHistogramAngles(float *ra_1, float *decl_1, float *ra_2, float *decl_2, unsigned int *hist, int N){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use shared memory for partial histograms
    __shared__ unsigned int shared_hist[NO_OF_BINS];
    for (int i = threadIdx.x; i < NO_OF_BINS; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Use this condition so that shared_hist is always initialized by each thread
    // but only the valid threads (i.e. idx < N) contribute to the calculations
    if (idx < N){

        float a1 = ra_1[idx];
        float d1 = decl_1[idx];
        float a2, d2, angle;
        unsigned int bin;
        float sinfd1 = sinf(d1);
        float cosfd1 = cosf(d1);

        for (int j = 0; j < N; j++) {
            a2 = ra_2[j];
            d2 = decl_2[j];
            angle = sinfd1 * sinf(d2) + cosfd1 * cosf(d2) * cosf(a1 - a2);
            angle = fmaxf(-1.0f, fminf(angle, 1.0f)); // Clamp the angle
            angle = acosf(angle);
            bin = (int)(angle * ONE_RAD * 4.0f);

            // Update local histogram in shared memory
            atomicAdd(&shared_hist[bin % NO_OF_BINS], 1);
        }
    }
    __syncthreads();

    // Update global histogram
    for (int i = threadIdx.x; i < NO_OF_BINS; i += blockDim.x){
        atomicAdd(&hist[i], shared_hist[i]);
    }
}

int main(int argc, char *argv[])
{
    int noofblocks;
    int readdata(char *argv1, char *argv2);
    int getDevice(int deviceno);
    long int histogramDRsum = 0, histogramDDsum = 0, histogramRRsum = 0;
    double start, end, kerneltime;
    struct timeval _ttime;
    struct timezone _tzone;
    cudaError_t myError;

    if (argc != 3)
    {
        printf("Usage: a.out real_data random_data\n");
        return (-1);
    }

    if (getDevice(0) != 0)
        return (-1);

    if (readdata(argv[1], argv[2]) != 0)
        return (-1);

    // allocate memory on the GPU
    // Using unified memory
    cudaMallocManaged(&histogramDD, NO_OF_BINS * sizeof(unsigned int));
    cudaMallocManaged(&histogramDR, NO_OF_BINS * sizeof(unsigned int));
    cudaMallocManaged(&histogramRR, NO_OF_BINS * sizeof(unsigned int));

    // Initialize the memory to 0
    cudaMemset(histogramDD, 0, NO_OF_BINS);
    cudaMemset(histogramDR, 0, NO_OF_BINS);
    cudaMemset(histogramRR, 0, NO_OF_BINS);

    noofblocks = (NoofReal + THREADS_PER_BLK - 1) / THREADS_PER_BLK;

    int sharedMemSize = NO_OF_BINS * sizeof(unsigned int);
    cudaDeviceSynchronize();

    printf("# of blocks = %d , # threads in block = %d, # of total threads = %d \n", noofblocks, THREADS_PER_BLK, THREADS_PER_BLK * noofblocks);

    myError = cudaGetLastError();
    if (myError != cudaSuccess)
    {
        printf("ERROR: %s\n", cudaGetErrorString(myError));
        exit(-1);
    }
    else
    {
        printf("No error during initialization..\n");
    }

    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

    // copy data to the GPU -- Not needed as using unified memory

    // run the kernels on the GP
    // DD
    calculateHistogramAngles<<<noofblocks, THREADS_PER_BLK, sharedMemSize>>>(ra_real, decl_real, ra_real, decl_real, histogramDD, NoofReal);
    cudaDeviceSynchronize();

    myError = cudaGetLastError();
    if (myError != cudaSuccess)
    {
        printf("ERROR: %s\n", cudaGetErrorString(myError));
        exit(-1);
    }
    else
    {
        printf("No error during DD..\n");
    }

    
    // DR
    calculateHistogramAngles<<<noofblocks, THREADS_PER_BLK, sharedMemSize>>>(ra_real, decl_real, ra_sim, decl_sim, histogramDR, NoofReal);
    cudaDeviceSynchronize();

    myError = cudaGetLastError();
    if (myError != cudaSuccess)
    {
        printf("ERROR: %s\n", cudaGetErrorString(myError));
        exit(-1);
    }
    else
    {
        printf("No error during DR..\n");
    }

    // RR
    calculateHistogramAngles<<<noofblocks, THREADS_PER_BLK, sharedMemSize>>>(ra_sim, decl_sim, ra_sim, decl_sim, histogramRR, NoofSim);
    cudaDeviceSynchronize();

    myError = cudaGetLastError();
    if (myError != cudaSuccess)
    {
        printf("ERROR: %s\n", cudaGetErrorString(myError));
        exit(-1);
    }
    else
    {
        printf("No error during RR..\n");
    }
    
    for (int i = 0; i < NO_OF_BINS; ++i)
    {
        histogramDDsum += histogramDD[i];
    }
    printf("histogramDDsum = %ld\n", histogramDDsum);

    for (int i = 0; i < NO_OF_BINS; ++i)
    {
        histogramDRsum += histogramDR[i];
    }
    printf("histogramDRsum = %ld\n", histogramDRsum);

    for (int i = 0; i < NO_OF_BINS; ++i)
    {
        histogramRRsum += histogramRR[i];
    }
    printf("histogramRRsum = %ld\n", histogramRRsum);

    // copy the results back to the CPU -- Not needed as using unified memory

    // calculate omega values on the CPU
    float *omega = (float *)malloc(NO_OF_BINS * sizeof(float));
    memset(omega, 0.0f, NO_OF_BINS);

    for (int i = 0; i < NO_OF_BINS; i++)
    {
        if (histogramRR[i])
        {
            omega[i] = (float)(histogramDD[i] - 2.0f * histogramDR[i] + histogramRR[i]) / histogramRR[i];
        }
    }

    // Print first 5 values of the histograms
    for (int i = 0; i < 5; i++)
    {
        printf("%d -- %f: %d %d %d\n", i, omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);
    }

    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;
    kerneltime += end - start;
    printf("Kernel time: %lf\n", kerneltime);

    // Free Unified memory
    // Not required for unified memory
    
    return (0);
}

int readdata(char *argv1, char *argv2){
    int i, linecount;
    char inbuf[180];
    double ra, dec, phi, theta;
    FILE *infil;

    printf("   Assuming input data is given in arc minutes!\n");

    infil = fopen(argv1, "r");
    if (infil == NULL)
    {
        printf("Cannot open input file %s\n", argv1);
        return (-1);
    }

    // read the number of galaxies in the input file
    int announcednumber;
    if (fscanf(infil, "%d\n", &announcednumber) != 1)
    {
        printf(" cannot read file %s\n", argv1);
        return (-1);
    }
    linecount = 0;
    while (fgets(inbuf, 180, infil) != NULL)
        ++linecount;
    rewind(infil);

    if (linecount == announcednumber)
        printf("   %s contains %d galaxies\n", argv1, linecount);
    else
    {
        printf("   %s does not contain %d galaxies but %d\n", argv1, announcednumber, linecount);
        return (-1);
    }

    NoofReal = linecount;
    
    // Unified memory for real data
    cudaMallocManaged(&ra_real, NoofReal * sizeof(float));
    cudaMallocManaged(&decl_real, NoofReal * sizeof(float));

    // Initialize the memory to 0
    cudaMemset(ra_real, 0, NoofReal);
    cudaMemset(decl_real, 0, NoofReal);
    
    // skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL)
        return (-1);
    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv1);
            fclose(infil);
            return (-1);
        }
        // Convert ra and decl from arcmin to rad
        phi = ra / 60.0 * M_PI / 180.0;
        theta = dec / 60.0 * M_PI / 180.0;

        ra_real[i] = (float)phi;
        decl_real[i] = (float)theta;
        ++i;
    }

    fclose(infil);

    if (i != NoofReal)
    {
        printf("   Cannot read %s correctly\n", argv1);
        return (-1);
    }

    infil = fopen(argv2, "r");
    if (infil == NULL)
    {
        printf("Cannot open input file %s\n", argv2);
        return (-1);
    }

    if (fscanf(infil, "%d\n", &announcednumber) != 1)
    {
        printf(" cannot read file %s\n", argv2);
        return (-1);
    }
    linecount = 0;
    while (fgets(inbuf, 80, infil) != NULL)
        ++linecount;
    rewind(infil);

    if (linecount == announcednumber)
        printf("   %s contains %d galaxies\n", argv2, linecount);
    else
    {
        printf("   %s does not contain %d galaxies but %d\n", argv2, announcednumber, linecount);
        return (-1);
    }

    NoofSim = linecount;

    // Unified memory for simulated data
    cudaMallocManaged(&ra_sim, NoofSim * sizeof(float));
    cudaMallocManaged(&decl_sim, NoofSim * sizeof(float));

    // Initialize the memory to 0
    cudaMemset(ra_sim, 0, NoofSim);
    cudaMemset(decl_sim, 0, NoofSim);

    // skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL)
        return (-1);
    i = 0;
    while (fgets(inbuf, 80, infil) != NULL)
    {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
        {
            printf("   Cannot read line %d in %s\n", i + 1, argv2);
            fclose(infil);
            return (-1);
        }
        // Convert ra and decl from arcmin to rad
        phi = ra / 60.0 * M_PI / 180.0;
        theta = dec / 60.0 * M_PI / 180.0;

        ra_sim[i] = (float)phi;
        decl_sim[i] = (float)theta;
        ++i;
    }

    fclose(infil);

    if (i != NoofSim)
    {
        printf("   Cannot read %s correctly\n", argv2);
        return (-1);
    }

    printf("Real data: first and last\n%f %f \n%f %f\n", 
        ra_real[0]*60.0*180.0/M_PI, decl_real[0]*60.0*180.0/M_PI,
        ra_real[NoofReal-1]*60.0*180.0/M_PI, decl_real[NoofSim-1]*60.0*180.0/M_PI);

    printf("Synth data: first and last\n%f %f \n%f %f\n", 
        ra_sim[0]*60.0*180.0/M_PI, decl_sim[0]*60.0*180.0/M_PI,
        ra_sim[NoofReal-1]*60.0*180.0/M_PI, decl_sim[NoofSim-1]*60.0*180.0/M_PI);    
    return (0);
}

int getDevice(int deviceNo)
{

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("   Found %d CUDA devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128)
        return (-1);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("      Device %s                  device %d\n", deviceProp.name, device);
        printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem / 1000000000.0);
        printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
        printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate / 1000.0);
        printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
        printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
        printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
        printf("         maxGridSize                   =   %d x %d x %d\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("         concurrentKernels             =   ");
        if (deviceProp.concurrentKernels == 1)
            printf("     yes\n");
        else
            printf("    no\n");
        printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
        if (deviceProp.deviceOverlap == 1)
            printf("            Concurrently copy memory/execute kernel\n");
    }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if (device != 0)
        printf("   Unable to set device 0, using %d instead", device);
    else
        printf("   Using CUDA device %d\n\n", device);

    return (0);
}
