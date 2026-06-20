#include <stdio.h>
#include <cuda.h>

__global__ void pandemicKernel(float *infection, float *rate, float *projected,
 int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        projected[i] = infection[i] * rate[i];
    }
}

int main()
{
    int N = 1024;

    float h_infection[N];
    float h_rate[N];
    float h_projected[N];

  
    for (int i = 0; i < N; i++)
    {
        h_infection[i] = (float)(i + 1);
        h_rate[i] = 1.0f + (i % 5) * 0.1f;
    }

    float *d_infection, *d_rate, *d_projected;

   
    cudaMalloc((void**)&d_infection, N * sizeof(float));
    cudaMalloc((void**)&d_rate, N * sizeof(float));
    cudaMalloc((void**)&d_projected, N * sizeof(float));

    cudaMemcpy(d_infection, h_infection, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rate, h_rate, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    pandemicKernel<<<numBlocks, blockSize>>>(d_infection, d_rate, d_projected, N);

    cudaMemcpy(h_projected, d_projected, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)    
    {
        printf("City %d: Infections = %.2f, Rate = %.2f, Projected = %.2f\n",
               i, h_infection[i], h_rate[i], h_projected[i]);
    }

    cudaFree(d_infection);
    cudaFree(d_rate);
    cudaFree(d_projected);

    return 0;
}
