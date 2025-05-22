#pragma once  // 添加头文件保护
#include "ntt.cuh"    // 包含 TestDataType 定义文件
#include <string>     // 添加 string 支持
#include <vector>
using namespace std;

// 打印函数
template<typename T>
void print_array(const T *data, const std::string &title) {
    std::cout << "\n[" << title << "]\n前5个元素: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << data[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

#define VERIFY_RESULTS(output_ptr, ref_result, success_msg) \
do { \
    bool check = true; \
    for (int i = 0; i < BATCH; i++) { \
        check = check_result( \
            output_ptr + (i * parameters_2n.n), \
            ref_result[i].data(), \
            parameters_2n.n \
        ); \
        if (!check) { \
            std::cout << "第 " << i << " 个多项式验证失败" << std::endl; \
            break; \
        } \
    } \
    if (check) std::cout << success_msg << std::endl; \
} while(0)

// 使用示例 VERIFY_RESULTS(Output_Host, ntt_result, "NTT结果正确");

// 点积核
template <typename T>
__global__ void PointwiseMultiplyKernel(
    T* input1, 
    T* input2, 
    T* output, 
    Modulus<T> modulus, 
    int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        output[idx] = OPERATOR_GPU<T>::mult(input1[idx], input2[idx], modulus);
    }
}

// 点乘方法
template <typename T>
static void PointwiseMultiply(
    T* device_in1,
    T* device_in2,
    T* device_out,
    Modulus<T> modulus,
    int n,
    int batch = 1,
    cudaStream_t stream = 0)
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

        PointwiseMultiplyKernel<<<gridSize, blockSize, 0, stream>>>(
            device_in1, 
            device_in2, 
            device_out, 
            modulus, 
            n
        );
    cudaDeviceSynchronize();
    cudaGetLastError();
}

// 点积加法核函数
template <typename T>
__global__ void PointwiseAddKernel(
    T* input1, 
    T* input2, 
    T* output, 
    Modulus<T> modulus, 
    int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        output[idx] = OPERATOR_GPU<T>::add(input1[idx], input2[idx], modulus);
    }
}

// 点积减法核函数
template <typename T>
__global__ void PointwiseSubtractKernel(
    T* input1, 
    T* input2, 
    T* output, 
    Modulus<T> modulus, 
    int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        output[idx] = OPERATOR_GPU<T>::sub(input1[idx], input2[idx], modulus);
    }
}

// 点加方法
template <typename T>
static void PointwiseAdd(
    T* device_in1,
    T* device_in2,
    T* device_out,
    Modulus<T> modulus,
    int n,
    int batch = 1,  // 保留参数用于未来扩展
    cudaStream_t stream = 0)
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    PointwiseAddKernel<<<gridSize, blockSize, 0, stream>>>(
        device_in1, 
        device_in2, 
        device_out,
        modulus,
        n
    );
    cudaDeviceSynchronize();
    cudaGetLastError();
}

// 点减方法
template <typename T>
static void PointwiseSubtract(
    T* device_in1,
    T* device_in2,
    T* device_out,
    Modulus<T> modulus,
    int n,
    int batch = 1,  // 保留参数用于未来扩展
    cudaStream_t stream = 0)
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    PointwiseSubtractKernel<<<gridSize, blockSize, 0, stream>>>(
        device_in1, 
        device_in2, 
        device_out,
        modulus,
        n
    );
    cudaDeviceSynchronize();
    cudaGetLastError();
}

// // Karatsuba核函数
// template <typename T>
// __global__ void karatsuba_kernel(
//     const T* __restrict__ fi,
//     const T* __restrict__ fj,
//     const T* __restrict__ gi,
//     const T* __restrict__ gj,
//     const T* __restrict__ fg_i,
//     const T* __restrict__ fg_j,
//           T*       output,
//     Modulus<T> modulus,
//     int size) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= size) return;

//     // 读全局
//     auto a = fi[idx], b = fj[idx], c = gi[idx], d = gj[idx], e = fg_i[idx], f = fg_j[idx];
//     // (fi+fj)*(gi+gj) - fi*gi - fj*gj
//     T sum_f = OPERATOR_GPU<T>::add(a, b, modulus);
//     T sum_g = OPERATOR_GPU<T>::add(c, d, modulus);
//     T Karatsuba_prod = OPERATOR_GPU<T>::mult(sum_f, sum_g, modulus);
//     Karatsuba_prod = OPERATOR_GPU<T>::sub(Karatsuba_prod, e, modulus);
//     Karatsuba_prod = OPERATOR_GPU<T>::sub(Karatsuba_prod, f, modulus);
//     output[idx] = OPERATOR_GPU<T>::add(output[idx], Karatsuba_prod, modulus);
// }

// // Karatsuba
// template <typename T>  
// static void karatsuba(
//     const T* __restrict__ fi,
//     const T* __restrict__ fj,
//     const T* __restrict__ gi,
//     const T* __restrict__ gj,
//     const T* __restrict__ fg_i,
//     const T* __restrict__ fg_j,
//           T*       output,
//     Modulus<T> modulus,
//     int n,
//     cudaStream_t stream = 0)
// {
//     constexpr int blockSize = 256;
//     const int gridSize = (n + blockSize - 1) / blockSize;
//     karatsuba_kernel<T><<<gridSize, blockSize, 0, stream>>>(
//         fi, fj, gi, gj, fg_i, fg_j, output, modulus, n);
    
//     // 错误检查
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "karatsuba_step kernel launch failed: %s\n",
//                 cudaGetErrorString(err));
//     }
// }

// 合并核函数
template <typename T>
__global__ void merge_kernel(
    T* input, 
    T* output, 
    Modulus<T> modulus, 
    int size,
    int batch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        // result_merged[0] = result[0][0] + result[batch-1][1]
        output[idx] = OPERATOR_GPU<T>::add(input[idx], input[2 * size * (batch-1) + size + idx], modulus); 
        #pragma unroll
        for(int i = 1; i < batch; i++) {
            // result_merged[i] = result[i][0] + result[i-1][1]
            output[size * i + idx] = OPERATOR_GPU<T>::add(input[2 * size * i + idx], input[2 * size * (i-1) + size + idx], modulus);
        }
   
    }
}

// 合并方法
template <typename T>
static void merge(
    T* device_in,
    T* device_out,
    Modulus<T> modulus,
    int n,
    int batch,
    cudaStream_t stream = 0) 
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    merge_kernel<T><<<gridSize, blockSize, 0, stream>>>(device_in, device_out, modulus, n, batch);
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "merge kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

// Karatsuba核函数
template <typename T>
__global__ void karatsuba_kernel(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const T* __restrict__ input3,
          T*              output,
    Modulus<T> modulus,
    int size,
    int batch) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;


    #pragma unroll
    for (int i = 0; i < batch; i++)
    {   
        #pragma unroll
        for  (int j = i+1; j < batch; j++) 
        { 
            T a = input1[i * size + idx], b = input1[j * size + idx], 
              c = input2[i * size + idx], d = input2[j * size + idx], 
              e = input3[i * size + idx], f = input3[j * size + idx];

            // (fi+fj)*(gi+gj) - fi*gi - fj*gj
            T sum_f = OPERATOR_GPU<T>::add(a, b, modulus);
            T sum_g = OPERATOR_GPU<T>::add(c, d, modulus);
            T Karatsuba_prod = OPERATOR_GPU<T>::mult(sum_f, sum_g, modulus);
            Karatsuba_prod = OPERATOR_GPU<T>::sub(Karatsuba_prod, e, modulus);
            Karatsuba_prod = OPERATOR_GPU<T>::sub(Karatsuba_prod, f, modulus);
            output[(i+j) * size + idx] = OPERATOR_GPU<T>::add(output[(i+j) * size + idx], Karatsuba_prod, modulus);
        }
    }

}

// Karatsuba 
template <typename T>
static void karatsuba(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const T* __restrict__ input3,
          T*              output,
    Modulus<T> modulus,
    int n,
    int batch,
    cudaStream_t stream = 0) 
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    karatsuba_kernel<T><<<gridSize, blockSize, 0, stream>>>(input1, input2, input3, output, modulus, n, batch);
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "karatsuba_step kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

