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
