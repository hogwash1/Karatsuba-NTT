#pragma once  // 添加头文件保护
#include "ntt.cuh"    // 包含 TestDataType 定义文件
#include <string>     // 添加 string 支持
#include <vector>
using namespace std;

// 打印函数
template<typename T>
void print_array(const T *data, const std::string &title) {
    return ;
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
// template <typename T>
// __global__ void merge_kernel(
//     T* input, 
//     T* output, 
//     Modulus<T> modulus, 
//     int size,
//     int batch)
// {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx < size) {
//         // result_merged[0] = result[0][0] + result[batch-1][1]
//         output[idx] = OPERATOR_GPU<T>::add(input[idx], input[2 * size * (batch-1) + size + idx], modulus); 
//         #pragma unroll
//         for(int i = 1; i < batch; i++) {
//             // result_merged[i] = result[i][0] + result[i-1][1]
//             output[size * i + idx] = OPERATOR_GPU<T>::add(input[2 * size * i + idx], input[2 * size * (i-1) + size + idx], modulus);
//         }
   
//     }
// }

template <typename T>
__global__ void merge_kernel(
    T* input, 
    T* output, 
    Modulus<T> modulus, 
    int size
    )
{
    const int batch = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // result_merged[i] = result[i][0] + result[i-1][1]
    output[size * batch + idx] = OPERATOR_GPU<T>::add(input[2 * size * batch + idx], input[2 * size * ((batch + gridDim.y - 1)% gridDim.y) + size + idx], modulus);
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
    dim3 grid((n + blockSize - 1) / blockSize, batch);  // 二维网格
    merge_kernel<T><<<grid, blockSize, 0, stream>>>(device_in, device_out, modulus, n);
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
    int size
    ) 
{
    // extern __shared__ char shared_memory_typed[];
    // T* shared_mem = reinterpret_cast<T*>(shared_memory_typed);

    // const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int batch = gridDim.y;

    // if (idx >= size) return;

    // 每个block处理一个i值的计算
    if(idx >= size || i >= batch) return;


    {
        // // 共享内存布局:
        // // [0, size-1] : 当前i的input1 (a)
        // // [size, 2*size-1] : 当前i的input2 (c)
        // // [2*size, 3*size-1] : 当前i的input3 (e)
        // T* a_shared = shared_mem;
        // T* c_shared = shared_mem + size;
        // T* e_shared = shared_mem + 2 * size;
        // a_shared[tid] = input1[i * size + idx];
        // c_shared[tid] = input2[i * size + idx];
        // e_shared[tid] = input3[i * size + idx];
        // __syncthreads();
        // T a = a_shared[tid];
        // T c = c_shared[tid];
        // T e = e_shared[tid];
    }

    T a = input1[i * size + idx], 
      c = input2[i * size + idx], 
      e = input3[i * size + idx];
    
    for  (int j = i+1; j < batch; j++) 
    { 
        T  b = input1[j * size + idx], 
           d = input2[j * size + idx], 
           f = input3[j * size + idx];
        // (fi+fj)*(gi+gj) - fi*gi - fj*gj
        T sum_f = OPERATOR_GPU<T>::add(a, b, modulus);
        T sum_g = OPERATOR_GPU<T>::add(c, d, modulus);
        T Karatsuba_prod = OPERATOR_GPU<T>::mult(sum_f, sum_g, modulus);
        Karatsuba_prod = OPERATOR_GPU<T>::sub(Karatsuba_prod, e, modulus);
        Karatsuba_prod = OPERATOR_GPU<T>::sub(Karatsuba_prod, f, modulus);
        // output[(i+j) * size + idx] = OPERATOR_GPU<T>::add(output[(i+j) * size + idx], Karatsuba_prod, modulus);
        // 获取目标内存地址的指针
        T* address = &output[(i+j) * size + idx];

        // 使用 atomicCAS 实现原子操作
        T old_val, new_val;
        do {
            old_val = *address; // 读取当前值
            new_val = OPERATOR_GPU<T>::add(old_val, Karatsuba_prod, modulus); // 计算新值（带模运算）
        } while (atomicCAS(address, old_val, new_val) != old_val); // 原子替换
        // output[(i+j) * size + idx] = shared_memory[(i+j) * size + idx];
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
    dim3 grid((n + blockSize - 1) / blockSize, batch);  // 二维网格

    // size_t shared_size = 3 * n * sizeof(T);  // 仅需存储3个size长度的数组
    karatsuba_kernel<T><<<grid, blockSize, 0, stream>>>(input1, input2, input3, output, modulus, n);
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "karatsuba_step kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

// // 合并对角项和非对角项核函数
// template <typename T>
// __global__ void merge_diag_non_diag_kernel(
//     T* diag_term,
//     T* other_term, 
//     T* output, 
//     Modulus<T> modulus, 
//     int size,
//     int batch)
// { 
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx < size) {
//         #pragma unroll
//         for(int i = 0; i < batch; i++) {
//             // other_term[2i] = other_term[2i] + diagonal_term[i]
//             output[2 * i * size + idx] = OPERATOR_GPU<T>::add(other_term[2 * i * size + idx], diag_term[i * size + idx], modulus);
//         }
//     }
// }

// 合并对角项和非对角项核函数
template <typename T>
__global__ void merge_diag_non_diag_kernel(
    T* diag_term,
    T* other_term, 
    T* output, 
    Modulus<T> modulus, 
    int size
    )
{ 
    const int batch = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // other_term[2i] = other_term[2i] + diagonal_term[i]
    output[2 * batch * size + idx] = OPERATOR_GPU<T>::add(other_term[2 * batch * size + idx], diag_term[batch * size + idx], modulus);
}

// 合并对角项和非对角项
template <typename T>
static void merge_diag_non_diag(
    T* diag_term,
    T* other_term,
    T* output,
    Modulus<T> modulus,
    int n,
    int batch,
    cudaStream_t stream = 0)
{
    constexpr int blockSize = 256;
    dim3 grid((n + blockSize - 1) / blockSize, batch);  // 二维网格
    merge_diag_non_diag_kernel<T><<<grid, blockSize, 0, stream>>>(diag_term, other_term, output, modulus, n);
}


