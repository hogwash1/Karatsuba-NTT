#include <cstdlib>    // 标准库头文件
#include <random>     // 随机数生成库
#include "ntt.cuh"    // GPU-NTT 库核心头文件
#include "lib/helper.cuh"

#define DEFAULT_MODULUS  // 启用默认模数模式

using namespace std;
using namespace gpuntt;  // 使用 GPU-NTT 库的命名空间

// 全局参数声明
int LOGN;   // NTT变换的log2长度（实际长度N=2^LOGN）
int BATCH;  // 批量处理的多项式数量

// 选择测试数据类型（64位版本）
typedef Data64 TestDataType;  // 定义用于测试的数据类型（32/64位可切换）

int main(int argc, char* argv[]) {
    // 初始化CUDA设备
    CudaDevice();  // 自定义CUDA设备初始化函数（可能包含错误检查）

    // 设置CUDA设备
    int device = 0;
    cudaSetDevice(device);  // 选择0号GPU设备
    
    // 获取设备属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "最大网格尺寸: " 
            << prop.maxGridSize[0] << " x "
            << prop.maxGridSize[1] << " x " 
            << prop.maxGridSize[2] << std::endl;


    // 处理命令行参数
    if (argc < 3) {  // 默认参数
        LOGN = 12;    // 2^12 = 4096点NTT
        BATCH = 1;    // 默认处理1个多项式
    } else {          // 从命令行读取参数
        LOGN = atoi(argv[1]);  // 第一个参数为LOGN
        BATCH = atoi(argv[2]); // 第二个参数为BATCH
    }
    BATCH = 1;
    cout << "LOGN = " << LOGN << ", BATCH = " << BATCH << endl;

    // 初始化NTT参数
#ifdef DEFAULT_MODULUS
    // 使用默认模数配置
    NTTParameters<TestDataType> parameters(LOGN, ReductionPolynomial::X_N_minus);
#else
    // 自定义模数配置
    NTTFactors factor((Modulus) 576460752303415297, 288482366111684746, 238394956950829);
    NTTParameters parameters(LOGN, factor, ReductionPolynomial::X_N_minus);
#endif

    // 创建CPU端NTT生成器
    NTTCPU<TestDataType> generator(parameters);  // 用于生成参考结果的CPU实现

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(0);  // 固定种子用于结果可复现
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(0, maxNumber);

    // 生成随机输入数据
    vector<vector<TestDataType>> input1(BATCH);
    for (int j = 0; j < BATCH; j++) {
        for (int i = 0; i < parameters.n; i++) {  // parameters.n = 2^LOGN
            input1[j].push_back(dis(gen));  // 生成[0, modulus-1]范围内的随机数
        }
        print_array(input1[j].data(), "input1[" + std::to_string(j) + "]");
    }

    // 执行CPU端NTT（生成参考结果）
    vector<vector<TestDataType>> ntt_result(BATCH);
    for (int i = 0; i < BATCH; i++) {
        ntt_result[i] = generator.ntt(input1[i]);  // CPU NTT计算
        print_array(ntt_result[i].data(), "ntt_result[" + std::to_string(i) + "]");
    }

    // 执行CPU端INTT（生成参考结果）
    vector<vector<TestDataType>> intt_result(BATCH);
    for (int i = 0; i < BATCH; i++) {
        intt_result[i] = generator.intt(ntt_result[i]);  // CPU INTT计算
        print_array(intt_result[i].data(), "intt_result[" + std::to_string(i) + "]");
    }

    // GPU内存分配与数据传输-----------------------------------------------
    TestDataType* InOut_Datas;  // GPU输入/输出数据指针
    GPUNTT_CUDA_CHECK(  // 带错误检查的CUDA内存分配
        cudaMalloc(&InOut_Datas, BATCH * parameters.n * sizeof(TestDataType)));

    // 分批拷贝数据到GPU
    for (int j = 0; j < BATCH; j++) {
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(InOut_Datas + (parameters.n * j),  // 目标地址
                       input1[j].data(),                   // 源数据
                       parameters.n * sizeof(TestDataType),// 数据大小
                       cudaMemcpyHostToDevice));           // 传输方向
    }

    // 准备旋转因子表-----------------------------------------------------
    Root<TestDataType>* Forward_Omega_Table_Device;  // GPU端正向旋转因子表指针
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Forward_Omega_Table_Device,
                parameters.root_of_unity_size * sizeof(Root<TestDataType>)));

    // 生成并拷贝旋转因子表
    vector<Root<TestDataType>> forward_omega_table = 
        parameters.gpu_root_of_unity_table_generator(  // 生成旋转因子表
            parameters.forward_root_of_unity_table);
    
    GPUNTT_CUDA_CHECK(cudaMemcpy(Forward_Omega_Table_Device,
                                forward_omega_table.data(),
                                parameters.root_of_unity_size * sizeof(Root<TestDataType>),
                                cudaMemcpyHostToDevice));

   // 准备逆旋转因子表-----------------------------------------------------
    Root<TestDataType>* Inverse_Omega_Table_Device;

    GPUNTT_CUDA_CHECK(
    cudaMalloc(&Inverse_Omega_Table_Device,
            parameters.root_of_unity_size * sizeof(Root<TestDataType>)));

    vector<Root<TestDataType>> inverse_omega_table =
    parameters.gpu_root_of_unity_table_generator(
                parameters.inverse_root_of_unity_table);
    GPUNTT_CUDA_CHECK(cudaMemcpy(Inverse_Omega_Table_Device,
                                inverse_omega_table.data(),
                                parameters.root_of_unity_size * sizeof(Root<TestDataType>),
                                cudaMemcpyHostToDevice));


    // 配置模数参数-------------------------------------------------------
    Modulus<TestDataType>* test_modulus;  // GPU端模数参数指针
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));
    
    Modulus<TestDataType> test_modulus_[1] = {parameters.modulus};  // 主机端模数
    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, 
                               test_modulus_,
                               sizeof(Modulus<TestDataType>),
                               cudaMemcpyHostToDevice));

    // 配置NTT参数--------------------------------------------------------
    ntt_rns_configuration<TestDataType> cfg_ntt = {
        .n_power = LOGN,                     // log2(N)
        .ntt_type = FORWARD,                 // 正向变换
        .reduction_poly = ReductionPolynomial::X_N_minus,  // 约减多项式类型
        .zero_padding = false,               // 无零填充
        .stream = 0                          // 使用默认流
    };

    // 执行GPU NTT--------------------------------------------------------
    GPU_NTT_Inplace(  // 原地NTT变换
        InOut_Datas,               // 输入/输出数据指针
        Forward_Omega_Table_Device,// 旋转因子表
        test_modulus,              // 模数参数
        cfg_ntt,                   // 配置参数
        BATCH,                     // 批量数
        1                          // 流数量
    );

    // 回传结果并验证-----------------------------------------------------
    TestDataType* Output_Host =  // 主机端结果缓冲区
        (TestDataType*)malloc(BATCH * parameters.n * sizeof(TestDataType));
    
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host,   // 目标地址
                               InOut_Datas,   // 源数据
                               BATCH * parameters.n * sizeof(TestDataType),
                               cudaMemcpyDeviceToHost));

    // 结果验证
    bool check = true;
    for (int i = 0; i < BATCH; i++) {
        check = check_result(  // 自定义结果验证函数
            Output_Host + (i * parameters.n),  // GPU结果位置
            ntt_result[i].data(),              // CPU参考结果
            parameters.n                       // 数据长度
        );

        if (!check) {
            cout << "第 " << i << " 个多项式验证失败" << endl;
            break;
        }
    }
    if (check) cout << "NTT结果正确" << endl;

    // 配置模数参数-------------------------------------------------------
    Ninverse<TestDataType>* test_ninverse;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(Ninverse<TestDataType>)));

    Ninverse<TestDataType> test_ninverse_[1] = {parameters.n_inv};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_,
                                 sizeof(Ninverse<TestDataType>), cudaMemcpyHostToDevice));

    // 配置INTT参数--------------------------------------------------------
    ntt_rns_configuration<TestDataType> cfg_intt = {
    .n_power = LOGN,
    .ntt_type = INVERSE,
    .reduction_poly = ReductionPolynomial::X_N_minus,
    .zero_padding = false,
    .mod_inverse = test_ninverse,
    .stream = 0};

    // 执行GPU INTT--------------------------------------------------------
    TestDataType* Out_Datas;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Out_Datas, BATCH * parameters.n * sizeof(TestDataType)));

    GPU_NTT(InOut_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus,
            cfg_intt, BATCH, 1);

    // 回传结果并验证-----------------------------------------------------
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host,   // 目标地址
                            Out_Datas,   // 源数据
                            BATCH * parameters.n * sizeof(TestDataType),
                            cudaMemcpyDeviceToHost));

    // 结果验证
    check = true;
    for (int i = 0; i < BATCH; i++) {
        check = check_result(  // 自定义结果验证函数
            Output_Host + (i * parameters.n),  // GPU结果位置
            intt_result[i].data(),             // CPU参考结果
            parameters.n                       // 数据长度
        );

        if (!check) {
            cout << "第 " << i << " 个多项式验证失败" <<endl;
        }
    }
    if (check) cout << "INTT结果正确" << endl;

    // 资源释放-----------------------------------------------------------
    GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));

    GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
    GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
    free(Output_Host);

    return EXIT_SUCCESS;
}