#include <cstdlib>    // 标准库头文件
#include <random>     // 随机数生成库
#include <chrono>     // 时间库头文件
#include "ntt.cuh"    // GPU-NTT 库核心头文件
#include "lib/helper.cuh"

// #define DEFAULT_MODULUS  // 启用默认模数模式

using namespace std;
using namespace gpuntt;  // 使用 GPU-NTT 库的命名空间

// 全局参数声明
int LOGN;   // NTT变换的log2长度（实际长度N=2^LOGN）
int BATCH;  // 批量处理的多项式数量
int alpha;  
// 选择测试数据类型（64位版本）
typedef Data64 TestDataType;  // 定义用于测试的数据类型（32/64位可切换）

// 切分多项式
vector<vector<TestDataType>> split_poly(const vector<TestDataType> &a)
{
    int part_size = a.size() / 2; 
    vector<vector<TestDataType>> a_parts(2);
    for (int i = 0; i < 2; ++i)
    {
        a_parts[i] = vector<TestDataType>(a.begin() + i * part_size, a.begin() + (i + 1) * part_size);
    }
    return a_parts;
}

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
        // BATCH = 1;    // 默认处理1个多项式
        alpha = 1;
    } else {          // 从命令行读取参数
        LOGN = atoi(argv[1]);  // 第一个参数为LOGN
        // BATCH = atoi(argv[2]); // 第二个参数为BATCH
        alpha = atoi(argv[2]);
    }
    // int alpha = 3;
    BATCH = 1 << alpha;

    cout << "LOGN = " << LOGN << ", BATCH = " << BATCH << endl;

    // 初始化NTT参数
#ifdef DEFAULT_MODULUS
    // 使用默认模数配置
    NTTParameters<TestDataType> parameters(LOGN, ReductionPolynomial::X_N_minus);
#else
    // 自定义模数配置
    NTTFactors<TestDataType> factor(Modulus<TestDataType>(576460752303415297), 288482366111684746, 238394956950829);
    NTTParameters<TestDataType> parameters(LOGN, factor, ReductionPolynomial::X_N_minus);
#endif

    // 创建CPU端NTT生成器
    // parameters.modulus.value = 268582913;
    NTTCPU<TestDataType> generator(parameters);  // 用于生成参考结果的CPU实现

    cout << "默认模数 = " << parameters.modulus.value << endl;
    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(0);  // 固定种子用于结果可复现
    unsigned long long maxNumber = parameters.modulus.value - 1;
    // unsigned long long maxNumber = 5;
    std::uniform_int_distribution<unsigned long long> dis(0, maxNumber);

    cout << endl << "-----CPU 计算-----" << endl;
    // 生成随机输入数据
    cout << endl << "1.生成随机输入数据: input1, input2" << endl;
    vector<vector<TestDataType>> input1(BATCH), input2(BATCH);
    for (int j = 0; j < BATCH; j++) {
        for (int i = 0; i < parameters.n; i++) {  // parameters.n = 2^LOGN
            input1[j].push_back(dis(gen));  // 生成[0, modulus-1]范围内的随机数
            // input1[j].push_back(0);
            input2[j].push_back(dis(gen));
            // input2[j].push_back(2);
        }
        print_array(input1[j].data(), "input1[" + std::to_string(j) + "]");
        print_array(input2[j].data(), "input2[" + std::to_string(j) + "]");
    }
    // input1[0][0] = 1;input1[0][1] = 2;input1[0][parameters.n-1] = 3; input1[1][0] = 4;
    // input2[0][0] = 1;input2[0][1] = 1;input2[0][2] = 1;              input2[1][0] = 1;
    // input2[0][0] = 1;
    // 合并输入验证
    cout << endl << "合并输入验证 merge(input1), merge(input2)" << endl;
    vector<TestDataType> merged_input1, merged_input2;
    for (const auto &part : input1)
    {
        merged_input1.insert(merged_input1.end(), part.begin(), part.end());
    }
    // print_array(merged_input1.data(), "merged_input1");
    for(const auto &part : input2)
    {
        merged_input2.insert(merged_input2.end(), part.begin(), part.end());
    }
    // print_array(merged_input2.data(), "merged_input2");

    // NTT 乘法验证
    NTTParameters<TestDataType> parameters_merged(LOGN + alpha, ReductionPolynomial::X_N_minus);
    // parameters_merged.modulus.value = 268460033;
    // cout << "merged模数 = " << parameters_merged.modulus.value << endl;
    cout << endl << "INTT( NTT(merged_input1) * NTT(merged_input2) ) " << endl;

    // 添加计时器
    auto cpu_ntt_start = chrono::high_resolution_clock::now(); // 开始计时

    NTTCPU<TestDataType> generator_merged(parameters_merged);
    vector<TestDataType> merged_ntt_result1 = generator_merged.ntt(merged_input1);
    vector<TestDataType> merged_ntt_result2 = generator_merged.ntt(merged_input2); 
    vector<TestDataType> merged_ntt_result = generator_merged.mult(merged_ntt_result1, merged_ntt_result2);
    vector<TestDataType> intt_merged_ntt_result = generator_merged.intt(merged_ntt_result);
    print_array(intt_merged_ntt_result.data(), "intt_merged_ntt_result");

    auto cpu_ntt_end = chrono::high_resolution_clock::now(); // 结束计时
    chrono::duration<double, milli> cpu_ntt_duration = cpu_ntt_end - cpu_ntt_start;

    // cout << "merged_input1[n-1]:" << merged_input1[parameters.n-1] << endl;
    // 执行schoolbook乘法
    cout << endl << "schoolbook_poly_multiplication( merged_input1, merged_input2 )" << endl;

    // 添加计时器
    auto schoolbook_mult_start = chrono::high_resolution_clock::now(); // 开始计时
    vector<TestDataType> merged_schoolbook_result = schoolbook_poly_multiplication<TestDataType>(
    merged_input1, 
    merged_input2,
    parameters_merged.modulus,
    parameters_merged.poly_reduction  // 来自 NTTParameters
    );
    print_array(merged_schoolbook_result.data(), "merged_schoolbook_result");

    auto schoolbook_mult_end = chrono::high_resolution_clock::now(); // 结束计时
    chrono::duration<double, milli> schoolbook_mult_duration = schoolbook_mult_end - schoolbook_mult_start;
   


    // 给每个多项式补0
    for(int i = 0; i < BATCH; i++)
    {
        input1[i].resize(2*parameters.n, 0);
        input2[i].resize(2*parameters.n, 0);
    }
    NTTParameters<TestDataType> parameters_2n(LOGN+1, ReductionPolynomial::X_N_minus);
    NTTCPU<TestDataType> generator_2n(parameters_2n);

    // 执行CPU端NTT（生成参考结果）
    cout << endl << "2.CPU NTT: ntt(input1), ntt(input2) " << endl;
    vector<vector<TestDataType>> ntt_result(BATCH), ntt_result2(BATCH);
    for (int i = 0; i < BATCH; i++) {
        ntt_result[i] = generator_2n.ntt(input1[i]);  // CPU NTT计算
        ntt_result2[i] = generator_2n.ntt(input2[i]);
        print_array(ntt_result[i].data(), "ntt_result[" + std::to_string(i) + "]");
        print_array(ntt_result2[i].data(), "ntt_result2[" + std::to_string(i) + "]");
    }

    // Karatsuba 计算
    cout << endl << "3.Karatsuba(ntt(input1), ntt(input2)) " << endl;
    // 处理对角线项 diag_term[i] = ntt_result[i] * ntt_result2[i]
    vector<vector<TestDataType>> diag_term(BATCH);
    for(int i = 0; i < BATCH; i++)
    {
        diag_term[i].resize(2*parameters.n, 0);
        diag_term[i] = generator_2n.mult(ntt_result[i], ntt_result2[i]);
    }

    //  处理其它项 other_term[i] = ntt_result[i] * ntt_result2[i]
    vector<vector<TestDataType>> other_term(2 * BATCH, vector<TestDataType>(2*parameters.n, 0));
    for(int i = 0; i < BATCH; i++)
    {
        for(int j = i + 1; j < BATCH; j++)
        {
            // 计算fi*gj + fj*gi
            // 使用 (fi + fj) * (gi + gj) - fi*gi - fj*gj 优化计算
            vector<TestDataType> tmp1 = generator_2n.add(ntt_result[i], ntt_result[j]);
            vector<TestDataType> tmp2 = generator_2n.add(ntt_result2[i], ntt_result2[j]);
            vector<TestDataType> tmp3 = generator_2n.mult(tmp1, tmp2);
            tmp3 = generator_2n.sub(tmp3, diag_term[i]);
            tmp3 = generator_2n.sub(tmp3, diag_term[j]);
            other_term[i + j] = generator_2n.add(other_term[i + j], tmp3);
        }

    }
    // 对角线项加上其它项
    for (int i = 0; i < BATCH; ++i)
    {
        other_term[2 * i] = generator_2n.add(other_term[2 * i], diag_term[i]);
    }
    // other_term[i] + other_term[i + BATCH] 
    vector<vector<TestDataType>> ntt_result_final(BATCH);
    for(int i = 0; i < BATCH; ++i)
    {
        ntt_result_final[i].resize(2*parameters.n, 0);
        ntt_result_final[i] = generator_2n.add(other_term[i], other_term[i + BATCH]);
        print_array(ntt_result_final[i].data(), "ntt_result_final[" + std::to_string(i) + "]");
    }
    // INTT操作
    cout << endl << "4.INTT( Karatsuba( ntt(input1), ntt(input2) ) ) " << endl;
    vector<vector<TestDataType>> result_final(BATCH);
    for (int i = 0; i < BATCH; ++i)
    {
        result_final[i] = generator_2n.intt(ntt_result_final[i]);
        print_array(result_final[i].data(), "result_final[" + std::to_string(i) + "]");
    }
    // 合并结果
    vector<vector<vector<TestDataType>>> result_parts_split(BATCH);
    for (int i = 0; i < BATCH; ++i)
    {
        result_parts_split[i] = split_poly(result_final[i]); // 将每个部分拆分为两个基向量
    }
        
    for (int i = 0; i < BATCH; ++i)
    {
        // 前后相邻两个基向量相加 (第一个部分与最后一个部分相加)
        if (i == 0)
        {
            result_final[i] = generator_2n.add(result_parts_split[i][0], result_parts_split[BATCH - 1][1]);
            continue;
        }
        result_final[i] = generator_2n.add(result_parts_split[i - 1][1], result_parts_split[i][0]);
    }
    vector<TestDataType> result_merged;
    for (const auto &part : result_final)
    {
        result_merged.insert(result_merged.end(), part.begin(), part.end());
    }
    print_array(result_merged.data(), "result_merged");


   
    cout << endl << "LOGN = " << LOGN << ", " << "alpha = " << alpha << ", " 
    << "BATCH = " << BATCH << " , " << "LOGN + alpha = " << LOGN + alpha << endl;
    cout << "CPU NTT总耗时: " << cpu_ntt_duration.count() << " ms" << endl;
    cout << "Schoolbook multiplication 总耗时: " << schoolbook_mult_duration.count() << " ms" << endl;





    cout << endl << "-----------------GPU 计算-----------------" << endl;

    cout << endl << "-----------------普通 GPU-NTT 乘法-----------------" << endl;

    {
        // 在开始处添加总计时事件
        cudaEvent_t total_start, total_stop;
        GPUNTT_CUDA_CHECK(cudaEventCreate(&total_start));
        GPUNTT_CUDA_CHECK(cudaEventCreate(&total_stop));
        // 计时
        cudaEvent_t start, stop;
        GPUNTT_CUDA_CHECK(cudaEventCreate(&start));
        GPUNTT_CUDA_CHECK(cudaEventCreate(&stop));
        float elapsedTime,totalElapsedTime = 0;

        GPUNTT_CUDA_CHECK(cudaEventRecord(total_start, 0));

        // 在主要计算开始前记录总耗时起点
        GPUNTT_CUDA_CHECK(cudaEventRecord(total_start, 0));

        // 准备旋转因子表-----------------------------------------------------
        Root<TestDataType>* Forward_Omega_Table_Device;  // GPU端正向旋转因子表指针
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&Forward_Omega_Table_Device,
                    parameters_merged.root_of_unity_size * sizeof(Root<TestDataType>)));

        // 生成并拷贝旋转因子表
        vector<Root<TestDataType>> forward_omega_table = 
            parameters_merged.gpu_root_of_unity_table_generator(  // 生成旋转因子表
                parameters_merged.forward_root_of_unity_table);
        
        GPUNTT_CUDA_CHECK(cudaMemcpy(Forward_Omega_Table_Device,
                                    forward_omega_table.data(),
                                    parameters_merged.root_of_unity_size * sizeof(Root<TestDataType>),
                                    cudaMemcpyHostToDevice));
        
        // 准备逆旋转因子表-----------------------------------------------------
        Root<TestDataType>* Inverse_Omega_Table_Device;

        GPUNTT_CUDA_CHECK(
        cudaMalloc(&Inverse_Omega_Table_Device,
                parameters_merged.root_of_unity_size * sizeof(Root<TestDataType>)));

        vector<Root<TestDataType>> inverse_omega_table =
        parameters_merged.gpu_root_of_unity_table_generator(
                    parameters_merged.inverse_root_of_unity_table);
        GPUNTT_CUDA_CHECK(cudaMemcpy(Inverse_Omega_Table_Device,
                                    inverse_omega_table.data(),
                                    parameters_merged.root_of_unity_size * sizeof(Root<TestDataType>),
                                    cudaMemcpyHostToDevice));


        // 配置模数参数-------------------------------------------------------
        Modulus<TestDataType>* test_modulus;  // GPU端模数参数指针
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));
        
        Modulus<TestDataType> test_modulus_[1] = {parameters_merged.modulus};  // 主机端模数
        GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, 
                                test_modulus_,
                                sizeof(Modulus<TestDataType>),
                                cudaMemcpyHostToDevice));

        // 配置NTT参数--------------------------------------------------------
        ntt_rns_configuration<TestDataType> cfg_ntt = {
            .n_power = LOGN + alpha,                     // log2(N)
            .ntt_type = FORWARD,                 // 正向变换
            .reduction_poly = ReductionPolynomial::X_N_minus,  // 约减多项式类型
            .zero_padding = false,               // 无零填充
            .stream = 0                          // 使用默认流
        };

        // GPU内存分配与数据传输-----------------------------------------------
        TestDataType* Merge_Datas1, *Merge_Datas2;  // GPU输入/输出数据指针
        GPUNTT_CUDA_CHECK(  // 带错误检查的CUDA内存分配
                        cudaMalloc(&Merge_Datas1,  parameters_merged.n * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(  // 带错误检查的CUDA内存分配
                        cudaMalloc(&Merge_Datas2,  parameters_merged.n * sizeof(TestDataType)));
        
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(Merge_Datas1,  // 目标地址
                       merged_input1.data(),                   // 源数据
                       parameters_merged.n * sizeof(TestDataType),// 数据大小
                       cudaMemcpyHostToDevice));           // 传输方向
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(Merge_Datas2,  // 目标地址
                       merged_input2.data(),                   // 源数据
                       parameters_merged.n * sizeof(TestDataType),// 数据大小
                       cudaMemcpyHostToDevice));           // 传输方向
                
        GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));
        // 执行GPU-NTT变换
        GPU_NTT_Inplace(  // 原地NTT变换
            Merge_Datas1,               // 输入/输出数据指针
            Forward_Omega_Table_Device,// 旋转因子表
            test_modulus,              // 模数参数
            cfg_ntt,                   // 配置参数
            1,                          // 批量数
            1                          // 流数量
        );
        GPU_NTT_Inplace(  // 原地NTT变换
            Merge_Datas2,               // 输入/输出数据指针
            Forward_Omega_Table_Device,// 旋转因子表
            test_modulus,              // 模数参数
            cfg_ntt,                   // 配置参数
            1,                          // 批量数
            1                          // 流数量
        );
        GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
        GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
        GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        std::cout << "普通NTT耗时: " << elapsedTime << " ms" << std::endl;
        totalElapsedTime += elapsedTime;

        GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));
        // 执行乘法操作
        PointwiseMultiply(Merge_Datas1, Merge_Datas2, Merge_Datas1, parameters_merged.modulus, parameters_merged.n, 1);
        
        GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
        GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
        GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        std::cout << "点乘耗时: " << elapsedTime << " ms" << std::endl;
        totalElapsedTime += elapsedTime;


        // 配置INTT模数参数-------------------------------------------------------
        Ninverse<TestDataType>* test_ninverse;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(Ninverse<TestDataType>)));

        Ninverse<TestDataType> test_ninverse_[1] = {parameters_merged.n_inv};

        GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_,
                                    sizeof(Ninverse<TestDataType>), cudaMemcpyHostToDevice));

        // 配置INTT参数--------------------------------------------------------
        ntt_rns_configuration<TestDataType> cfg_intt = {
            .n_power = LOGN + alpha,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = test_ninverse,
            .stream = 0
        };


        GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));
        // 执行INTT变换
        GPU_NTT_Inplace(  // 原地NTT变换
            Merge_Datas1,               // 输入/输出数据指针
            Inverse_Omega_Table_Device,// 旋转因子表
            test_modulus,              // 模数参数
            cfg_intt,                   // 配置参数
            1,                          // 批量数
            1                          // 流数量
        );

        GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
        GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
        GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        std::cout << "普通INTT耗时: " << elapsedTime << " ms" << std::endl;
        totalElapsedTime += elapsedTime;
        //  GPU_NTT(ntt_result_device, result_device, Inverse_Omega_Table_Device, test_modulus, cfg_intt, BATCH, 1);

        // 添加总耗时计算
        GPUNTT_CUDA_CHECK(cudaEventRecord(total_stop, 0)); 
        GPUNTT_CUDA_CHECK(cudaEventSynchronize(total_stop));
        float totalElapsed = 0;
        GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&totalElapsed, total_start, total_stop));
        std::cout << "\n----- 普通GPU-NTT总耗时分析 -----" << std::endl;
        std::cout << "[TOTAL] 核心计算耗时: " << totalElapsedTime << " ms" << std::endl;
        std::cout << "[TOTAL] 总耗时: " << totalElapsed << " ms" << std::endl;
        std::cout << "------------------------------" << std::endl;

        // 销毁总计时事件
        GPUNTT_CUDA_CHECK(cudaEventDestroy(total_start));
        GPUNTT_CUDA_CHECK(cudaEventDestroy(total_stop));

        // 销毁耗时统计
        GPUNTT_CUDA_CHECK(cudaEventDestroy(start));
        GPUNTT_CUDA_CHECK(cudaEventDestroy(stop));

        GPUNTT_CUDA_CHECK( cudaFree(Merge_Datas1) );
        GPUNTT_CUDA_CHECK( cudaFree(Merge_Datas2) );
        GPUNTT_CUDA_CHECK( cudaFree(Forward_Omega_Table_Device) );
        GPUNTT_CUDA_CHECK( cudaFree(Inverse_Omega_Table_Device) );
        GPUNTT_CUDA_CHECK( cudaFree(test_modulus) );
        GPUNTT_CUDA_CHECK( cudaFree(test_ninverse) );
    }


    cout << endl << "---------------Karatsuba-NTT乘法---------------" << endl;
    // 在GPU计算开始处添加总计时事件
    cudaEvent_t total_start, total_stop;
    GPUNTT_CUDA_CHECK(cudaEventCreate(&total_start));
    GPUNTT_CUDA_CHECK(cudaEventCreate(&total_stop));
    // 计时
    cudaEvent_t start, stop;
    GPUNTT_CUDA_CHECK(cudaEventCreate(&start));
    GPUNTT_CUDA_CHECK(cudaEventCreate(&stop));
    float elapsedTime,totalElapsedTime = 0;


    
    // 在主要计算开始前记录总耗时起点
    GPUNTT_CUDA_CHECK(cudaEventRecord(total_start, 0));

    // 准备旋转因子表-----------------------------------------------------
    Root<TestDataType>* Forward_Omega_Table_Device;  // GPU端正向旋转因子表指针
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Forward_Omega_Table_Device,
                parameters_2n.root_of_unity_size * sizeof(Root<TestDataType>)));

    // 生成并拷贝旋转因子表
    vector<Root<TestDataType>> forward_omega_table = 
        parameters_2n.gpu_root_of_unity_table_generator(  // 生成旋转因子表
            parameters_2n.forward_root_of_unity_table);
    
    GPUNTT_CUDA_CHECK(cudaMemcpy(Forward_Omega_Table_Device,
                                forward_omega_table.data(),
                                parameters_2n.root_of_unity_size * sizeof(Root<TestDataType>),
                                cudaMemcpyHostToDevice));

   // 准备逆旋转因子表-----------------------------------------------------
    Root<TestDataType>* Inverse_Omega_Table_Device;

    GPUNTT_CUDA_CHECK(
    cudaMalloc(&Inverse_Omega_Table_Device,
            parameters_2n.root_of_unity_size * sizeof(Root<TestDataType>)));

    vector<Root<TestDataType>> inverse_omega_table =
    parameters_2n.gpu_root_of_unity_table_generator(
                parameters_2n.inverse_root_of_unity_table);
    GPUNTT_CUDA_CHECK(cudaMemcpy(Inverse_Omega_Table_Device,
                                inverse_omega_table.data(),
                                parameters_2n.root_of_unity_size * sizeof(Root<TestDataType>),
                                cudaMemcpyHostToDevice));


    // 配置模数参数-------------------------------------------------------
    Modulus<TestDataType>* test_modulus;  // GPU端模数参数指针
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));
    
    Modulus<TestDataType> test_modulus_[1] = {parameters_2n.modulus};  // 主机端模数
    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, 
                               test_modulus_,
                               sizeof(Modulus<TestDataType>),
                               cudaMemcpyHostToDevice));

    // 配置NTT参数--------------------------------------------------------
    ntt_rns_configuration<TestDataType> cfg_ntt = {
        .n_power = LOGN + 1,                     // log2(N)
        .ntt_type = FORWARD,                 // 正向变换
        .reduction_poly = ReductionPolynomial::X_N_minus,  // 约减多项式类型
        .zero_padding = false,               // 无零填充
        .stream = 0                          // 使用默认流
    };

    // GPU内存分配与数据传输-----------------------------------------------
    TestDataType* InOut_Datas, *InOut_Datas2;  // GPU输入/输出数据指针
    GPUNTT_CUDA_CHECK(  // 带错误检查的CUDA内存分配
        cudaMalloc(&InOut_Datas, BATCH * parameters_2n.n * sizeof(TestDataType)));
    GPUNTT_CUDA_CHECK(  // 带错误检查的CUDA内存分配
        cudaMalloc(&InOut_Datas2, BATCH * parameters_2n.n * sizeof(TestDataType)));
    
    // 分批拷贝数据到GPU
    for (int j = 0; j < BATCH; j++) {
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(InOut_Datas + (parameters_2n.n * j),  // 目标地址
                       input1[j].data(),                   // 源数据
                       parameters_2n.n * sizeof(TestDataType),// 数据大小
                       cudaMemcpyHostToDevice));           // 传输方向
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(InOut_Datas2 + (parameters_2n.n * j) ,  // 目标地址
                       input2[j].data(),                   // 源数据
                       parameters_2n.n * sizeof(TestDataType),// 数据大小
                       cudaMemcpyHostToDevice));           
    }

    // 执行GPU NTT--------------------------------------------------------

    // 测量NTT时间
    GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));

    GPU_NTT_Inplace(  // 原地NTT变换
        InOut_Datas,               // 输入/输出数据指针
        Forward_Omega_Table_Device,// 旋转因子表
        test_modulus,              // 模数参数
        cfg_ntt,                   // 配置参数
        BATCH,                     // 批量数
        1                          // 流数量
    );
    GPU_NTT_Inplace(  // 原地NTT变换
        InOut_Datas2,               // 输入/输出数据指针
        Forward_Omega_Table_Device,// 旋转因子表
        test_modulus,              // 模数参数
        cfg_ntt,                   // 配置参数
        BATCH,                     // 批量数
        1                          // 流数量
    );

    GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
    GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
    GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "双路NTT总耗时: " << elapsedTime << " ms" << std::endl;
    totalElapsedTime += elapsedTime;



    TestDataType* diag_term_device; // GPU端对角项指针
    GPUNTT_CUDA_CHECK(cudaMalloc(&diag_term_device, BATCH * parameters_2n.n * sizeof(TestDataType)));
    TestDataType* other_term_device; // GPU端非对角项指针
    GPUNTT_CUDA_CHECK(cudaMalloc(&other_term_device, 2 * BATCH * parameters_2n.n * sizeof(TestDataType)));
    TestDataType* ntt_result_device; // GPU端NTT结果指针
    GPUNTT_CUDA_CHECK(cudaMalloc(&ntt_result_device, BATCH * parameters_2n.n * sizeof(TestDataType)));
    GPUNTT_CUDA_CHECK(cudaMemset(ntt_result_device, 0, BATCH * parameters_2n.n * sizeof(TestDataType)));
    TestDataType* result_device; // GPU端结果指针
    GPUNTT_CUDA_CHECK(cudaMalloc(&result_device, BATCH * parameters_2n.n * sizeof(TestDataType))); 
    TestDataType* result_merged_device; // GPU端结果内部折叠指针
    GPUNTT_CUDA_CHECK(cudaMalloc(&result_merged_device, BATCH * parameters.n * sizeof(TestDataType)));

    // TestDataType* diag_term_device;
    // GPUNTT_CUDA_CHECK(cudaMalloc(&diag_term_device, BATCH * parameters_2n.n * sizeof(TestDataType)));

    // 处理对角线项 (i = j) diagonal term
    // 在点乘操作添加计时
    GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));

        PointwiseMultiply<TestDataType>(
            InOut_Datas,                      // 第一个多项式地址
            InOut_Datas2,                     // 第二个多项式地址
            diag_term_device,                 // 输出地址
            parameters_2n.modulus,               // 模数参数
            BATCH * parameters_2n.n,             // 多项式长度
            1,                                // 批量数
            0                                 // 使用默认流
        );

    GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
    GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
    GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "对角线项计算耗时: " << elapsedTime << " ms" << std::endl;
    totalElapsedTime += elapsedTime;

    //验证
    TestDataType* Output_Host =  // 主机端结果缓冲区
    (TestDataType*)malloc(BATCH * parameters_2n.n * sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host,   // 目标地址
                               diag_term_device,   // 源数据
                               BATCH * parameters_2n.n * sizeof(TestDataType),
                               cudaMemcpyDeviceToHost));
    // print_array(Output_Host, "GPU对角线项结果");
    VERIFY_RESULTS(Output_Host, diag_term, "GPU计算对角线项结果正确");

    // 处理其它项
    // 在其它项计算添加计时
    GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));

    karatsuba(InOut_Datas, InOut_Datas2, diag_term_device, other_term_device, parameters_2n.modulus, parameters_2n.n, BATCH);
    merge_diag_non_diag(diag_term_device, other_term_device, other_term_device, parameters_2n.modulus, parameters_2n.n, BATCH);
    
    GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
    GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
    GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "非对角项计算耗时: " << elapsedTime << " ms" << std::endl;
    totalElapsedTime += elapsedTime;

    // GPUNTT_CUDA_CHECK(cudaGetLastError());
    


    // 验证other_term计算结果
    TestDataType* host_other_term = (TestDataType*)malloc(2*BATCH*parameters_2n.n*sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(host_other_term, other_term_device, 
                            2*BATCH*parameters_2n.n*sizeof(TestDataType),
                            cudaMemcpyDeviceToHost));
    VERIFY_RESULTS(host_other_term, other_term, "GPU计算其它项结果正确");

    // 合并结果
    // 在结果合并添加计时
    GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));
    PointwiseAdd(other_term_device, other_term_device + (BATCH * parameters_2n.n), ntt_result_device, parameters_2n.modulus, BATCH * parameters_2n.n);

    GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
    GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
    GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "合并对角线项和其它项耗时: " << elapsedTime << " ms" << std::endl;
    totalElapsedTime += elapsedTime;

    // 验证
    TestDataType* host_ntt_result = (TestDataType*)malloc(BATCH*parameters_2n.n*sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(host_ntt_result, ntt_result_device, 
                            BATCH*parameters_2n.n*sizeof(TestDataType),
                            cudaMemcpyDeviceToHost));
    // print_array(host_ntt_result, "GPU 计算合并结果");
    VERIFY_RESULTS(host_ntt_result, ntt_result_final, "GPU计算合并对角线项和其它项结果正确");

    // 配置INTT模数参数-------------------------------------------------------
    Ninverse<TestDataType>* test_ninverse;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(Ninverse<TestDataType>)));

    Ninverse<TestDataType> test_ninverse_[1] = {parameters_2n.n_inv};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_,
                                 sizeof(Ninverse<TestDataType>), cudaMemcpyHostToDevice));

    // 配置INTT参数--------------------------------------------------------
    ntt_rns_configuration<TestDataType> cfg_intt = {
        .n_power = LOGN + 1,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = test_ninverse,
        .stream = 0
    };
    
    // 执行GPU_INTT( ntt_result_device )--------------------------------------------------------
    // 在INTT操作添加计时
    GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));  

    // TestDataType* result_device;
    // GPUNTT_CUDA_CHECK(cudaMalloc(&result_device, BATCH * parameters_2n.n * sizeof(TestDataType))); 
    GPU_NTT(ntt_result_device, result_device, Inverse_Omega_Table_Device, test_modulus, cfg_intt, BATCH, 1);

    GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
    GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
    GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "INTT计算耗时: " << elapsedTime << " ms" << std::endl;
    totalElapsedTime += elapsedTime;
        
    // 合并结果
    // 在结果合并添加计时
    GPUNTT_CUDA_CHECK(cudaEventRecord(start, 0));
    // TestDataType* result_merged_device;
    // GPUNTT_CUDA_CHECK(cudaMalloc(&result_merged_device, BATCH * parameters.n * sizeof(TestDataType)));
    // #pragma unroll
    // for(int i = 0; i < BATCH; i++)
    // {
    //     TestDataType* result_i_0 = result_device + i * parameters_2n.n;                     // 表示 result[i][0]
    //     TestDataType* result_i_1 = result_device + ((i == 0) ? 
    //         (BATCH-1)*parameters_2n.n :             // 表示 result[BATCH-1][1]
    //         (i-1)*parameters_2n.n) + parameters.n;  // 表示 result[i-1][1]
    //     TestDataType* result_merged_i = result_merged_device + i * parameters.n;            // 表示 result_merged[i]
    //     // result_merged[i] = result[i][0] + result[i-1][1]
    //     PointwiseAdd(result_i_0 , result_i_1, result_merged_i, parameters_2n.modulus, parameters.n);    //使用parameters_2n.modulus
    // }
    merge(result_device, result_merged_device, parameters_2n.modulus, parameters.n, BATCH);

    


    GPUNTT_CUDA_CHECK(cudaEventRecord(stop, 0));
    GPUNTT_CUDA_CHECK(cudaEventSynchronize(stop));
    GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "合并结果耗时: " << elapsedTime << " ms" << std::endl;
    totalElapsedTime += elapsedTime;

    // 添加总耗时计算
    GPUNTT_CUDA_CHECK(cudaEventRecord(total_stop, 0)); 
    GPUNTT_CUDA_CHECK(cudaEventSynchronize(total_stop));
    float totalElapsed = 0;
    GPUNTT_CUDA_CHECK(cudaEventElapsedTime(&totalElapsed, total_start, total_stop));
    std::cout << "\n----- 总耗时分析 -----" << std::endl;
    std::cout << "[TOTAL] 核心计算耗时: " << totalElapsedTime << " ms" << std::endl;
    std::cout << "[TOTAL] 总耗时: " << totalElapsed << " ms" << std::endl;
    std::cout << "------------------------------" << std::endl;

    // 销毁总计时事件
    GPUNTT_CUDA_CHECK(cudaEventDestroy(total_start));
    GPUNTT_CUDA_CHECK(cudaEventDestroy(total_stop));

    // 销毁耗时统计
    GPUNTT_CUDA_CHECK(cudaEventDestroy(start));
    GPUNTT_CUDA_CHECK(cudaEventDestroy(stop));

    // 验证
    TestDataType* host_result = (TestDataType*)malloc(BATCH * parameters.n * sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(host_result,   // 目标地址
                            result_merged_device,   // 源数据
                            BATCH * parameters.n * sizeof(TestDataType),
                            cudaMemcpyDeviceToHost));
    bool check = true; 
    for (int i = 0; i < BATCH; i++) 
    { 
        check = check_result( 
            host_result + (i * parameters.n), 
            result_final[i].data(), 
            parameters.n 
        ); 
        if (!check) { 
            std::cout << "第 " << i << " 个多项式验证失败" << std::endl; 
            break; 
        } 
    } 
    if (check) std::cout << "Karatsuba_GPU计算结果正确" << std::endl;
    // VERIFY_RESULTS(host_result, result_final, "Karatsuba_GPU计算结果正确");


    // 资源释放-----------------------------------------------------------
    GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas2));

    GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
    GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
    free(Output_Host);

    GPUNTT_CUDA_CHECK(cudaFree(diag_term_device));
    GPUNTT_CUDA_CHECK(cudaFree(other_term_device));
    GPUNTT_CUDA_CHECK(cudaFree(result_merged_device));

    return EXIT_SUCCESS;
}
// cmake --build .