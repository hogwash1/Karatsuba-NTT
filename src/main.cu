#include <cstdlib>    // 标准库头文件
#include <random>     // 随机数生成库
#include "ntt.cuh"    // GPU-NTT 库核心头文件
#include "lib/helper.cuh"

// #define DEFAULT_MODULUS  // 启用默认模数模式

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
    int alpha = 2;
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
            // input2[j].push_back(0);
        }
        print_array(input1[j].data(), "input1[" + std::to_string(j) + "]");
        print_array(input2[j].data(), "input2[" + std::to_string(j) + "]");
    }
    // input1[0][0] = 1;input1[0][1] = 2;input1[0][parameters.n-1] = 3;
    // input2[0][0] = 1;input2[0][1] = 1;input2[0][2] = 1;

    // 执行CPU端NTT（生成参考结果）
    cout << endl << "2.CPU NTT: ntt(input1), ntt(input2) " << endl;
    vector<vector<TestDataType>> ntt_result(BATCH), ntt_result2(BATCH);
    for (int i = 0; i < BATCH; i++) {
        ntt_result[i] = generator.ntt(input1[i]);  // CPU NTT计算
        ntt_result2[i] = generator.ntt(input2[i]);
        print_array(ntt_result[i].data(), "ntt_result[" + std::to_string(i) + "]");
        print_array(ntt_result2[i].data(), "ntt_result2[" + std::to_string(i) + "]");
    }
{  
    // Karasuba 验证
    vector<TestDataType> test_result, x1, x2, test_result2;
    // f0 * g0 + f1 * g1
    x1 = generator.mult(ntt_result[0], ntt_result2[0]);
    x2 = generator.mult(ntt_result[1], ntt_result2[1]);
    test_result = generator.add(x1, x2);
    print_array(test_result.data(), "test_result");
    x1 = generator.mult(ntt_result[0], ntt_result2[1]);
    x2 = generator.mult(ntt_result[1], ntt_result2[0]);
    test_result2 = generator.add(x1, x2);
    print_array(test_result2.data(), "test_result2");
}
    // Karatsuba 计算
    cout << endl << "3.Karatsuba(ntt(input1), ntt(input2)) " << endl;
    // 处理对角线项 diag_term[i] = ntt_result[i] * ntt_result2[i]
    vector<vector<TestDataType>> diag_term(BATCH);
    for(int i = 0; i < BATCH; i++)
    {
        diag_term[i].resize(parameters.n, 0);
        diag_term[i] = generator.mult(ntt_result[i], ntt_result2[i]);
    }

    //  处理其它项 other_term[i] = ntt_result[i] * ntt_result2[i]
    vector<vector<TestDataType>> other_term(2 * BATCH, vector<TestDataType>(parameters.n, 0));
    for(int i = 0; i < BATCH; i++)
    {
        for(int j = i + 1; j < BATCH; j++)
        {
            // 计算fi*gj + fj*gi
            // 使用 (fi + fj) * (gi + gj) - fi*gi - fj*gj 优化计算
            vector<TestDataType> tmp1 = generator.add(ntt_result[i], ntt_result[j]);
            vector<TestDataType> tmp2 = generator.add(ntt_result2[i], ntt_result2[j]);
            vector<TestDataType> tmp3 = generator.mult(tmp1, tmp2);
            tmp3 = generator.sub(tmp3, diag_term[i]);
            tmp3 = generator.sub(tmp3, diag_term[j]);
            other_term[i + j] = generator.add(other_term[i + j], tmp3);
        }

    }
    // 对角线项加上其它项
    for (int i = 0; i < BATCH; ++i)
    {
        other_term[2 * i] = generator.add(other_term[2 * i], diag_term[i]);
    }
    // other_term[i] + other_term[i + BATCH] 
    vector<vector<TestDataType>> ntt_result_final(BATCH);
    for(int i = 0; i < BATCH; ++i)
    {
        ntt_result_final[i].resize(parameters.n, 0);
        ntt_result_final[i] = generator.add(other_term[i], other_term[i + BATCH]);
        print_array(ntt_result_final[i].data(), "ntt_result_final[" + std::to_string(i) + "]");
    }
    // INTT操作
    cout << endl << "4.INTT( Karatsuba( ntt(input1), ntt(input2) ) ) " << endl;
    vector<vector<TestDataType>> result_final(BATCH);
    for (int i = 0; i < BATCH; ++i)
    {
        result_final[i] = generator.intt(ntt_result_final[i]);
        print_array(result_final[i].data(), "result_final[" + std::to_string(i) + "]");
    }
    // 合并结果
    vector<TestDataType> result_merged;
    for (const auto &part : result_final)
    {
        result_merged.insert(result_merged.end(), part.begin(), part.end());
    }
    print_array(result_merged.data(), "result_merged");



    // 合并输入验证
    cout << endl << "5.合并输入验证 merge(input1), merge(input2)" << endl;
    vector<TestDataType> merged_input1, merged_input2;
    for (const auto &part : input1)
    {
        merged_input1.insert(merged_input1.end(), part.begin(), part.end());
    }
    print_array(merged_input1.data(), "merged_input1");
    for(const auto &part : input2)
    {
        merged_input2.insert(merged_input2.end(), part.begin(), part.end());
    }
    print_array(merged_input2.data(), "merged_input2");

    // NTT 乘法验证
    NTTParameters<TestDataType> parameters_merged(LOGN + alpha, ReductionPolynomial::X_N_minus);
    // parameters_merged.modulus.value = 268460033;
    // cout << "merged模数 = " << parameters_merged.modulus.value << endl;
    cout << endl << "INTT( NTT(merged_input1) * NTT(merged_input2) ) " << endl;
    NTTCPU<TestDataType> generator_merged(parameters_merged);
    vector<TestDataType> merged_ntt_result1 = generator_merged.ntt(merged_input1);
    vector<TestDataType> merged_ntt_result2 = generator_merged.ntt(merged_input2); 
    vector<TestDataType> merged_ntt_result = generator_merged.mult(merged_ntt_result1, merged_ntt_result2);
    vector<TestDataType> intt_merged_ntt_result = generator_merged.intt(merged_ntt_result);
    print_array(intt_merged_ntt_result.data(), "intt_merged_ntt_result");
    // cout << "merged_input1[n-1]:" << merged_input1[parameters.n-1] << endl;
    // 执行schoolbook乘法
    cout << endl << "schoolbook_poly_multiplication( merged_input1, merged_input2 )" << endl;
    vector<TestDataType> merged_schoolbook_result = schoolbook_poly_multiplication<TestDataType>(
    merged_input1, 
    merged_input2,
    parameters.modulus,
    parameters.poly_reduction  // 来自 NTTParameters
    );
    print_array(merged_schoolbook_result.data(), "merged_schoolbook_result");
    // 当Batch = 2， 计算ntt_result[0] * ntt_result[1]    
    {   
        // vector<TestDataType> ntt_mult_result(parameters.n);
        // { 
        //     // for (int i = 0; i < parameters.n; i++) {
        //     //     // ntt_mult_result[i] = OPERATOR<TestDataType>::mult(input1[0][i], input1[1][i], parameters.modulus);
        //     //     unsigned long long ref = (unsigned long long)ntt_result[0][i] * ntt_result[1][i];
        //     //     ntt_mult_result[i] = ref % parameters.modulus.value;
        //     // }
        // }
        // ntt_mult_result = generator.mult(ntt_result[0], ntt_result[1]);
        // // print_array(ntt_mult_result.data(), "ntt_mult_result");

        // // 执行CPU端乘法后INTT（生成参考结果）
        // vector<TestDataType> mult_result = generator.intt(ntt_mult_result);
        // // print_array(mult_result.data(), "mult_result");

        // // 执行schoolbook乘法
        // {   
        //     // vector<TestDataType> schoolbook_result = schoolbook_poly_multiplication<TestDataType>(
        //     // input1[0], 
        //     // input1[1],
        //     // parameters.modulus,
        //     // parameters.poly_reduction  // 来自 NTTParameters
        //     // );
        //     // print_array(schoolbook_result.data(), "schoolbook_result");
        // }
            // 执行CPU端INTT（生成参考结果）
        // vector<vector<TestDataType>> intt_result(BATCH);
        // for (int i = 0; i < BATCH; i++) {
        //     intt_result[i] = generator.intt(ntt_result[i]);  // CPU INTT计算
        //     // print_array(intt_result[i].data(), "intt_result[" + std::to_string(i) + "]");
        // }
    }
    


    



    cout << "-----------------GPU 计算-----------------" << endl;
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

    // GPU内存分配与数据传输-----------------------------------------------
    TestDataType* InOut_Datas, *InOut_Datas2;  // GPU输入/输出数据指针
    GPUNTT_CUDA_CHECK(  // 带错误检查的CUDA内存分配
        cudaMalloc(&InOut_Datas, BATCH * parameters.n * sizeof(TestDataType)));
    GPUNTT_CUDA_CHECK(  // 带错误检查的CUDA内存分配
        cudaMalloc(&InOut_Datas2, BATCH * parameters.n * sizeof(TestDataType)));
    
    // 分批拷贝数据到GPU
    for (int j = 0; j < BATCH; j++) {
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(InOut_Datas + (parameters.n * j),  // 目标地址
                       input1[j].data(),                   // 源数据
                       parameters.n * sizeof(TestDataType),// 数据大小
                       cudaMemcpyHostToDevice));           // 传输方向
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(InOut_Datas2 + (parameters.n * j) ,  // 目标地址
                       input2[j].data(),                   // 源数据
                       parameters.n * sizeof(TestDataType),// 数据大小
                       cudaMemcpyHostToDevice));           
    }

    // 执行GPU NTT--------------------------------------------------------
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
    {
    //     // 回传结果并验证-----------------------------------------------------
    //     TestDataType* Output_Host =  // 主机端结果缓冲区
    //         (TestDataType*)malloc(BATCH * parameters.n * sizeof(TestDataType));
        
    //     GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host,   // 目标地址
    //                             InOut_Datas,   // 源数据
    //                             BATCH * parameters.n * sizeof(TestDataType),
    //                             cudaMemcpyDeviceToHost));
    //     print_array(Output_Host, "GPU_NTT(input1)");
    //     // 结果验证
    //     VERIFY_RESULTS(Output_Host, ntt_result, "NTT结果正确");
    }

    {
        // bool check = true;
        // for (int i = 0; i < BATCH; i++) {
        //     check = check_result(  // 自定义结果验证函数
        //         Output_Host + (i * parameters.n),  // GPU结果位置
        //         ntt_result[i].data(),              // CPU参考结果
        //         parameters.n                       // 数据长度
        //     );

        //     if (!check) {
        //         cout << "第 " << i << " 个多项式验证失败" << endl;
        //         break;
        //     }
        // }
        // if (check) cout << "NTT结果正确" << endl;
    }
    // 处理对角线项 (i = j) diagonal term
    TestDataType* diag_term_device;
    GPUNTT_CUDA_CHECK(cudaMalloc(&diag_term_device, BATCH * parameters.n * sizeof(TestDataType)));
        PointwiseMultiply<TestDataType>(
        InOut_Datas,                      // 第一个多项式地址
        InOut_Datas2,                     // 第二个多项式地址
        diag_term_device,                 // 输出地址
        parameters.modulus,               // 模数参数
        BATCH * parameters.n,             // 多项式长度
        1,                                // 批量数
        0                                 // 使用默认流
    );
    //验证
    TestDataType* Output_Host =  // 主机端结果缓冲区
    (TestDataType*)malloc(BATCH * parameters.n * sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host,   // 目标地址
                               diag_term_device,   // 源数据
                               BATCH * parameters.n * sizeof(TestDataType),
                               cudaMemcpyDeviceToHost));
    // print_array(Output_Host, "GPU对角线项结果");
    VERIFY_RESULTS(Output_Host, diag_term, "GPU计算对角线项结果正确");

    // 处理其它项
    TestDataType* other_term_device,  *tmp_device1, *tmp_device2, *tmp_device1_2;
    GPUNTT_CUDA_CHECK(cudaMalloc(&other_term_device, 2 * BATCH * parameters.n * sizeof(TestDataType)));
    GPUNTT_CUDA_CHECK(cudaMemset(other_term_device, 0, 2 * BATCH * parameters.n * sizeof(TestDataType))); // 关键初始化
    GPUNTT_CUDA_CHECK(cudaMalloc(&tmp_device1, parameters.n * sizeof(TestDataType)));   // 暂存计算中间项
    GPUNTT_CUDA_CHECK(cudaMalloc(&tmp_device2, parameters.n * sizeof(TestDataType)));
    GPUNTT_CUDA_CHECK(cudaMalloc(&tmp_device1_2, parameters.n * sizeof(TestDataType)));
    for(int i = 0; i < BATCH; i++)
    {
        for(int j = i + 1; j < BATCH; j++)
        {
            TestDataType* fi = InOut_Datas + i * parameters.n;      // 表示fi
            TestDataType* fj = InOut_Datas + j * parameters.n;      // 表示fj
            PointwiseAdd(fi, fj, tmp_device1, parameters.modulus, parameters.n);    // fi + fj
            TestDataType* gi = InOut_Datas2 + i * parameters.n;     // 表示gi
            TestDataType* gj = InOut_Datas2 + j * parameters.n;     // 表示gj
            PointwiseAdd(gi, gj, tmp_device2, parameters.modulus, parameters.n);    // gi + gj
            PointwiseMultiply(tmp_device1, tmp_device2, tmp_device1_2, parameters.modulus, parameters.n);    // (fi + fj) * (gi + gj)
            TestDataType* diag_term_i = diag_term_device + i * parameters.n;    // 表示 fi * gi
            TestDataType* diag_term_j = diag_term_device + j * parameters.n;    // 表示 fj * gj
            PointwiseSubtract(tmp_device1_2, diag_term_i, tmp_device1_2, parameters.modulus, parameters.n); // (fi + fj) * (gi + gj) - fi * gi
            PointwiseSubtract(tmp_device1_2, diag_term_j, tmp_device1_2, parameters.modulus, parameters.n); // (fi + fj) * (gi + gj) - fi * gi - fj * gj
            TestDataType* other_term_device_i_j = other_term_device + ((i+j) * parameters.n);
            PointwiseAdd(other_term_device_i_j, tmp_device1_2, other_term_device_i_j, parameters.modulus, parameters.n);
            // 验证
            // 在关键步骤后添加打印
            {   
                // TestDataType* debug_host = (TestDataType*)malloc(parameters.n * sizeof(TestDataType));
                // GPUNTT_CUDA_CHECK(cudaMemcpy(debug_host, tmp_device1_2, parameters.n*sizeof(TestDataType), cudaMemcpyDeviceToHost));
                // print_array(debug_host, "tmp_device1_2");
            }
        }
    }
    GPUNTT_CUDA_CHECK(cudaGetLastError());
    // 对角线项加上其它项
    for(int i = 0; i < BATCH; i++)
    {
        //  diagonal_term[i] + other_term[2i];
        TestDataType* other_term_2i = other_term_device + 2 * i * parameters.n;  // 表示 other_term[2i]
        TestDataType* diag_term_i = diag_term_device + i * parameters.n;         // 表示 diagonal_term[i]
        PointwiseAdd(other_term_2i, diag_term_i, other_term_2i, parameters.modulus, parameters.n);    // 表示 other_term[2i] = other_term[2i] + diagonal_term[i]
    }
    // 验证other_term计算结果
    TestDataType* host_other_term = (TestDataType*)malloc(2*BATCH*parameters.n*sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(host_other_term, other_term_device, 
                            2*BATCH*parameters.n*sizeof(TestDataType),
                            cudaMemcpyDeviceToHost));
    // print_array(host_other_term, "GPU 计算其它项结果");
    VERIFY_RESULTS(host_other_term, other_term, "GPU计算其它项结果正确");

    // 合并结果
    TestDataType* ntt_result_device;
    GPUNTT_CUDA_CHECK(cudaMalloc(&ntt_result_device, BATCH * parameters.n * sizeof(TestDataType)));
    GPUNTT_CUDA_CHECK(cudaMemset(ntt_result_device, 0, BATCH * parameters.n * sizeof(TestDataType)));
    for(int i = 0; i < BATCH; i++)
    {
        // other_term[i] + other_term[i + BATCH] 
        TestDataType* other_term_i = other_term_device + i * parameters.n;
        TestDataType* other_term_i_BATCH = other_term_device + (i + BATCH) * parameters.n;
        TestDataType* ntt_result_i = ntt_result_device + i * parameters.n;
        PointwiseAdd(other_term_i, other_term_i_BATCH, ntt_result_i, parameters.modulus, parameters.n);
    }
    // 验证
    TestDataType* host_ntt_result = (TestDataType*)malloc(BATCH*parameters.n*sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(host_ntt_result, ntt_result_device, 
                            BATCH*parameters.n*sizeof(TestDataType),
                            cudaMemcpyDeviceToHost));
    // print_array(host_ntt_result, "GPU 计算合并结果");
    VERIFY_RESULTS(host_ntt_result, ntt_result_final, "GPU计算合并结果正确");

    // 配置INTT模数参数-------------------------------------------------------
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
    
    // 执行GPU_INTT( ntt_result_device )--------------------------------------------------------
    TestDataType* result_device;
    GPUNTT_CUDA_CHECK(cudaMalloc(&result_device, BATCH * parameters.n * sizeof(TestDataType))); 
    GPU_NTT(ntt_result_device, result_device, Inverse_Omega_Table_Device, test_modulus, cfg_intt, BATCH, 1);
    // 验证
    TestDataType* host_result = (TestDataType*)malloc(BATCH * parameters.n * sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(host_result,   // 目标地址
                            result_device,   // 源数据
                            BATCH * parameters.n * sizeof(TestDataType),
                            cudaMemcpyDeviceToHost));
    VERIFY_RESULTS(host_result, result_final, "Karatsuba_GPU计算结果正确");

    { 
    //     // 执行GPU INTT--------------------------------------------------------
    //     TestDataType* Out_Datas;
    //     GPUNTT_CUDA_CHECK(
    //         cudaMalloc(&Out_Datas, BATCH * parameters.n * sizeof(TestDataType))); 

    //     GPU_NTT(InOut_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus,
    //             cfg_intt, BATCH, 1);

    //     // 回传结果并验证-----------------------------------------------------
    //     GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host,   // 目标地址
    //                             Out_Datas,   // 源数据
    //                             BATCH * parameters.n * sizeof(TestDataType),
    //                             cudaMemcpyDeviceToHost));

    //     // 结果验证
    //     bool check = true;
    //     for (int i = 0; i < BATCH; i++) {
    //         check = check_result(  // 自定义结果验证函数
    //             Output_Host + (i * parameters.n),  // GPU结果位置
    //             intt_result[i].data(),             // CPU参考结果
    //             parameters.n                       // 数据长度
    //         );

    //         if (!check) {
    //             cout << "第 " << i << " 个多项式验证失败" <<endl;
    //         }
    //     }
    //     if (check) cout << "INTT结果正确" << endl;


    //     // GPU内存分配与数据传输-----------------------------------------------
    //     TestDataType* NTT_Mult_Result_GPU;  // GPU输入/输出数据指针
    //     GPUNTT_CUDA_CHECK(  // 带错误检查的CUDA内存分配
    //         cudaMalloc(&NTT_Mult_Result_GPU, parameters.n * sizeof(TestDataType)));
    //     // 计算 NTT乘法结果 NTT_Mult_Result_GPU = &InOut_Datas * &(InOut_Datas + parameters.n)


    //     // 在GPU NTT计算之后添加点乘调用
    //     PointwiseMultiply<TestDataType>(
    //         InOut_Datas,                      // 第一个多项式地址
    //         InOut_Datas + parameters.n,       // 第二个多项式地址（BATCH=2时偏移）
    //         NTT_Mult_Result_GPU,              // 输出地址
    //         parameters.modulus,               // 模数参数
    //         parameters.n,                     // 多项式长度
    //         1,                                // 批量数
    //         0                                 // 使用默认流
    //     );

    //     // 添加结果验证代码
    //     TestDataType* mult_host_result = (TestDataType*)malloc(parameters.n * sizeof(TestDataType));
    //     GPUNTT_CUDA_CHECK(cudaMemcpy(mult_host_result, NTT_Mult_Result_GPU, 
    //                             parameters.n * sizeof(TestDataType),
    //                             cudaMemcpyDeviceToHost));

    //     // 与CPU结果对比
    //     check = check_result(mult_host_result, 
    //                     ntt_mult_result.data(),
    //                     parameters.n);
    //     if (check) cout << "点乘结果正确" << endl;

    //     // INTT点乘结果验证
    //     TestDataType* Mult_Result_GPU;      
    //     GPUNTT_CUDA_CHECK(
    //         cudaMalloc(&Mult_Result_GPU, BATCH * parameters.n * sizeof(TestDataType)));
    //     // INTT( NTT_Mult_Result_GPU )
    //     GPU_NTT(NTT_Mult_Result_GPU, Mult_Result_GPU, Inverse_Omega_Table_Device, test_modulus,
    //             cfg_intt, 1, 1);

    //     // Mult_Result_GPU 与 CPU 结果对比
    //     GPUNTT_CUDA_CHECK(cudaMemcpy(mult_host_result, Mult_Result_GPU, 
    //                         parameters.n * sizeof(TestDataType),
    //                         cudaMemcpyDeviceToHost));
    //     check = check_result(mult_host_result, 
    //                     mult_result.data(),
    //                     parameters.n);
    //     // print_array(mult_host_result, "GPU NTT乘法结果");
    //     if (check) cout << "NTT乘法结果正确" << endl;
    }

    // 资源释放-----------------------------------------------------------
    GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas2));

    GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
    GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
    free(Output_Host);

    // GPUNTT_CUDA_CHECK(cudaFree(NTT_Mult_Result_GPU));
    // free(mult_host_result);
    // GPUNTT_CUDA_CHECK(cudaFree(Mult_Result_GPU));

    GPUNTT_CUDA_CHECK(cudaFree(diag_term_device));
    GPUNTT_CUDA_CHECK(cudaFree(other_term_device));


    return EXIT_SUCCESS;
}