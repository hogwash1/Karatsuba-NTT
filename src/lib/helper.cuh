#pragma once  // 添加头文件保护
#include "ntt.cuh"    // 包含 TestDataType 定义文件
#include <string>     // 添加 string 支持


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