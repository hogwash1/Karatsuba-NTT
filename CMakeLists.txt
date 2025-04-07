# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.26)

# 定义项目名称和支持的语言
project(my_project LANGUAGES CXX CUDA)

# 查找 CUDA 工具包
find_package(CUDAToolkit REQUIRED)

# 查找 GPU-NTT 库
find_package(GPUNTT)

# 添加可执行文件
add_executable(my_project src/main.cu)

# 链接 GPU-NTT 库和 CUDA 运行时库
target_link_libraries(my_project PRIVATE GPUNTT::ntt CUDA::cudart)

# 设置目标属性
set_target_properties(my_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)