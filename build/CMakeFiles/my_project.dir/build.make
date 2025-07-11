# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/admin-txl/mydirectory1/my_ntt_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/admin-txl/mydirectory1/my_ntt_project/build

# Include any dependencies generated for this target.
include CMakeFiles/my_project.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/my_project.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/my_project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/my_project.dir/flags.make

CMakeFiles/my_project.dir/src/main.cu.o: CMakeFiles/my_project.dir/flags.make
CMakeFiles/my_project.dir/src/main.cu.o: CMakeFiles/my_project.dir/includes_CUDA.rsp
CMakeFiles/my_project.dir/src/main.cu.o: /home/admin-txl/mydirectory1/my_ntt_project/src/main.cu
CMakeFiles/my_project.dir/src/main.cu.o: CMakeFiles/my_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/admin-txl/mydirectory1/my_ntt_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/my_project.dir/src/main.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/my_project.dir/src/main.cu.o -MF CMakeFiles/my_project.dir/src/main.cu.o.d -x cu -rdc=true -c /home/admin-txl/mydirectory1/my_ntt_project/src/main.cu -o CMakeFiles/my_project.dir/src/main.cu.o

CMakeFiles/my_project.dir/src/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/my_project.dir/src/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/my_project.dir/src/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/my_project.dir/src/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target my_project
my_project_OBJECTS = \
"CMakeFiles/my_project.dir/src/main.cu.o"

# External object files for target my_project
my_project_EXTERNAL_OBJECTS =

CMakeFiles/my_project.dir/cmake_device_link.o: CMakeFiles/my_project.dir/src/main.cu.o
CMakeFiles/my_project.dir/cmake_device_link.o: CMakeFiles/my_project.dir/build.make
CMakeFiles/my_project.dir/cmake_device_link.o: /usr/local/lib/libntt-1.0.a
CMakeFiles/my_project.dir/cmake_device_link.o: /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart.so
CMakeFiles/my_project.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/librt.a
CMakeFiles/my_project.dir/cmake_device_link.o: CMakeFiles/my_project.dir/deviceLinkLibs.rsp
CMakeFiles/my_project.dir/cmake_device_link.o: CMakeFiles/my_project.dir/deviceObjects1.rsp
CMakeFiles/my_project.dir/cmake_device_link.o: CMakeFiles/my_project.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/admin-txl/mydirectory1/my_ntt_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/my_project.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_project.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/my_project.dir/build: CMakeFiles/my_project.dir/cmake_device_link.o
.PHONY : CMakeFiles/my_project.dir/build

# Object files for target my_project
my_project_OBJECTS = \
"CMakeFiles/my_project.dir/src/main.cu.o"

# External object files for target my_project
my_project_EXTERNAL_OBJECTS =

my_project: CMakeFiles/my_project.dir/src/main.cu.o
my_project: CMakeFiles/my_project.dir/build.make
my_project: /usr/local/lib/libntt-1.0.a
my_project: /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart.so
my_project: /usr/lib/x86_64-linux-gnu/librt.a
my_project: CMakeFiles/my_project.dir/cmake_device_link.o
my_project: CMakeFiles/my_project.dir/linkLibs.rsp
my_project: CMakeFiles/my_project.dir/objects1.rsp
my_project: CMakeFiles/my_project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/admin-txl/mydirectory1/my_ntt_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable my_project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/my_project.dir/build: my_project
.PHONY : CMakeFiles/my_project.dir/build

CMakeFiles/my_project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/my_project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/my_project.dir/clean

CMakeFiles/my_project.dir/depend:
	cd /home/admin-txl/mydirectory1/my_ntt_project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/admin-txl/mydirectory1/my_ntt_project /home/admin-txl/mydirectory1/my_ntt_project /home/admin-txl/mydirectory1/my_ntt_project/build /home/admin-txl/mydirectory1/my_ntt_project/build /home/admin-txl/mydirectory1/my_ntt_project/build/CMakeFiles/my_project.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/my_project.dir/depend

