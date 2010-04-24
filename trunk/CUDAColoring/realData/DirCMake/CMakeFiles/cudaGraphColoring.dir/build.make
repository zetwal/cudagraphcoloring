# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/local/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake

# Include any dependencies generated for this target.
include CMakeFiles/cudaGraphColoring.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cudaGraphColoring.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cudaGraphColoring.dir/flags.make

CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o: CMakeFiles/cudaGraphColoring.dir/flags.make
CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o: /home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o -c /home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp

CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp > CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.i

CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp -o CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.s

CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.requires:
.PHONY : CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.requires

CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.provides: CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.requires
	$(MAKE) -f CMakeFiles/cudaGraphColoring.dir/build.make CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.provides.build
.PHONY : CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.provides

CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.provides.build: CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o
.PHONY : CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.provides.build

# Object files for target cudaGraphColoring
cudaGraphColoring_OBJECTS = \
"CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o"

# External object files for target cudaGraphColoring
cudaGraphColoring_EXTERNAL_OBJECTS =

cudaGraphColoring: CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o
cudaGraphColoring: /usr/local/cuda/lib64/libcudart.so
cudaGraphColoring: /usr/lib/libcuda.so
cudaGraphColoring: libCUDAGraphColoring.so
cudaGraphColoring: /usr/local/cuda/lib64/libcudart.so
cudaGraphColoring: /usr/lib/libcuda.so
cudaGraphColoring: /usr/local/cuda/lib64/libcudart.so
cudaGraphColoring: /usr/lib/libcuda.so
cudaGraphColoring: CMakeFiles/cudaGraphColoring.dir/build.make
cudaGraphColoring: CMakeFiles/cudaGraphColoring.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable cudaGraphColoring"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cudaGraphColoring.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cudaGraphColoring.dir/build: cudaGraphColoring
.PHONY : CMakeFiles/cudaGraphColoring.dir/build

CMakeFiles/cudaGraphColoring.dir/requires: CMakeFiles/cudaGraphColoring.dir/home/forrest/GraphColoring/CUDAColoring/realData/graphDriver.cpp.o.requires
.PHONY : CMakeFiles/cudaGraphColoring.dir/requires

CMakeFiles/cudaGraphColoring.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cudaGraphColoring.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cudaGraphColoring.dir/clean

CMakeFiles/cudaGraphColoring.dir/depend:
	cd /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake /home/forrest/GraphColoring/CUDAColoring/realData/DirCMake/CMakeFiles/cudaGraphColoring.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cudaGraphColoring.dir/depend

