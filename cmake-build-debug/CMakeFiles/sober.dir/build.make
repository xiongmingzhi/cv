# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "E:\Program Files\JetBrains\CLion 2020.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "E:\Program Files\JetBrains\CLion 2020.1\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Administrator\Desktop\cv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Administrator\Desktop\cv\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/sober.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sober.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sober.dir/flags.make

CMakeFiles/sober.dir/sober.cpp.obj: CMakeFiles/sober.dir/flags.make
CMakeFiles/sober.dir/sober.cpp.obj: CMakeFiles/sober.dir/includes_CXX.rsp
CMakeFiles/sober.dir/sober.cpp.obj: ../sober.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Administrator\Desktop\cv\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sober.dir/sober.cpp.obj"
	D:\mingw3264\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\sober.dir\sober.cpp.obj -c C:\Users\Administrator\Desktop\cv\sober.cpp

CMakeFiles/sober.dir/sober.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sober.dir/sober.cpp.i"
	D:\mingw3264\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Administrator\Desktop\cv\sober.cpp > CMakeFiles\sober.dir\sober.cpp.i

CMakeFiles/sober.dir/sober.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sober.dir/sober.cpp.s"
	D:\mingw3264\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\Administrator\Desktop\cv\sober.cpp -o CMakeFiles\sober.dir\sober.cpp.s

# Object files for target sober
sober_OBJECTS = \
"CMakeFiles/sober.dir/sober.cpp.obj"

# External object files for target sober
sober_EXTERNAL_OBJECTS =

sober.exe: CMakeFiles/sober.dir/sober.cpp.obj
sober.exe: CMakeFiles/sober.dir/build.make
sober.exe: CMakeFiles/sober.dir/linklibs.rsp
sober.exe: CMakeFiles/sober.dir/objects1.rsp
sober.exe: CMakeFiles/sober.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Administrator\Desktop\cv\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sober.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\sober.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sober.dir/build: sober.exe

.PHONY : CMakeFiles/sober.dir/build

CMakeFiles/sober.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\sober.dir\cmake_clean.cmake
.PHONY : CMakeFiles/sober.dir/clean

CMakeFiles/sober.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\Administrator\Desktop\cv C:\Users\Administrator\Desktop\cv C:\Users\Administrator\Desktop\cv\cmake-build-debug C:\Users\Administrator\Desktop\cv\cmake-build-debug C:\Users\Administrator\Desktop\cv\cmake-build-debug\CMakeFiles\sober.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sober.dir/depend
