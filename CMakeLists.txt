cmake_minimum_required(VERSION 3.8)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

project(Portnov_N_S)

include(build_prefix.cmake)

cmake_minimum_required(VERSION 2.8)
add_executable(Coursach Coursach.cpp)

link_directories(D:/programming/Anaconda1/envs/Coursach/libs)
include_directories(D:/programming/proga/pybind11/include/pybind11)
include_directories(D:/programming/Anaconda1/envs/Coursach/include)
include_directories(D:/programming/Anaconda1/envs/Coursach/libs)

target_link_libraries(Coursach D:/programming/Anaconda1/envs/Coursach/libs/python36.lib )

set_property(TARGET Coursach PROPERTY CXX_STANDARD 14)