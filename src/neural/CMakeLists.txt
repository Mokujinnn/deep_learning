cmake_minimum_required(VERSION 3.14)

set(target_name neural)

file(GLOB CPPSRC "src/*.cpp")
file(GLOB HPPSRC "include/*.hpp")

add_library(${target_name} STATIC ${CPPSRC} ${HPPSRC})

target_include_directories(${target_name} PRIVATE include/neural)
target_include_directories(${target_name} PUBLIC include)