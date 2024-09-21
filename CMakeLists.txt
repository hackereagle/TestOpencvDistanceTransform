cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME TestOpencvDistanceTransform)
project(${PROJECT_NAME})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

aux_source_directory(. SRCS)
add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} ${CMAKE_DL_LIBS} ${OpenCV_LIBS})