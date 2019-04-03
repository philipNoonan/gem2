cmake_minimum_required(VERSION 2.8)

project(gem)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(BasicConfig)

find_package(Aruco REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE SRC "src/*.cpp")
file(GLOB_RECURSE HDR "include/*.h")
list(APPEND SRC ${HDR})

add_library(${PROJECT_NAME} STATIC ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_PDB_NAME_DEBUG "${PROJECT_NAME}${CMAKE_DEBUG_POSTFIX}")

target_include_directories(${PROJECT_NAME} PUBLIC "include")
target_include_directories(${PROJECT_NAME} PUBLIC ${GLM_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${ARUCO_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC ${ARUCO_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
