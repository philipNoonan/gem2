cmake_minimum_required(VERSION 3.8)

# Toolchain file needs to be set before project() 
### FIND VCPKG
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
else()
	message(STATUS "VCPKG_ROOT not set in environment variables")	  
endif()


project(gem2)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(BasicConfig)

find_package(aruco REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(eigen3 REQUIRED)

file(GLOB_RECURSE SRC "src/*.cpp")
file(GLOB_RECURSE HDR "include/*.h")
list(APPEND SRC ${HDR})

add_library(${PROJECT_NAME} SHARED ${SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_PDB_NAME_DEBUG "${PROJECT_NAME}${CMAKE_DEBUG_POSTFIX}")

target_include_directories(${PROJECT_NAME} PUBLIC "include")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${aruco_LIBS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} glm)


CONFIGURE_FILE(gem2.cmake.in "${PROJECT_BINARY_DIR}/gem2Config.cmake" @ONLY)

INSTALL(TARGETS gem2 DESTINATION lib RUNTIME DESTINATION bin)
INSTALL(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION include PATTERN "*.in" EXCLUDE)
#INSTALL(DIRECTORY "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" DESTINATION include)
INSTALL(FILES "${PROJECT_BINARY_DIR}/gem2Config.cmake" DESTINATION share/gem2/)
#INSTALL(FILES "${PROJECT_BINARY_DIR}/gem2ConfigVersion.cmake" DESTINATION lib/cmake/gem2/)
