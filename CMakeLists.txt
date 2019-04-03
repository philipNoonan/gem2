cmake_minimum_required(VERSION 3.8)

# Toolchain file needs to be set before project() 
### FIND VCPKG
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
else()
	message(STATUS "VCPKG_ROOT not set in environment variables")	  
endif()


project(gem2 VERSION 2.0)

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

target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${aruco_LIBS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} glm)

install(TARGETS ${PROJECT_NAME}
    EXPORT gem2Export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(EXPORT gem2Export FILE ${PROJECT_NAME}Config.cmake NAMESPACE ${PROJECT_NAME}:: DESTINATION share/${PROJECT_NAME})

#install(FILES " DESTINATION include)


#set ( GEM2_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${INCLUDE_INSTALL_DIR}" )
#set ( GEM2_ROOT_DIR ${CMAKE_INSTALL_PREFIX} )
#
#include(CMakePackageConfigHelpers)

#write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/gem2ConfigVersion.cmake"
#COMPATIBILITY AnyNewerVersion)

#configure_package_config_file(CMake/gem2Config.cmake.in gem2Config.cmake
#    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
#    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/bin
#    PATH_VARS ${GEM2_INCLUDE_DIR} ${GEM2_ROOT_DIR}
#)

#install(TARGETS ${PROJECT_NAME}
#    EXPORT gem2Targets
#    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gem2"
#)

#install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/gem2
#        DESTINATION ${GEM2_INCLUDE_DIR}
#)

#install(EXPORT gem2Targets
#        FILE gem2Targets.cmake
#        NAMESPACE ${PROJECT_NAME}::
#        DESTINATION ${CMAKECONFIG_INSTALL_DIR}
#)

#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gem2Config.cmake"
#        DESTINATION ${CMAKECONFIG_INSTALL_DIR}
#)

#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gem2ConfigVersion.cmake"
#        DESTINATION ${CMAKECONFIG_INSTALL_DIR}
#)


#CONFIGURE_FILE(gem2.cmake.in "${PROJECT_BINARY_DIR}/gem2Config.cmake" @ONLY)

#INSTALL(TARGETS gem2 DESTINATION lib RUNTIME DESTINATION bin)
#INSTALL(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION include PATTERN "*.in" EXCLUDE)
#INSTALL(FILES "${PROJECT_BINARY_DIR}/gem2Config.cmake" DESTINATION share/gem2/)

