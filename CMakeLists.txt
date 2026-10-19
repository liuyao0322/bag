cmake_minimum_required(VERSION 3.24)
project(bag VERSION 0.1.0 LANGUAGES CXX)

option(BAG_ENABLE_CUDA "Build the CUDA implementation" ON)
option(BAG_BUILD_BENCHMARKS "Build CUDA benchmarks and validation programs" ON)

if(BAG_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(bag_cuda INTERFACE)
    add_library(bag::cuda ALIAS bag_cuda)
    target_include_directories(bag_cuda INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_features(bag_cuda INTERFACE cxx_std_17 cuda_std_17)
    target_link_libraries(bag_cuda INTERFACE CUDA::cudart)

    if(BAG_BUILD_BENCHMARKS)
        foreach(program IN ITEMS fp32_baseline strassen bag)
            add_executable(benchmark_${program} benchmarks/cuda/${program}.cu)
            target_link_libraries(benchmark_${program} PRIVATE bag::cuda CUDA::cublas)
            target_compile_options(benchmark_${program} PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>")
        endforeach()

        include(CTest)
        if(BUILD_TESTING)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_results)
            add_test(NAME cuda_bag_correctness COMMAND benchmark_bag
                --sizes 24,128 --depths 1,2 --basis both --schedule all
                --seed-offset 1009 --validate-only
                --output ${CMAKE_CURRENT_BINARY_DIR}/test_results/bag)
            add_test(NAME cuda_strassen_correctness COMMAND benchmark_strassen
                --mode endtoend --sizes 128 --backends column_tile64_k16_r4x8
                --schedules four_batched_atomic --seed-offset 1009 --validate-only
                --output ${CMAKE_CURRENT_BINARY_DIR}/test_results/strassen)
            set_tests_properties(cuda_bag_correctness cuda_strassen_correctness
                PROPERTIES RESOURCE_LOCK gpu TIMEOUT 300)
        endif()
    endif()

    include(GNUInstallDirs)
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    set_property(TARGET bag_cuda PROPERTY EXPORT_NAME cuda)
    install(TARGETS bag_cuda EXPORT bagTargets)
    install(EXPORT bagTargets NAMESPACE bag::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bag)
    include(CMakePackageConfigHelpers)
    configure_package_config_file(cmake/bagConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/bagConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bag)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/bagConfigVersion.cmake
        VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bagConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/bagConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bag)
endif()
