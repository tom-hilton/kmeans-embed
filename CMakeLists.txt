cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

project(vector_quant)

# define empty list of libraries to link
set(PROJECT_LIBS "")

find_package( GTest REQUIRED )
find_package( OpenMP REQUIRED )
find_package( OpenCV REQUIRED )
find_package( HighFive REQUIRED )
find_package( HDF5 REQUIRED )

enable_testing()

add_executable(vector_quant src/main.cpp)

target_link_libraries(vector_quant PRIVATE ${OpenCV_LIBS})
target_link_libraries(vector_quant PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(vector_quant PRIVATE HDF5::HDF5)
target_link_libraries(vector_quant PRIVATE HighFive)