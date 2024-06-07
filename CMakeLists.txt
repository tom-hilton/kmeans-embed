cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

project(kmeans_embed)

# define empty list of libraries to link
set(PROJECT_LIBS "")

find_package( GTest REQUIRED )
find_package( OpenMP REQUIRED )
find_package( OpenCV REQUIRED )
find_package( HighFive REQUIRED )
find_package( HDF5 REQUIRED )
find_package( Armadillo REQUIRED )

enable_testing()

add_executable(kmeans_embed src/main.cpp)

target_link_libraries(kmeans_embed PRIVATE ${OpenCV_LIBS})
target_link_libraries(kmeans_embed PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(kmeans_embed PRIVATE HDF5::HDF5)
target_link_libraries(kmeans_embed PRIVATE HighFive)
target_link_libraries(kmeans_embed PRIVATE ${ARMADILLO_LIBRARIES})