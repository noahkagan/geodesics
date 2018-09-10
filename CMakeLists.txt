project(geodesics)
cmake_minimum_required(VERSION 3.0)

include_directories(tinyobjloader)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(geodesics geodesics.cpp trackball.cpp)
target_link_libraries(geodesics
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  glfw
)
install(TARGETS geodesics DESTINATION bin)
