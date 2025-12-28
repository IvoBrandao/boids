# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(Boids3D)

set(OpenGL_GL_PREFERENCE GLVND)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLU REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GLEW REQUIRED)

add_executable(boids main.cpp)
if(TARGET GLU::GLU)
	target_link_libraries(boids OpenGL::GL GLUT::GLUT GLU::GLU)
else()
	# Fall back to the variable exposed by the FindGLU module if imported target isn't available
	target_link_libraries(boids OpenGL::GL GLUT::GLUT ${GLU_LIBRARY})
	include_directories(${GLU_INCLUDE_DIR})
endif()
if(OpenMP_CXX_FOUND)
	target_link_libraries(boids OpenMP::OpenMP_CXX)
endif()
if(TARGET GLEW::GLEW)
	target_link_libraries(boids GLEW::GLEW)
else()
	target_link_libraries(boids GLEW)
endif()