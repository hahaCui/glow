set(TEST_SHADER_SRC ${CMAKE_BINARY_DIR}/test_shaders.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


#set(SAMPLE_SHADER_SRC ${CMAKE_BINARY_DIR}/sample_shaders.cpp)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#
#COMPILE_SHADERS(
#        ${SAMPLE_SHADER_SRC}
#        shader/test_image.frag
#        shader/quad.geom
#        shader/empty.vert
#       )

add_executable(process_image process_image.cpp)
target_link_libraries(process_image ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )