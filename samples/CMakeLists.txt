set(TEST_SHADER_SRC ${CMAKE_BINARY_DIR}/test_shaders.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



find_package(OpenCV REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS} )

add_executable(process_image process_image.cpp)
target_link_libraries(process_image ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(vao_buffer vao_buffer.cpp)
target_link_libraries(vao_buffer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(multi_samplers multi_samplers.cpp)
target_link_libraries(multi_samplers ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(transform_feedback transform_feedback.cpp)
target_link_libraries(transform_feedback ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )
