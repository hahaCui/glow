set(TEST_SHADER_SRC ${CMAKE_BINARY_DIR}/test_shaders.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



find_package(OpenCV REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS} )

add_executable(upload_and_download upload_and_download.cpp)
target_link_libraries(upload_and_download ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(process_image process_image.cpp)
target_link_libraries(process_image ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )


add_executable(vao_buffer vao_buffer.cpp)
target_link_libraries(vao_buffer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(sampler_sample sampler_sample.cpp)
target_link_libraries(sampler_sample ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )


add_executable(multi_samplers multi_samplers.cpp)
target_link_libraries(multi_samplers ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(transform_feedback transform_feedback.cpp)
target_link_libraries(transform_feedback ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(index_buffer index_buffer.cpp)
target_link_libraries(index_buffer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(particle_se2_translation particle_se2_translation.cpp)
target_link_libraries(particle_se2_translation ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )


add_executable(detect_points_in_view detect_points_in_view.cpp)
target_link_libraries(detect_points_in_view ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )

add_executable(detect_points_in_view_frame_buffer detect_points_in_view_frame_buffer.cpp)
target_link_libraries(detect_points_in_view_frame_buffer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )



add_executable(detect_points_in_view_tf_fbo detect_points_in_view_tf_fbo.cpp)
target_link_libraries(detect_points_in_view_tf_fbo ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )


