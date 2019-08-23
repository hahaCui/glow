set(TEST_SHADER_SRC ${CMAKE_BINARY_DIR}/test_shaders.cpp)



add_executable(particle_se2_transform_feedback particle_se2_transform_feedback.cpp)
target_link_libraries(particle_se2_transform_feedback ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glow glow_util )
