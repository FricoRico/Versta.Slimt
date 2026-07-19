set(LEANMT_PUBLIC_HEADERS
    leanmt.hh
    Aligned.hh
    Annotation.hh
    Batcher.hh
    Cache.hh
    Frontend.hh
    Io.hh
    Model.hh
    Modules.hh
    Response.hh
    Shortlist.hh
    Tensor.hh
    TextProcessor.hh
    Transformer.hh
    Types.hh
    Vocabulary.hh
    Utils.hh)

# Configure generated headers. Version.hh can be configured here.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Version.hh.in
               ${CMAKE_CURRENT_BINARY_DIR}/Version.hh @ONLY)

# Version.hh is configured above.
set(LEANMT_GENERATED_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/Version.hh)

set(LEANMT_PRIVATE_HEADERS
    QMM.hh
    Request.hh
    TensorOps.hh
    Utils.hh)

set(LEANMT_SOURCES
    Aligned.cc
    Annotation.cc
    Arena.cc
    Batcher.cc
    Frontend.cc
    Input.cc
    Io.cc
    Model.cc
    Modules.cc
    QMM.cc
    Request.cc
    Response.cc
    Shortlist.cc
    Tensor.cc
    TensorOps.cc
    TextProcessor.cc
    Transformer.cc
    Utils.cc
    Vocabulary.cc)

set(LEANMT_LIBRARIES)

add_library(leanmt STATIC ${LEANMT_SOURCES} ${LEANMT_PUBLIC_HEADERS}
                        ${LEANMT_PRIVATE_HEADERS})
list(APPEND LEANMT_LIBRARIES leanmt)

foreach(LEANMT_LIB IN LISTS LEANMT_LIBRARIES)
  set_target_properties(${LEANMT_LIB} PROPERTIES OUTPUT_NAME "leanmt")

  target_link_libraries(
    ${LEANMT_LIB}
    PUBLIC ${LEANMT_PUBLIC_LIBS}
    INTERFACE "$<BUILD_INTERFACE:${LEANMT_INTERFACE_LIBS}>"
    PRIVATE "$<BUILD_INTERFACE:${LEANMT_PRIVATE_LIBS}>")

  target_include_directories(
    ${LEANMT_LIB}
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>"
      "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>"
      "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>"
  )

  target_link_options(${LEANMT_LIB} PUBLIC ${LEANMT_LINK_OPTIONS})
  target_compile_options(${LEANMT_LIB} PRIVATE ${LEANMT_COMPILE_OPTIONS})
  target_compile_definitions(${LEANMT_LIB} PRIVATE ${LEANMT_COMPILE_DEFINITIONS})
endforeach()

if(UNIX)
  install(FILES ${LEANMT_PUBLIC_HEADERS}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/leanmt)
  install(FILES ${LEANMT_GENERATED_HEADERS}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/leanmt/)
  install(TARGETS ${LEANMT_LIBRARIES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(UNIX)
