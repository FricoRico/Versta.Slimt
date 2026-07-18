cmake_minimum_required(VERSION 3.10...3.27)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(slimt VERSION 1.0.0 LANGUAGES CXX C)
set(PROJECT_DESCRIPTION "Inference frontend for bergamot models.")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBLITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINE_HIDDEN YES)

option(WITH_ASAN "Enable AddressSanitizer in builds" OFF)

option(WITH_RUY "Use ruy" ON)

option(USE_BUILTIN_SENTENCEPIECE "Use SentencePiece supplied as 3rd-party" ON)

option(USE_SSE2 "Use SSE2" OFF)
option(USE_NEON "Use NEON" OFF)

option(BUILD_SHARED "Build shared libraries" ON)
option(BUILD_STATIC "Build static libraries" ON)

include(MacroEnsureOutOfSourceBuild)
macro_ensure_out_of_source_build(
  "${PROJECT_NAME} requires an out of source build. Please create a separate build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
)

set(SLIMT_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wno-unknown-pragmas
    -Werror
    -Wno-missing-field-initializers
    -Wno-unused-function
    -pedantic
    # -pedantic fires `;` on older systems, not on new. Might as well remove.
    # See: https://stackoverflow.com/q/77254582/4565794
    -Wextra-semi)

set(SLIMT_PUBLIC_LIBS "")
set(SLIMT_INTERFACE_LIBS "")
set(SLIMT_PRIVATE_LIBS "")
set(SLIMT_COMPILE_DEFINITIONS "")
set(SLIMT_LINK_OPTIONS "")

if(WITH_ASAN)
  list(APPEND SLIMT_COMPILE_OPTIONS -fsanitize=address -fsanitize=undefined)
  list(APPEND SLIMT_LINK_OPTIONS -fsanitize=address -fsanitize=undefined)
endif(WITH_ASAN)

if(WITH_TCMALLOC)
  find_library(
    TCMALLOC_LIBRARIES
    NAMES tcmalloc_minimal
    PATHS /usr/lib /usr/local/lib)

  if(TCMALLOC_LIBRARIES)
    message(STATUS "tcmalloc found: ${TCMALLOC_LIBRARIES}")
  else(TCMALLOC_LIBRARIES)
    message(FATAL_ERROR "tcmalloc not found")
  endif(TCMALLOC_LIBRARIES)

  list(APPEND SLIMT_PRIVATE_LIBS ${TCMALLOC_LIBRARIES})
endif(WITH_TCMALLOC)

add_subdirectory(3rd-party)

list(APPEND SLIMT_PUBLIC_LIBS SentencePiece::SentencePiece)

if(NOT USE_BUILTIN_SENTENCEPIECE)
  list(APPEND SLIMT_PRIVATE_LIBS SentencePiece::Protobuf)
endif(NOT USE_BUILTIN_SENTENCEPIECE)

# Single ruy backend (other backends removed).
list(APPEND SLIMT_COMPILE_DEFINITIONS SLIMT_HAS_RUY)
list(APPEND SLIMT_PRIVATE_LIBS ruy)

if(USE_SSE2)
  list(APPEND SLIMT_COMPILE_DEFINITIONS USE_SSE2)
  list(APPEND SLIMT_COMPILE_OPTIONS -msse2)
endif(USE_SSE2)

if(USE_NEON)
  list(APPEND SLIMT_COMPILE_DEFINITIONS USE_NEON)
endif(USE_NEON)

# cmake-format: off
set(CMAKE_CXX_FLAGS_PROFILE           "${CMAKE_CXX_FLAGS_RELEASE} -pg" CACHE STRING "Flags used by the C++ compiler during profile builds." FORCE)
set(CMAKE_C_FLAGS_PROFILE             "${CMAKE_C_FLAGS_RELEASE} -pg" CACHE STRING "Flags used by the C compiler during profile builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_PROFILE    "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -pg" CACHE STRING "Flags used for linking binaries during profile builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_PROFILE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}-pg" CACHE STRING "Flags used by the shared libraries linker during profile builds." FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_PROFILE
  CMAKE_C_FLAGS_PROFILE
  CMAKE_EXE_LINKER_FLAGS_PROFILE
  CMAKE_SHARED_LINKER_FLAGS_PROFILE
)
# cmake-format: on

message(STATUS "Project name: ${PROJECT_NAME}")
set(PROJECT_VERSION_STRING "${PROJECT_VERSION}")
set(PROJECT_VERSION_STRING_FULL "v${PROJECT_VERSION}")
message(STATUS "Project version: ${PROJECT_VERSION_STRING_FULL}")

add_subdirectory(slimt)
