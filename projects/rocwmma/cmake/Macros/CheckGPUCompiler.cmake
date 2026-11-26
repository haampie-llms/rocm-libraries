# Checks of the compiler that compiles the GPU sources: the HIP compiler when
# USE_HIPCXX is enabled, the C++ compiler otherwise.

include(CheckCXXCompilerFlag)
if(USE_HIPCXX)
  include(CheckCompilerFlag)
endif()

# Like check_cxx_compiler_flag
macro(rocwmma_check_gpu_compiler_flag FLAG RESULT_VAR)
  if(USE_HIPCXX)
    check_compiler_flag(HIP "${FLAG}" ${RESULT_VAR})
  else()
    check_cxx_compiler_flag("${FLAG}" ${RESULT_VAR})
  endif()
endmacro()

# Like rocm_check_target_ids
function(rocwmma_check_target_ids VARIABLE)
  cmake_parse_arguments(PARSE "" "" "TARGETS" ${ARGN})
  if(NOT USE_HIPCXX)
    rocm_check_target_ids(_supported_target_ids TARGETS ${PARSE_TARGETS})
  else()
    foreach(_target_id ${PARSE_TARGETS})
      string(REPLACE "+" "_on" _result_var "${_target_id}")
      string(REPLACE "-" "_off" _result_var "${_result_var}")
      string(MAKE_C_IDENTIFIER "HIP_COMPILER_HAS_TARGET_ID_${_result_var}" _result_var)
      check_compiler_flag(HIP "--offload-arch=${_target_id}" ${_result_var})
      if(${_result_var})
        list(APPEND _supported_target_ids "${_target_id}")
      endif()
    endforeach()
  endif()
  set(${VARIABLE} "${_supported_target_ids}" PARENT_SCOPE)
endfunction()
