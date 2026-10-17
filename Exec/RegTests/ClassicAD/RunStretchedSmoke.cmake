if(NOT DEFINED ERF_EXECUTABLE OR NOT EXISTS "${ERF_EXECUTABLE}")
  message(FATAL_ERROR "ERF_EXECUTABLE must name a built erf_exec")
endif()
foreach(_path_var INPUT_FILE LOCATION_FILE VALIDATOR)
  if(NOT DEFINED ${_path_var} OR NOT EXISTS "${${_path_var}}")
    message(FATAL_ERROR "${_path_var} must name an existing file")
  endif()
endforeach()
if(NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "TEST_ROOT is required")
endif()

find_program(_python NAMES python3 python REQUIRED)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef _run_id)
set(_run_dir "${TEST_ROOT}/classic_ad_stretched_smoke_${_run_id}")
file(MAKE_DIRECTORY "${_run_dir}")

set(_launch)
if(DEFINED MPIEXEC_EXECUTABLE AND NOT "${MPIEXEC_EXECUTABLE}" STREQUAL "")
  list(APPEND _launch "${MPIEXEC_EXECUTABLE}")
  if(DEFINED MPIEXEC_NUMPROC_FLAG AND NOT "${MPIEXEC_NUMPROC_FLAG}" STREQUAL "")
    list(APPEND _launch "${MPIEXEC_NUMPROC_FLAG}" 2)
  endif()
  if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
    list(APPEND _launch ${MPIEXEC_PREFLAGS})
  endif()
endif()

set(_base_command ${_launch}
  "${ERF_EXECUTABLE}"
  "${INPUT_FILE}"
  "erf.windfarm_loc_table=${LOCATION_FILE}")
execute_process(
  COMMAND ${_base_command}
  WORKING_DIRECTORY "${_run_dir}"
  RESULT_VARIABLE _initial_result
  OUTPUT_VARIABLE _initial_stdout
  ERROR_VARIABLE _initial_stderr)
if(NOT "${_initial_result}" STREQUAL "0")
  message(FATAL_ERROR
    "stretched ClassicAD smoke run failed (exit ${_initial_result})\nstdout:\n${_initial_stdout}\nstderr:\n${_initial_stderr}")
endif()
execute_process(
  COMMAND "${_python}" "${VALIDATOR}" "${_run_dir}"
  RESULT_VARIABLE _validation_result
  OUTPUT_VARIABLE _validation_stdout
  ERROR_VARIABLE _validation_stderr)
if(NOT "${_validation_result}" STREQUAL "0")
  message(FATAL_ERROR
    "stretched ClassicAD validation failed\n${_validation_stdout}\n${_validation_stderr}")
endif()

execute_process(
  COMMAND ${_base_command}
    "amr.restart=chk00002" "max_step=3" "erf.check_int=-1"
  WORKING_DIRECTORY "${_run_dir}"
  RESULT_VARIABLE _restart_result
  OUTPUT_VARIABLE _restart_stdout
  ERROR_VARIABLE _restart_stderr)
if(NOT "${_restart_result}" STREQUAL "0")
  message(FATAL_ERROR
    "stretched ClassicAD restart failed (exit ${_restart_result})\nstdout:\n${_restart_stdout}\nstderr:\n${_restart_stderr}")
endif()
execute_process(
  COMMAND "${_python}" "${VALIDATOR}" "${_run_dir}"
  RESULT_VARIABLE _restart_validation_result
  OUTPUT_VARIABLE _restart_validation_stdout
  ERROR_VARIABLE _restart_validation_stderr)
if(NOT "${_restart_validation_result}" STREQUAL "0")
  message(FATAL_ERROR
    "stretched ClassicAD restart validation failed\n${_restart_validation_stdout}\n${_restart_validation_stderr}")
endif()
message(STATUS "${_restart_validation_stdout}")
