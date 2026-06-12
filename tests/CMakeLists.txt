# Tests needing model assets carry the "model" label and skip (exit 77) when
# PF_GGUF_DIR is unset. Fast gate: ctest -LE model.
function(pf_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE pf)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

pf_add_test(test_loader)
set_tests_properties(test_loader PROPERTIES LABELS "model")
