# CMakeLists untuk tools MLIR
add_subdirectory(mshqc-opt)

# mshqc JIT Profiler Target
add_executable(mshqc-jit-bench jit-bench/jit-bench.cc)

set_target_properties(mshqc-jit-bench PROPERTIES ENABLE_EXPORTS TRUE)

target_link_libraries(mshqc-jit-bench
     PRIVATE
     mshqc_compiler
     MLIRIR
     MLIRParser
     MLIRLLVMDialect
     MLIROpenMPDialect
     MLIRSCFDialect
     MLIRExecutionEngine
)
