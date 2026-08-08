# CMakeLists untuk tools MLIR
add_subdirectory(mshqc-opt)

# mshqc JIT Profiler Target
add_executable(mshqc-jit-bench jit-bench/jit-bench.cc)
target_link_libraries(mshqc-jit-bench 
    PRIVATE 
    mshqc_compiler 
    MLIRIR 
    MLIRParser 
    MLIRLLVMDialect 
    MLIRExecutionEngine
)
