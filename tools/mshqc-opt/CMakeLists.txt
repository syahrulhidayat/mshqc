get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(mshqc-opt mshqc-opt.cc)
llvm_update_compile_flags(mshqc-opt)

target_link_libraries(mshqc-opt PRIVATE
    mshqc_compiler
    ${dialect_libs}
    MLIROptLib
    MLIRPass
)
