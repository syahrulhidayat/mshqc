#ifndef MSHQC_COMPILER_FRONTEND_GRAPHBUILDER_H_
#define MSHQC_COMPILER_FRONTEND_GRAPHBUILDER_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <memory>
#include <vector>
#include <string>

namespace mshqc {
namespace compiler {

class GraphBuilder {
public:
    GraphBuilder();
    ~GraphBuilder();

    // Inisialisasi modul (Fungsi utama graf komputasi)
    void initializeModule(const std::string& functionName);

    // Membangun operasi mshqc.contract dari dimensi spesifik
    mlir::Value emitContractOp(const std::vector<int64_t>& lhsShape,
                               const std::vector<int64_t>& rhsShape,
                               const std::string& einsum_eq);

    // Verifikasi validitas graf sebelum optimasi
    bool verifyGraph();

    // Membuka akses untuk Pass Manager
    mlir::ModuleOp getModule() const { return module.get(); }
    mlir::MLIRContext* getContext() { return &context; }

private:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace compiler
} // namespace mshqc

#endif // MSHQC_COMPILER_FRONTEND_GRAPHBUILDER_H_
