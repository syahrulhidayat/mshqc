// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================


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

    void initializeModule(const std::string& functionName);

    mlir::Value emitContractOp(const std::vector<int64_t>& lhsShape,
                               const std::vector<int64_t>& rhsShape,
                               const std::vector<int64_t>& resShape,
                               const std::string& einsum_eq);

    bool verifyGraph();

    mlir::ModuleOp getModule() const { return module.get(); }
    mlir::MLIRContext* getContext() { return &context; }

private:
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::OwningOpRef<mlir::ModuleOp> module;
};

}
}

#endif
