// ==============================================================================
// MSHQC - MLIR JIT Execution Manager
// Handles compilation, caching, and execution of MLIR modules
// ==============================================================================

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Module.h"

namespace mshqc {
namespace jit {

class ExecutionManager {
public:
    ExecutionManager() = default;
    ~ExecutionManager() = default;

    // Kompilasi dan simpan JIT Engine
    void compileAndCache(const std::string& kernel_name, mlir::OwningOpRef<mlir::ModuleOp>& module) {
        if (engine_cache_.find(kernel_name) != engine_cache_.end()) {
            return; // Kernel sudah dikompilasi
        }
        
        mlir::ExecutionEngineOptions engineOptions;
        auto maybeEngine = mlir::ExecutionEngine::create(module.get(), engineOptions);
        
        if (!maybeEngine) {
            throw std::runtime_error("JIT Compilation Failed for: " + kernel_name);
        }
        
        engine_cache_[kernel_name] = std::move(maybeEngine.get());
        std::cout << "[JIT] Compiled and Cached Kernel: " << kernel_name << "\n";
    }

    // Eksekusi fungsi dari engine yang ter-cache
    template <typename... Args>
    void execute(const std::string& kernel_name, const std::string& func_name, Args&... args) {
        auto it = engine_cache_.find(kernel_name);
        if (it == engine_cache_.end()) {
            throw std::runtime_error("JIT Engine not found for kernel: " + kernel_name);
        }
        
        auto error = it->second->invokePacked(func_name, args...);
        if (error) {
            throw std::runtime_error("JIT Execution Failed for function: " + func_name);
        }
    }

private:
    std::unordered_map<std::string, std::unique_ptr<mlir::ExecutionEngine>> engine_cache_;
};

} // namespace jit
} // namespace mshqc
