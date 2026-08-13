// ==============================================================================
// MSHQC - MLIR JIT Execution Manager (Centralized Singleton Registry)
// ==============================================================================

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mshqc {
namespace jit {

class ExecutionManager {
public:
    // Meyer's Singleton: C++11 Thread-Safe Initialization
    static ExecutionManager& getInstance() {
        static ExecutionManager instance;
        return instance;
    }

    // Mencegah Copy & Assignment
    ExecutionManager(const ExecutionManager&) = delete;
    ExecutionManager& operator=(const ExecutionManager&) = delete;

    void compileAndCache(const std::string& kernel_name, mlir::ModuleOp module) {
        if (engine_cache_.find(kernel_name) != engine_cache_.end()) return;
        
        mlir::ExecutionEngineOptions engineOptions;
        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        if (!maybeEngine) throw std::runtime_error("JIT Compilation Failed for: " + kernel_name);
        
        engine_cache_[kernel_name] = std::move(maybeEngine.get());
    }

    template <typename... Args>
    void execute(const std::string& kernel_name, const std::string& func_name, Args&... args) {
        auto it = engine_cache_.find(kernel_name);
        if (it == engine_cache_.end()) throw std::runtime_error("JIT Engine not found: " + kernel_name);
        
        if (auto error = it->second->invokePacked(func_name, args...)) {
            throw std::runtime_error("JIT Execution Failed for function: " + func_name);
        }
    }

public:
    ExecutionManager() = default;
    ~ExecutionManager() = default;
private:
    std::unordered_map<std::string, std::unique_ptr<mlir::ExecutionEngine>> engine_cache_;
};

} // namespace jit
} // namespace mshqc
