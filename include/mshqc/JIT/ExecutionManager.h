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

    static ExecutionManager& getInstance() {
        static ExecutionManager instance;
        return instance;
    }

    ExecutionManager(const ExecutionManager&) = delete;
    ExecutionManager& operator=(const ExecutionManager&) = delete;

    void compileAndCache(const std::string& kernel_name, mlir::OwningOpRef<mlir::ModuleOp>& module) {
        if (engine_cache_.find(kernel_name) != engine_cache_.end()) return;

        mlir::ExecutionEngineOptions engineOptions;
        auto maybeEngine = mlir::ExecutionEngine::create(module.get(), engineOptions);
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

private:
    ExecutionManager() = default;
    ~ExecutionManager() = default;
    std::unordered_map<std::string, std::unique_ptr<mlir::ExecutionEngine>> engine_cache_;
};

}
}
