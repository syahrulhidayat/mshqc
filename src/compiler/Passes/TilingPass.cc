// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
// MSHQC - MLIR Dynamic Cache-Aware Tiling Pass
// ==============================================================================

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mshqc/compiler/Passes/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#if defined(__linux__) || defined(__APPLE__)
    #include <unistd.h>
#endif

namespace mshqc {
namespace compiler {

#define GEN_PASS_DEF_LINALGTILING
#include "Passes.h.inc"

namespace {

// Utilitas Deteksi Hardware Dinamis
void get_cache_sizes(int64_t& l1d_size_kb, int64_t& l2_size_kb) {
    l1d_size_kb = 32; // Fallback Default
    l2_size_kb = 256;
    
#if defined(__linux__)
    long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l1 > 0) l1d_size_kb = l1 / 1024;
    if (l2 > 0) l2_size_kb = l2 / 1024;
#endif
}

struct LinalgTilingPass : public impl::LinalgTilingBase<LinalgTilingPass> {
    using LinalgTilingBase::LinalgTilingBase;

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::scf::SCFDialect, mlir::linalg::LinalgDialect>();
    }

    void runOnOperation() override {
        mlir::func::FuncOp funcOp = getOperation();
        mlir::IRRewriter rewriter(&getContext());

        // Polling Hardware OS 
        int64_t l1d_kb, l2_kb;
        get_cache_sizes(l1d_kb, l2_kb);
        
        // Kalkulasi batas tile (Asumsi elemen FP64 = 8 bytes)
        // Heuristik: Alokasikan ~50% dari L2 cache untuk matriks intermediat
        int64_t l2_elements = (l2_kb * 1024 / 8) / 2; 
        int64_t l2_tile_dim = std::cbrt(l2_elements); // Untuk loop orde-3/4
        
        int64_t l1_elements = (l1d_kb * 1024 / 8) / 2;
        int64_t l1_tile_dim = std::sqrt(l1_elements); // Optimasi matrix block register

        llvm::SmallVector<mlir::linalg::LinalgOp, 4> targetOps;
        funcOp.walk([&](mlir::linalg::LinalgOp op) {
            if (!op->hasAttr("tiled") && mlir::isa<mlir::linalg::GenericOp>(op)) {
                targetOps.push_back(op);
            }
        });

        for (auto op : targetOps) {
            llvm::SmallVector<int64_t> l2Tiles;
            for (auto iterType : op.getIteratorTypesArray()) {
                if (iterType == mlir::utils::IteratorType::parallel) {
                    l2Tiles.push_back(l2_tile_dim); // Dinamis berdasarkan L2 Host
                } else if (iterType == mlir::utils::IteratorType::reduction) {
                    l2Tiles.push_back(l2_tile_dim / 2); 
                } else {
                    l2Tiles.push_back(0);
                }
            }
            
            mlir::linalg::LinalgTilingOptions l2Options;
            l2Options.setTileSizes(l2Tiles);
            l2Options.setLoopType(mlir::linalg::LinalgTilingLoopType::ParallelLoops);
            
            rewriter.setInsertionPoint(op);
            mlir::FailureOr<mlir::linalg::TiledLinalgOp> l2Result = 
                 mlir::linalg::tileLinalgOp(rewriter, op, l2Options);

            if (mlir::succeeded(l2Result)) {
                llvm::SmallVector<int64_t> l1Tiles;
                for (auto iterType : l2Result->op.getIteratorTypesArray()) {
                    if (iterType == mlir::utils::IteratorType::parallel) {
                        l1Tiles.push_back(8); // Vektorisasi presisi register AVX-512 (8 elemen double)
                    } else if (iterType == mlir::utils::IteratorType::reduction) {
                        l1Tiles.push_back(l1_tile_dim); // Tiling L1 Dinamis
                    } else {
                        l1Tiles.push_back(0);
                    }
                }
                
                mlir::linalg::LinalgTilingOptions l1Options;
                l1Options.setTileSizes(l1Tiles);
                l1Options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);
                
                rewriter.setInsertionPoint(l2Result->op);
                mlir::FailureOr<mlir::linalg::TiledLinalgOp> l1Result =
                    mlir::linalg::tileLinalgOp(rewriter, l2Result->op, l1Options);

                if (mlir::succeeded(l1Result)) {
                    l1Result->op->setAttr("tiled", rewriter.getUnitAttr());
                    rewriter.replaceOp(op, l1Result->tensorResults);
                    rewriter.eraseOp(l2Result->op);
                }
            }
        }
    }
};

} // end namespace

std::unique_ptr<mlir::Pass> createLinalgTilingPass() {
    return std::make_unique<LinalgTilingPass>();
}

} // namespace compiler
} // namespace mshqc
