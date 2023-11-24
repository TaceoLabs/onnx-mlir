/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gather.cpp - Lowering Gather Op ---------------------===//
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Gather Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/zkml/IR/Gather.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
using ZKMLGatherOp = zk_ml_toolchain::zkml::GatherOp;

namespace onnx_mlir {

struct ONNXGatherOpLowering : public OpConversionPattern<ONNXGatherOp> {
  ONNXGatherOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool zkMl)
      : OpConversionPattern(typeConverter, ctx), zkMl(zkMl) {}

private:
  bool zkMl;

  LogicalResult matchAndRewrite(ONNXGatherOp gatherOp,
      ONNXGatherOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = gatherOp.getOperation();
    Location loc = ONNXLoc<ONNXGatherOp>(op);
    ValueRange operands = adaptor.getOperands();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXGatherOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();

    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Operands and attributes.
    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    int64_t axisLit = adaptor.getAxis();
    int64_t dataRank = data.getType().cast<MemRefType>().getRank();
    int64_t indicesRank = indices.getType().cast<MemRefType>().getRank();

    if (this->zkMl) {
      OpBuilder builder(op);
      auto NewGatherOp = builder.create<ZKMLGatherOp>(
          op->getLoc(), outputMemRefType, data, indices, gatherOp.getAxis());
      rewriter.replaceOp(op, NewGatherOp->getResult(0));
    } else {
      // Insert an allocation and deallocation for the output of this operation.
      Value alloc = insertAllocAndDeallocSimple(
          rewriter, op, outputMemRefType, loc, shapeHelper.getOutputDims());

      int64_t axisLit = gatherOp.getAxis();
      int64_t dataRank = data.getType().cast<MemRefType>().getRank();
      int64_t indicesRank = indices.getType().cast<MemRefType>().getRank();

      // Determine whether indices may be negative.
      bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

      // Negative value means counting dimensions from the back.
      axisLit = axisLit < 0 ? axisLit + dataRank : axisLit;

      int64_t outputRank = shapeHelper.getOutputDims().size();
      int iIndexStart = 0;
      int jIndexStart = iIndexStart + axisLit;
      int kIndexStart = jIndexStart + indicesRank - (axisLit + 1);

      LiteralIndexExpr zeroIE(0);
      DimsExpr dataDims;
      create.krnlIE.getShapeAsDims(data, dataDims);

      /*
        The pattern that we are using is that of numpy.take.

        Ni, Nk = data.shape[:axis], data.shape[axis+1:]
        Nj = indices.shape
        for ii in ndindex(Ni):
          for jj in ndindex(Nj):
            for kk in ndindex(Nk):
              out[ii + jj + kk] = data[ii + (indices[jj],) + kk]

      // Define loops and iteration trip counts (equivalent to size of output)
      // */
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      DimsExpr lbs(outputRank, zeroIE);
      create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(),
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Insert code inside the loop.
            IndexExprScope innerLoopScope(createKrnl);
            SymbolIndexExpr axisDim(dataDims[axisLit]);

            // Save data into output
            createKrnl.storeIE(dataVal, alloc, outputAccessFct);
          });
      rewriter.replaceOp(op, alloc);
      onnxToKrnlSimdReport(op);
    }
    return success();
  };

  void populateLoweringONNXGatherOpPattern(RewritePatternSet &patterns,
      TypeConverter &typeConverter, MLIRContext *ctx, bool zkMl) {
    patterns.insert<ONNXGatherOpLowering>(typeConverter, ctx, zkMl);
  }

} // namespace onnx_mlir
