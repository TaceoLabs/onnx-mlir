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

#include "mlir/Dialect/zkml/IR/Gather.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

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

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder, ZkMlBuilder>
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
    MemRefType dataMemRef = data.getType().cast<MemRefType>();
    int64_t dataRank = dataMemRef.getRank();
    MemRefType indicesMemRef = indices.getType().cast<MemRefType>();
    int64_t indicesRank = indicesMemRef.getRank();

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
    */
    // Define loops and iteration trip counts (equivalent to size of output)
    ValueRange loopDef = create.krnl.defineLoops(outputRank);
    DimsExpr lbs(outputRank, zeroIE);
    if (zkMl) {
      Type outputElementType = outputMemRefType.getElementType();
      Type indicesElementType = indicesMemRef.getElementType();
      Value zeroVal = create.math.constant(outputElementType, 0);
      Value zeroInt = create.math.constant(indicesMemRef.getElementType(), 0);
      Value wrapConstant = create.math.constant(
          indicesElementType, dataMemRef.getShape()[axisLit]);
      create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(),
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerLoopScope(createKrnl);
            SymbolIndexExpr axisDim(dataDims[axisLit]);

            // compute the loop indices for the output
            SmallVector<IndexExpr, 4> outputAccessFct;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessFct);

            // Compute access function for indices[jjs].
            SmallVector<IndexExpr, 4> indicesAccessFct;
            for (int j = 0; j < indicesRank; ++j)
              indicesAccessFct.emplace_back(outputAccessFct[jIndexStart + j]);
            Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
            if (indicesMayBeNegative) {
              // %3 = arith.cmpi slt, %2, %c0 : index
              // %4 = arith.addi %2, %c3 : index
              // %5 = arith.select %3, %4, %2 : index
              Value isNegative = create.math.slt(indexVal, zeroInt);
              Value wrapedIndex = create.math.add(indexVal, wrapConstant);
              indexVal = create.math.select(isNegative, wrapedIndex, indexVal);
            }
            Value accPtr =
                create.mem.alloc(MemRefType::get({}, outputElementType));
            createKrnl.store(zeroVal, accPtr);

            ValueRange innerLoopDef = create.krnl.defineLoops(1);
            SmallVector<IndexExpr, 1> innerLbs(1, zeroIE);
            SmallVector<IndexExpr, 1> innerUbs(1, axisDim);
            // add another loop for zk
            createKrnl.iterateIE(innerLoopDef, innerLoopDef, innerLbs, innerUbs,
                [&](KrnlBuilder &createKrnl, ValueRange innerIndex) {
                  Value acc = createKrnl.load(accPtr);
                  // Compute access function of data: data[ii + (indices[jj],) +
                  // kk]
                  SmallVector<IndexExpr, 4> dataAccessFct;
                  // First add indices iis
                  for (int i = 0; i < axisLit; ++i)
                    dataAccessFct.emplace_back(
                        outputAccessFct[iIndexStart + i]);
                  // Then add indices[jj] (indexVal).
                  // getIndexExprList<DimIndexExpr>(innerIndex, dataAccessFct);
                  dataAccessFct.emplace_back(DimIndexExpr(innerIndex[0]));
                  // Then add kks.
                  for (int k = axisLit + 1; k < dataRank; ++k)
                    dataAccessFct.emplace_back(
                        outputAccessFct[kIndexStart + k]);
                  llvm::outs() << "[";
                  for (auto t : dataAccessFct) {
                  llvm::outs() << "hello :)\n";
                  }
                  llvm::outs() << "]\n";
                  Value dataVal = createKrnl.loadIE(data, dataAccessFct);
                  Value newAcc = create.zkml.Gather(
                      outputElementType, acc, dataVal, indexVal, innerIndex[0]);
                  createKrnl.store(newAcc, accPtr);
                });
            Value acc = createKrnl.load(accPtr);
            createKrnl.storeIE(acc, alloc, outputAccessFct);
          });
    } else {
      create.krnl.iterateIE(loopDef, loopDef, lbs, shapeHelper.getOutputDims(),
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Insert code inside the loop.
            IndexExprScope innerLoopScope(createKrnl);
            SymbolIndexExpr axisDim(dataDims[axisLit]);

            // compute the loop indices for the output
            SmallVector<IndexExpr, 4> outputAccessFct;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessFct);

            // Compute access function for indices[jjs].
            SmallVector<IndexExpr, 4> indicesAccessFct;
            for (int j = 0; j < indicesRank; ++j)
              indicesAccessFct.emplace_back(outputAccessFct[jIndexStart + j]);
            Value indexVal = createKrnl.loadIE(indices, indicesAccessFct);
            // Loaded value is an index that is not affine
            IndexExpr index = NonAffineIndexExpr(indexVal);
            // When index may be negative, add axis Dim to it.
            if (indicesMayBeNegative)
              index = index.selectOrSelf(index < zeroIE, index + axisDim);

            // Compute access function of data: data[ii + (indices[jj],) + kk]
            SmallVector<IndexExpr, 4> dataAccessFct;
            // First add indices iis
            for (int i = 0; i < axisLit; ++i)
              dataAccessFct.emplace_back(outputAccessFct[iIndexStart + i]);
            // Then add indices[jj] (indexVal).
            dataAccessFct.emplace_back(index);
            // Then add kks.
            for (int k = axisLit + 1; k < dataRank; ++k)
              dataAccessFct.emplace_back(outputAccessFct[kIndexStart + k]);
            Value dataVal = createKrnl.loadIE(data, dataAccessFct);

            // Save data into output
            createKrnl.storeIE(dataVal, alloc, outputAccessFct);
          });
    }
    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXGatherOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool zkMl) {
  patterns.insert<ONNXGatherOpLowering>(typeConverter, ctx, zkMl);
}

} // namespace onnx_mlir
