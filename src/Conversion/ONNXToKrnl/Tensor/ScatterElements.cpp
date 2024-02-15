/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- ScatterElements.cpp - Lowering ScatterElements Op ----------===//
//
// Copyright 2022-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ScatterElements Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXScatterElementsOpLowering
    : public OpConversionPattern<ONNXScatterElementsOp> {
  ONNXScatterElementsOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool zkMl)
      : OpConversionPattern(typeConverter, ctx), zkMl(zkMl) {}

private:
  bool zkMl;
  LogicalResult matchAndRewrite(ONNXScatterElementsOp scatterElementsOp,
      ONNXScatterElementsOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = scatterElementsOp.getOperation();
    Location loc = ONNXLoc<ONNXScatterElementsOp>(op);

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder, ZkMlBuilder>
        create(rewriter, loc);

    // Operands and attributes.
    Value data = adaptor.getData();
    Value updates = adaptor.getUpdates();
    Value indices = adaptor.getIndices();
    int64_t axis = adaptor.getAxis();
    MemRefType dataMemRefType = data.getType().cast<MemRefType>();
    int64_t dataRank = data.getType().cast<MemRefType>().getRank();
    int64_t updatesRank = updates.getType().cast<MemRefType>().getRank();
    MemRefType indicesMemRefType = indices.getType().cast<MemRefType>();
    int64_t indicesRank = indicesMemRefType.getRank();
    assert(updatesRank == dataRank && indicesRank == dataRank &&
           "All input tensors must have the same rank");

    // Determine whether indices may be negative.
    bool indicesMayBeNegative = !indicesAreNonNegativeConstants(indices);

    // Negative value means counting dimensions from the back.
    axis = axis < 0 ? axis + dataRank : axis;

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    int64_t outputRank = outputMemRefType.getShape().size();
    assert(outputRank == dataRank && "Output rank not equal to data rank");

    // Insert an allocation and deallocation for the result of this operation.
    IndexExprScope indexScope(create.krnl);
    DimsExpr dataDims;
    create.krnlIE.getShapeAsDims(data, dataDims);
    Value output = create.mem.alignedAlloc(outputMemRefType, dataDims);

    // Step1: copy the data array into the output array.
    Value numOfElements = getDynamicMemRefSize(rewriter, loc, data);
    create.krnl.memcpy(output, data, numOfElements);

    // Step2: scatter the updates array into the output array.
    //   index = indices[i][j]...[n]
    //   val = updates[i][j]...[n]
    //   output[i][j]..[index]..[n] = val (index used at position axis)
    //
    ValueRange loopDef = create.krnl.defineLoops(updatesRank);
    DimsExpr lbs(updatesRank, LiteralIndexExpr(0)), ubs;
    create.krnlIE.getShapeAsDims(updates, ubs);
    LiteralIndexExpr zeroIE(0);
    SymbolIndexExpr axisDim(dataDims[axis]);
    if (zkMl) {
      Type indicesElementType = indicesMemRefType.getElementType();
      Value zeroInt = create.math.constant(indicesElementType, 0);
      Value wrapConstant = create.math.constant(
          indicesElementType, dataMemRefType.getShape()[axis]);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Insert code inside the loop.
            IndexExprScope innerLoopScope(createKrnl);

            // Access function for updates and indices.
            SmallVector<IndexExpr, 4> accessFct;
            getIndexExprList<DimIndexExpr>(loopInd, accessFct);

            Value updateVal = createKrnl.loadIE(updates, accessFct);
            Value indexVal = createKrnl.loadIE(indices, accessFct);

            // When index may be negative, add axis dim to it.
            if (indicesMayBeNegative) {
              // %3 = arith.cmpi slt, %2, %c0 : index
              // %4 = arith.addi %2, %c3 : index
              // %5 = arith.select %3, %4, %2 : index
              Value isNegative = create.math.slt(indexVal, zeroInt);
              Value wrapedIndex = create.math.add(indexVal, wrapConstant);
              indexVal = create.math.select(isNegative, wrapedIndex, indexVal);
            }
            ValueRange innerLoopDef = create.krnl.defineLoops(1);
            SmallVector<IndexExpr, 1> innerLbs(1, zeroIE);
            SmallVector<IndexExpr, 1> innerUbs(1, axisDim);
            createKrnl.iterateIE(innerLoopDef, innerLoopDef, innerLbs, innerUbs,
                [&](KrnlBuilder &createKrnl, ValueRange innerIndex) {
                  // IndexExprScope innerMostLoopScope(createKrnl);
                  // Get the old value
                  SmallVector<IndexExpr, 4> outputAccessFct;
                  for (int i = 0; i < dataRank; ++i)
                    outputAccessFct.emplace_back(
                        (i == axis) ? DimIndexExpr(innerIndex[0])
                                    : accessFct[i]);
                  Value valueAtOutput =
                      createKrnl.loadIE(output, outputAccessFct);
                  Value promotedIndex =
                      create.math.cast(indicesElementType, innerIndex[0]);
                  Value cmp = create.math.eq(indexVal, promotedIndex);
                  Value storeVal =
                      create.math.select(cmp, updateVal, valueAtOutput);
                  createKrnl.storeIE(storeVal, output, outputAccessFct);
                });

            // // Access function for the output.
            // SmallVector<IndexExpr, 4> outputAccessFct;
            // for (int i = 0; i < dataRank; ++i)
            //   outputAccessFct.emplace_back((i == axis) ? index :
            //   accessFct[i]);
            //
            // // Scatter updateVal into the output tensor.
            // createKrnl.storeIE(updateVal, output, outputAccessFct);
          });
    } else {
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            // Insert code inside the loop.
            IndexExprScope innerLoopScope(createKrnl);

            // Access function for updates and indices.
            SmallVector<IndexExpr, 4> accessFct;
            getIndexExprList<DimIndexExpr>(loopInd, accessFct);

            Value updateVal = createKrnl.loadIE(updates, accessFct);
            Value indexVal = createKrnl.loadIE(indices, accessFct);
            IndexExpr index = NonAffineIndexExpr(indexVal);

            // When index may be negative, add axis dim to it.
            if (indicesMayBeNegative) {
              index = index.selectOrSelf(index < zeroIE, index + axisDim);
            }

            // Access function for the output.
            SmallVector<IndexExpr, 4> outputAccessFct;
            for (int i = 0; i < dataRank; ++i)
              outputAccessFct.emplace_back((i == axis) ? index : accessFct[i]);

            // Scatter updateVal into the output tensor.
            createKrnl.storeIE(updateVal, output, outputAccessFct);
          });
    }
    rewriter.replaceOp(op, output);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXScatterElementsOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool zkMl) {
  patterns.insert<ONNXScatterElementsOpLowering>(typeConverter, ctx, zkMl);
}

} // namespace onnx_mlir
