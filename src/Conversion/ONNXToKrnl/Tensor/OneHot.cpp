/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- OneHot.cpp - Lowering OneHot Op -------------------===//
//
// Copyright 2021-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX OneHot Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXOneHotOpLowering : public OpConversionPattern<ONNXOneHotOp> {
  ONNXOneHotOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool zkMl)
      : OpConversionPattern(typeConverter, ctx), zkMl(zkMl) {}

private:
  bool zkMl;

  LogicalResult matchAndRewrite(ONNXOneHotOp onehotOp,
      ONNXOneHotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = onehotOp.getOperation();
    Location loc = ONNXLoc<ONNXOneHotOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value indices = adaptor.getIndices();
    Value values = adaptor.getValues();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXOneHotOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    int64_t axis = shapeHelper.axis;

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();
    assert(indices.getType().isa<MemRefType>() &&
           "Indices must be memref for OneHot");
    MemRefType indicesMemRef = indices.getType().cast<MemRefType>();
    // Insert an allocation and deallocation for the output of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Load off/on vals found in values memref.
    LiteralIndexExpr minusOneIE(-1), zeroIE(0), oneIE(1);
    Value offVal = create.krnl.loadIE(values, zeroIE);
    Value onVal = create.krnl.loadIE(values, oneIE);

    mlir::Type indicesType = indicesMemRef.getElementType().isa<FloatType>()
                                 ? IntegerType::get(getContext(), 64)
                                 : indicesMemRef.getElementType();

    Value minusOneInt = create.math.constant(indicesType, -1);
    Value zeroInt = create.math.constant(indicesType, 0);
    Value depthInt;
    if (zkMl) {
      assert(shapeHelper.depth.isLiteral() &&
             "Depth must be literal index expr for zkML OneHot");
      depthInt =
          create.math.constant(indicesType, shapeHelper.depth.getLiteral());
    }
    // just add the value of getDepth(). if it is rank1 tensor add a load
    //  Value depthInt = create.math.constant(indicesType, onehotOp.getDepth());

    // Iterate over all of the inputs.
    int64_t indicesRank = create.krnlIE.getShapedTypeRank(indices);
    SmallVector<IndexExpr, 4> indicesLbs(indicesRank, zeroIE);
    SmallVector<IndexExpr, 4> indicesUbs;
    create.krnlIE.getShapeAsDims(indices, indicesUbs);
    ValueRange indicesLoopDef = create.krnl.defineLoops(indicesRank);
    create.krnl.iterateIE(indicesLoopDef, indicesLoopDef, indicesLbs,
        indicesUbs, [&](KrnlBuilder createKrnl, ValueRange indicesLoopInd) {
          // Loop for all input values.
          // Input val is allowed to be any integer/float. Read and convert to
          // index type.
          Value inputVal = createKrnl.load(indices, indicesLoopInd);
          SymbolIndexExpr depth(shapeHelper.depth);
          if (zkMl) {
            MultiDialectBuilder<MathBuilder, ZkMlBuilder> create(rewriter, loc);
            // wrap around input val
            //  Value inputNegVal = create.add(inputVal, depthInt);
            if (inputVal.getType().isa<FloatType>()) {
              inputVal = create.math.cast(indicesType, inputVal);
            }
            ValueRange depthLoopDef = createKrnl.defineLoops(1);
            Value inputNeg = create.math.add(inputVal, depthInt);
            Value isNeg = create.math.lt(inputVal, zeroInt);
            Value inputIndex = create.math.select(isNeg, inputNeg, inputVal);
            Value isTooSmall = create.math.lt(inputIndex, zeroInt);
            Value isTooBig = create.math.ge(inputIndex, depthInt);
            Value outOfBounds = create.math.ori(isTooSmall, isTooBig);
            Value finalValue =
                create.math.select(outOfBounds, minusOneInt, inputIndex);
            createKrnl.iterateIE(depthLoopDef, depthLoopDef, {zeroIE}, {depth},
                [&](KrnlBuilder createBuilder, ValueRange depthLoopInd) {
                  MultiDialectBuilder<MathBuilder, ZkMlBuilder> create(
                      rewriter, loc);
                  Value res =
                      create.zkml.CmpSet(outputMemRefType.getElementType(),
                          finalValue, onVal, offVal, depthLoopInd[0]);
                  SmallVector<Value, 4> outputAccessFct;
                  int64_t dec = 0;
                  for (int64_t i = 0; i < indicesRank + 1; ++i) {
                    if (i == axis) {
                      outputAccessFct.emplace_back(depthLoopInd[0]);
                      dec = 1;
                    } else {
                      outputAccessFct.emplace_back(indicesLoopInd[i - dec]);
                    }
                  }
                  createKrnl.store(res, alloc, outputAccessFct);
                });
          } else {
            MathBuilder createMath(createKrnl);
            Value inputIndexVal = createMath.castToIndex(inputVal);
            IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
            NonAffineIndexExpr input(inputIndexVal);
            // Because valid input is from [-depth...depth-1], we must add depth
            // to input values that are negative. This will define inputIndex.
            IndexExpr inputNegVal = input + depth;
            IndexExpr isNeg = input < zeroIE;
            IndexExpr inputIndex = IndexExpr::select(isNeg, inputNegVal, input);
            // Now compute in inputIndex is still out of bound, in which case
            // all values are off.
            IndexExpr isTooSmall = inputIndex < zeroIE;
            IndexExpr isTooBig = inputIndex >= depth;
            IndexExpr outOfBound = isTooSmall | isTooBig;
            // Define here the index that has the on Value. If out of bound, put
            // -1 here as this value will never occur.
            IndexExpr onValueIndex =
                IndexExpr::select(outOfBound, minusOneIE, inputIndex);
            Value onValueIndexVal = onValueIndex.getValue();
            // Now we have the index that is on, iterate over the depth values
            // along axis, and set the right one to the value on.
            ValueRange depthLoopDef = createKrnl.defineLoops(1);
            createKrnl.iterateIE(depthLoopDef, depthLoopDef, {zeroIE}, {depth},
                [&](KrnlBuilder createBuilder, ValueRange depthLoopInd) {
                  MathBuilder createMath(createKrnl);
                  Value onCond =
                      createMath.eq(depthLoopInd[0], onValueIndexVal);
                  Value res = createMath.select(onCond, onVal, offVal);
                  // Output access function is input indices with depth index
                  // spliced in the axis location.
                  SmallVector<Value, 4> outputAccessFct;
                  int64_t dec = 0;
                  for (int64_t i = 0; i < indicesRank + 1; ++i) {
                    if (i == axis) {
                      outputAccessFct.emplace_back(depthLoopInd[0]);
                      dec = 1;
                    } else {
                      outputAccessFct.emplace_back(indicesLoopInd[i - dec]);
                    }
                  }
                  createKrnl.store(res, alloc, outputAccessFct);
                });
          }
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXOneHotOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool zkMl) {
  patterns.insert<ONNXOneHotOpLowering>(typeConverter, ctx, zkMl);
}

} // namespace onnx_mlir
