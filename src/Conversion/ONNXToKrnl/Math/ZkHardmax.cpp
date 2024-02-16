// =============================================================================
//
// This file lowers ONNX softmax operator to Krnl dialect to be used in zk
// compilation.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
using namespace mlir;

namespace onnx_mlir {

struct ONNXZkHardmaxOpLowering : public OpConversionPattern<ONNXHardmaxOp> {
  ONNXZkHardmaxOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}
  LogicalResult matchAndRewrite(ONNXHardmaxOp hardmaxOp,
      ONNXHardmaxOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = hardmaxOp.getOperation();
    Location loc = ONNXLoc<ONNXHardmaxOp>(op);
    Value input = adaptor.getInput();

    MultiDialectBuilder<MathBuilder, KrnlBuilder, IndexExprBuilderForKrnl,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    Type elementType = memRefType.getElementType();
    Type i1Type = IntegerType::get(getContext(), 1);
    Value zero = create.math.constantIndex(0);
    Value smallOne = create.math.constant(i1Type, 1);
    Value zeroVal = create.math.constant(elementType, 0);
    Value one = create.math.constant(elementType, 1);
    IndexExpr zeroIE = LiteralIndexExpr(0);

    int64_t rank = memRefType.getRank();
    int64_t axis = llvm::dyn_cast<ONNXHardmaxOp>(op).getAxis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);
    Value resMemRef = create.mem.alloc(memRefType);
    create.krnl.memset(resMemRef, zeroVal);

    SmallVector<IndexExpr> lbs(rank - 1, zeroIE);
    SmallVector<IndexExpr, 4> ubs;
    DimIndexExpr axisDim;
    for (int64_t i = 0; i < rank; ++i) {
      if (i == axis) {
        axisDim = create.krnlIE.getShapeAsDim(input, i);
      } else {
        ubs.push_back(create.krnlIE.getShapeAsDim(input, i));
      }
    }

    ValueRange loopDef = create.krnl.defineLoops(rank - 1);
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange inputLoopInd) {
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder,
              SCFBuilder>
              create(createKrnl);
          // get the max value
          Value maxPtr = create.mem.alloc(MemRefType::get({}, elementType));
          // set to first element
          SmallVector<IndexExpr, 4> firstElementAccessFct;
          unsigned loopIndices = 0;
          for (int64_t i = 0; i < rank; ++i) {
            if (i == axis) {
              firstElementAccessFct.push_back(zeroIE);
            } else {
              firstElementAccessFct.push_back(
                  DimIndexExpr(inputLoopInd[loopIndices++]));
            }
          }
          Value firstElement = create.krnl.loadIE(input, firstElementAccessFct);
          create.krnl.store(zeroVal, maxPtr);
          ValueRange maxLoopDef = create.krnl.defineLoops(1);
          SmallVector<IndexExpr, 1> maxLoopLbs(1, LiteralIndexExpr(0));
          SmallVector<IndexExpr, 1> maxLoopUbs(1, axisDim);
          create.krnl.iterateIE(maxLoopDef, maxLoopDef, maxLoopLbs, maxLoopUbs,
              [&](KrnlBuilder &createKrnl, ValueRange maxLoopInd) {
                MultiDialectBuilder<MathBuilder, KrnlBuilder, MathBuilder,
                    SCFBuilder>
                    create(createKrnl);
                // get the max value
                SmallVector<IndexExpr, 4> nextElementAccessFct;
                unsigned loopIndices = 0;
                for (int64_t i = 0; i < rank; ++i) {
                  if (i == axis) {
                    nextElementAccessFct.push_back(DimIndexExpr(maxLoopInd[0]));
                  } else {
                    nextElementAccessFct.push_back(
                        DimIndexExpr(inputLoopInd[loopIndices++]));
                  }
                }
                Value nextElement =
                    create.krnl.loadIE(input, nextElementAccessFct);
                Value currentMax = create.krnl.load(maxPtr);
                Value isGreater = create.math.gt(nextElement, currentMax);
                Value nextMax =
                    create.math.select(isGreater, nextElement, currentMax);
                create.krnl.store(nextMax, maxPtr);
              });
          Value finalMax = create.krnl.load(maxPtr);
          Value notDonePtr = create.mem.alloc(MemRefType::get({}, i1Type));
          create.krnl.store(smallOne, notDonePtr);
          // set the zeros/ones
          ValueRange somethingTestLoopDef = create.krnl.defineLoops(1);
          SmallVector<IndexExpr, 1> somethingTestLoopLbs(
              1, LiteralIndexExpr(0));
          SmallVector<IndexExpr, 1> somethingTestLoopUbs(1, axisDim);
          create.krnl.iterateIE(somethingTestLoopDef, somethingTestLoopDef,
              somethingTestLoopLbs, somethingTestLoopUbs,
              [&](KrnlBuilder &createKrnl, ValueRange setLoopInd) {
                MultiDialectBuilder<MathBuilder, KrnlBuilder, MathBuilder,
                    SCFBuilder>
                    create(createKrnl);
                // get the max value
                SmallVector<IndexExpr, 4> nextElementAccessFct;
                unsigned loopIndices = 0;
                for (int64_t i = 0; i < rank; ++i) {
                  if (i == axis) {
                    nextElementAccessFct.push_back(DimIndexExpr(setLoopInd[0]));
                  } else {
                    nextElementAccessFct.push_back(
                        DimIndexExpr(inputLoopInd[loopIndices++]));
                  }
                }
                Value nextElement =
                    create.krnl.loadIE(input, nextElementAccessFct);
                Value isMax = create.math.eq(nextElement, finalMax);
                Value notDone = create.krnl.load(notDonePtr);
                Value valueToStore = create.math.mul(isMax, notDone);
                create.krnl.storeIE(create.math.cast(elementType, valueToStore),
                    resMemRef, nextElementAccessFct);
                Value notDoneUpdate =
                    create.math.mul(create.math.sub(smallOne, isMax), notDone);
                create.krnl.store(notDoneUpdate, notDonePtr);
              });
        });

    rewriter.replaceOp(op, resMemRef);
    return success();
  }
};

void populateLoweringONNXZkHardmaxOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXZkHardmaxOpLowering>(typeConverter, ctx);
}
} // namespace onnx_mlir
