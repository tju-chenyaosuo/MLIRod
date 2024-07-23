; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -codegen-opt-level=1 -S -mtriple=aarch64-- -atomic-expand %s | FileCheck %s
; RUN: opt -codegen-opt-level=1 -S -mtriple=aarch64-- -mattr=+outline-atomics -atomic-expand %s | FileCheck %s --check-prefix=OUTLINE-ATOMICS

define void @atomic_swap_f16(ptr %ptr, half %val) nounwind {
; CHECK-LABEL: @atomic_swap_f16(
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast half [[VAL:%.*]] to i16
; CHECK-NEXT:    br label [[ATOMICRMW_START:%.*]]
; CHECK:       atomicrmw.start:
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i16) [[PTR:%.*]])
; CHECK-NEXT:    [[TMP4:%.*]] = trunc i64 [[TMP3]] to i16
; CHECK-NEXT:    [[TMP5:%.*]] = zext i16 [[TMP2]] to i64
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @llvm.aarch64.stxr.p0(i64 [[TMP5]], ptr elementtype(i16) [[PTR]])
; CHECK-NEXT:    [[TRYAGAIN:%.*]] = icmp ne i32 [[TMP6]], 0
; CHECK-NEXT:    br i1 [[TRYAGAIN]], label [[ATOMICRMW_START]], label [[ATOMICRMW_END:%.*]]
; CHECK:       atomicrmw.end:
; CHECK-NEXT:    [[TMP7:%.*]] = bitcast i16 [[TMP4]] to half
; CHECK-NEXT:    ret void
;
; OUTLINE-ATOMICS-LABEL: @atomic_swap_f16(
; OUTLINE-ATOMICS-NEXT:    [[TMP2:%.*]] = bitcast half [[VAL:%.*]] to i16
; OUTLINE-ATOMICS-NEXT:    [[TMP3:%.*]] = atomicrmw xchg ptr [[PTR:%.*]], i16 [[TMP2]] acquire, align 2
; OUTLINE-ATOMICS-NEXT:    [[TMP4:%.*]] = bitcast i16 [[TMP3]] to half
; OUTLINE-ATOMICS-NEXT:    ret void
;
  %t1 = atomicrmw xchg ptr %ptr, half %val acquire
  ret void
}

define void @atomic_swap_f32(ptr %ptr, float %val) nounwind {
; CHECK-LABEL: @atomic_swap_f32(
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast float [[VAL:%.*]] to i32
; CHECK-NEXT:    br label [[ATOMICRMW_START:%.*]]
; CHECK:       atomicrmw.start:
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i32) [[PTR:%.*]])
; CHECK-NEXT:    [[TMP4:%.*]] = trunc i64 [[TMP3]] to i32
; CHECK-NEXT:    [[TMP5:%.*]] = zext i32 [[TMP2]] to i64
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @llvm.aarch64.stxr.p0(i64 [[TMP5]], ptr elementtype(i32) [[PTR]])
; CHECK-NEXT:    [[TRYAGAIN:%.*]] = icmp ne i32 [[TMP6]], 0
; CHECK-NEXT:    br i1 [[TRYAGAIN]], label [[ATOMICRMW_START]], label [[ATOMICRMW_END:%.*]]
; CHECK:       atomicrmw.end:
; CHECK-NEXT:    [[TMP7:%.*]] = bitcast i32 [[TMP4]] to float
; CHECK-NEXT:    ret void
;
; OUTLINE-ATOMICS-LABEL: @atomic_swap_f32(
; OUTLINE-ATOMICS-NEXT:    [[TMP2:%.*]] = bitcast float [[VAL:%.*]] to i32
; OUTLINE-ATOMICS-NEXT:    [[TMP3:%.*]] = atomicrmw xchg ptr [[PTR:%.*]], i32 [[TMP2]] acquire, align 4
; OUTLINE-ATOMICS-NEXT:    [[TMP4:%.*]] = bitcast i32 [[TMP3]] to float
; OUTLINE-ATOMICS-NEXT:    ret void
;
  %t1 = atomicrmw xchg ptr %ptr, float %val acquire
  ret void
}

define void @atomic_swap_f64(ptr %ptr, double %val) nounwind {
; CHECK-LABEL: @atomic_swap_f64(
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast double [[VAL:%.*]] to i64
; CHECK-NEXT:    br label [[ATOMICRMW_START:%.*]]
; CHECK:       atomicrmw.start:
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i64) [[PTR:%.*]])
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @llvm.aarch64.stxr.p0(i64 [[TMP2]], ptr elementtype(i64) [[PTR]])
; CHECK-NEXT:    [[TRYAGAIN:%.*]] = icmp ne i32 [[TMP4]], 0
; CHECK-NEXT:    br i1 [[TRYAGAIN]], label [[ATOMICRMW_START]], label [[ATOMICRMW_END:%.*]]
; CHECK:       atomicrmw.end:
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i64 [[TMP3]] to double
; CHECK-NEXT:    ret void
;
; OUTLINE-ATOMICS-LABEL: @atomic_swap_f64(
; OUTLINE-ATOMICS-NEXT:    [[TMP2:%.*]] = bitcast double [[VAL:%.*]] to i64
; OUTLINE-ATOMICS-NEXT:    [[TMP3:%.*]] = atomicrmw xchg ptr [[PTR:%.*]], i64 [[TMP2]] acquire, align 8
; OUTLINE-ATOMICS-NEXT:    [[TMP4:%.*]] = bitcast i64 [[TMP3]] to double
; OUTLINE-ATOMICS-NEXT:    ret void
;
  %t1 = atomicrmw xchg ptr %ptr, double %val acquire
  ret void
}