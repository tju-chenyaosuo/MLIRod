; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+m,+v,+zfh,+zvfh < %s | FileCheck %s --check-prefixes=CHECK,ZVFH
; RUN: llc -mtriple=riscv64 -mattr=+m,+v,+zfh,+zvfh < %s | FileCheck %s --check-prefixes=CHECK,ZVFH
; RUN: llc -mtriple=riscv32 -mattr=+m,+v,+zfh,+zvfhmin < %s | FileCheck %s --check-prefixes=CHECK,ZVFHMIN
; RUN: llc -mtriple=riscv64 -mattr=+m,+v,+zfh,+zvfhmin < %s | FileCheck %s --check-prefixes=CHECK,ZVFHMIN

declare <4 x i7> @llvm.vp.fptoui.v4i7.v4f16(<4 x half>, <4 x i1>, i32)

define <4 x i7> @vfptoui_v4i7_v4f16(<4 x half> %va, <4 x i1> %m, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i7_v4f16:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e8, mf4, ta, ma
; ZVFH-NEXT:    vfncvt.rtz.x.f.w v9, v8, v0.t
; ZVFH-NEXT:    vmv1r.v v8, v9
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i7_v4f16:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v9, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfncvt.rtz.x.f.w v8, v9, v0.t
; ZVFHMIN-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; ZVFHMIN-NEXT:    vnsrl.wi v8, v8, 0, v0.t
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i7> @llvm.vp.fptoui.v4i7.v4f16(<4 x half> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i7> %v
}

declare <4 x i8> @llvm.vp.fptoui.v4i8.v4f16(<4 x half>, <4 x i1>, i32)

define <4 x i8> @vfptoui_v4i8_v4f16(<4 x half> %va, <4 x i1> %m, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i8_v4f16:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e8, mf4, ta, ma
; ZVFH-NEXT:    vfncvt.rtz.xu.f.w v9, v8, v0.t
; ZVFH-NEXT:    vmv1r.v v8, v9
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i8_v4f16:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v9, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfncvt.rtz.xu.f.w v8, v9, v0.t
; ZVFHMIN-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; ZVFHMIN-NEXT:    vnsrl.wi v8, v8, 0, v0.t
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f16(<4 x half> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i8> %v
}

define <4 x i8> @vfptoui_v4i8_v4f16_unmasked(<4 x half> %va, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i8_v4f16_unmasked:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e8, mf4, ta, ma
; ZVFH-NEXT:    vfncvt.rtz.xu.f.w v9, v8
; ZVFH-NEXT:    vmv1r.v v8, v9
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i8_v4f16_unmasked:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v9, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfncvt.rtz.xu.f.w v8, v9
; ZVFHMIN-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; ZVFHMIN-NEXT:    vnsrl.wi v8, v8, 0
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f16(<4 x half> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i8> %v
}

declare <4 x i16> @llvm.vp.fptoui.v4i16.v4f16(<4 x half>, <4 x i1>, i32)

define <4 x i16> @vfptoui_v4i16_v4f16(<4 x half> %va, <4 x i1> %m, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i16_v4f16:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFH-NEXT:    vfcvt.rtz.xu.f.v v8, v8, v0.t
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i16_v4f16:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v9, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfncvt.rtz.xu.f.w v8, v9, v0.t
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i16> @llvm.vp.fptoui.v4i16.v4f16(<4 x half> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i16> %v
}

define <4 x i16> @vfptoui_v4i16_v4f16_unmasked(<4 x half> %va, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i16_v4f16_unmasked:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFH-NEXT:    vfcvt.rtz.xu.f.v v8, v8
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i16_v4f16_unmasked:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v9, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfncvt.rtz.xu.f.w v8, v9
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i16> @llvm.vp.fptoui.v4i16.v4f16(<4 x half> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i16> %v
}

declare <4 x i32> @llvm.vp.fptoui.v4i32.v4f16(<4 x half>, <4 x i1>, i32)

define <4 x i32> @vfptoui_v4i32_v4f16(<4 x half> %va, <4 x i1> %m, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i32_v4f16:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFH-NEXT:    vfwcvt.rtz.xu.f.v v9, v8, v0.t
; ZVFH-NEXT:    vmv1r.v v8, v9
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i32_v4f16:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v9, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; ZVFHMIN-NEXT:    vfcvt.rtz.xu.f.v v8, v9, v0.t
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i32> @llvm.vp.fptoui.v4i32.v4f16(<4 x half> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i32> %v
}

define <4 x i32> @vfptoui_v4i32_v4f16_unmasked(<4 x half> %va, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i32_v4f16_unmasked:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFH-NEXT:    vfwcvt.rtz.xu.f.v v9, v8
; ZVFH-NEXT:    vmv1r.v v8, v9
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i32_v4f16_unmasked:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v9, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; ZVFHMIN-NEXT:    vfcvt.rtz.xu.f.v v8, v9
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i32> @llvm.vp.fptoui.v4i32.v4f16(<4 x half> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i32> %v
}

declare <4 x i64> @llvm.vp.fptoui.v4i64.v4f16(<4 x half>, <4 x i1>, i32)

define <4 x i64> @vfptoui_v4i64_v4f16(<4 x half> %va, <4 x i1> %m, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i64_v4f16:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFH-NEXT:    vfwcvt.f.f.v v10, v8, v0.t
; ZVFH-NEXT:    vsetvli zero, zero, e32, m1, ta, ma
; ZVFH-NEXT:    vfwcvt.rtz.xu.f.v v8, v10, v0.t
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i64_v4f16:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v10, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.rtz.xu.f.v v8, v10, v0.t
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i64> @llvm.vp.fptoui.v4i64.v4f16(<4 x half> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i64> %v
}

define <4 x i64> @vfptoui_v4i64_v4f16_unmasked(<4 x half> %va, i32 zeroext %evl) {
; ZVFH-LABEL: vfptoui_v4i64_v4f16_unmasked:
; ZVFH:       # %bb.0:
; ZVFH-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; ZVFH-NEXT:    vfwcvt.f.f.v v10, v8
; ZVFH-NEXT:    vsetvli zero, zero, e32, m1, ta, ma
; ZVFH-NEXT:    vfwcvt.rtz.xu.f.v v8, v10
; ZVFH-NEXT:    ret
;
; ZVFHMIN-LABEL: vfptoui_v4i64_v4f16_unmasked:
; ZVFHMIN:       # %bb.0:
; ZVFHMIN-NEXT:    vsetivli zero, 4, e16, mf2, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.f.f.v v10, v8
; ZVFHMIN-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; ZVFHMIN-NEXT:    vfwcvt.rtz.xu.f.v v8, v10
; ZVFHMIN-NEXT:    ret
  %v = call <4 x i64> @llvm.vp.fptoui.v4i64.v4f16(<4 x half> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i64> %v
}

declare <4 x i8> @llvm.vp.fptoui.v4i8.v4f32(<4 x float>, <4 x i1>, i32)

define <4 x i8> @vfptoui_v4i8_v4f32(<4 x float> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i8_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v9, v8, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v9, 0, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f32(<4 x float> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i8> %v
}

define <4 x i8> @vfptoui_v4i8_v4f32_unmasked(<4 x float> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i8_v4f32_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v9, v8
; CHECK-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v9, 0
; CHECK-NEXT:    ret
  %v = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f32(<4 x float> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i8> %v
}

declare <4 x i16> @llvm.vp.fptoui.v4i16.v4f32(<4 x float>, <4 x i1>, i32)

define <4 x i16> @vfptoui_v4i16_v4f32(<4 x float> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i16_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v9, v8, v0.t
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x i16> @llvm.vp.fptoui.v4i16.v4f32(<4 x float> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i16> %v
}

define <4 x i16> @vfptoui_v4i16_v4f32_unmasked(<4 x float> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i16_v4f32_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v9, v8
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x i16> @llvm.vp.fptoui.v4i16.v4f32(<4 x float> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i16> %v
}

declare <4 x i32> @llvm.vp.fptoui.v4i32.v4f32(<4 x float>, <4 x i1>, i32)

define <4 x i32> @vfptoui_v4i32_v4f32(<4 x float> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i32_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v8, v8, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x i32> @llvm.vp.fptoui.v4i32.v4f32(<4 x float> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i32> %v
}

define <4 x i32> @vfptoui_v4i32_v4f32_unmasked(<4 x float> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i32_v4f32_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v8, v8
; CHECK-NEXT:    ret
  %v = call <4 x i32> @llvm.vp.fptoui.v4i32.v4f32(<4 x float> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i32> %v
}

declare <4 x i64> @llvm.vp.fptoui.v4i64.v4f32(<4 x float>, <4 x i1>, i32)

define <4 x i64> @vfptoui_v4i64_v4f32(<4 x float> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i64_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfwcvt.rtz.xu.f.v v10, v8, v0.t
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x i64> @llvm.vp.fptoui.v4i64.v4f32(<4 x float> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i64> %v
}

define <4 x i64> @vfptoui_v4i64_v4f32_unmasked(<4 x float> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i64_v4f32_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfwcvt.rtz.xu.f.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x i64> @llvm.vp.fptoui.v4i64.v4f32(<4 x float> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i64> %v
}

declare <4 x i8> @llvm.vp.fptoui.v4i8.v4f64(<4 x double>, <4 x i1>, i32)

define <4 x i8> @vfptoui_v4i8_v4f64(<4 x double> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i8_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v8, 0, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f64(<4 x double> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i8> %v
}

define <4 x i8> @vfptoui_v4i8_v4f64_unmasked(<4 x double> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i8_v4f64_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0
; CHECK-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v8, 0
; CHECK-NEXT:    ret
  %v = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f64(<4 x double> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i8> %v
}

declare <4 x i16> @llvm.vp.fptoui.v4i16.v4f64(<4 x double>, <4 x i1>, i32)

define <4 x i16> @vfptoui_v4i16_v4f64(<4 x double> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i16_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x i16> @llvm.vp.fptoui.v4i16.v4f64(<4 x double> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i16> %v
}

define <4 x i16> @vfptoui_v4i16_v4f64_unmasked(<4 x double> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i16_v4f64_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0
; CHECK-NEXT:    ret
  %v = call <4 x i16> @llvm.vp.fptoui.v4i16.v4f64(<4 x double> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i16> %v
}

declare <4 x i32> @llvm.vp.fptoui.v4i32.v4f64(<4 x double>, <4 x i1>, i32)

define <4 x i32> @vfptoui_v4i32_v4f64(<4 x double> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i32_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8, v0.t
; CHECK-NEXT:    vmv.v.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x i32> @llvm.vp.fptoui.v4i32.v4f64(<4 x double> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i32> %v
}

define <4 x i32> @vfptoui_v4i32_v4f64_unmasked(<4 x double> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i32_v4f64_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8
; CHECK-NEXT:    vmv.v.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x i32> @llvm.vp.fptoui.v4i32.v4f64(<4 x double> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i32> %v
}

declare <4 x i64> @llvm.vp.fptoui.v4i64.v4f64(<4 x double>, <4 x i1>, i32)

define <4 x i64> @vfptoui_v4i64_v4f64(<4 x double> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i64_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v8, v8, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x i64> @llvm.vp.fptoui.v4i64.v4f64(<4 x double> %va, <4 x i1> %m, i32 %evl)
  ret <4 x i64> %v
}

define <4 x i64> @vfptoui_v4i64_v4f64_unmasked(<4 x double> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v4i64_v4f64_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v8, v8
; CHECK-NEXT:    ret
  %v = call <4 x i64> @llvm.vp.fptoui.v4i64.v4f64(<4 x double> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x i64> %v
}

declare <32 x i64> @llvm.vp.fptoui.v32i64.v32f64(<32 x double>, <32 x i1>, i32)

define <32 x i64> @vfptoui_v32i64_v32f64(<32 x double> %va, <32 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v32i64_v32f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli zero, 2, e8, mf4, ta, ma
; CHECK-NEXT:    li a2, 16
; CHECK-NEXT:    vslidedown.vi v24, v0, 2
; CHECK-NEXT:    mv a1, a0
; CHECK-NEXT:    bltu a0, a2, .LBB25_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    li a1, 16
; CHECK-NEXT:  .LBB25_2:
; CHECK-NEXT:    vsetvli zero, a1, e64, m8, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v8, v8, v0.t
; CHECK-NEXT:    addi a1, a0, -16
; CHECK-NEXT:    sltu a0, a0, a1
; CHECK-NEXT:    addi a0, a0, -1
; CHECK-NEXT:    and a0, a0, a1
; CHECK-NEXT:    vsetvli zero, a0, e64, m8, ta, ma
; CHECK-NEXT:    vmv1r.v v0, v24
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v16, v16, v0.t
; CHECK-NEXT:    ret
  %v = call <32 x i64> @llvm.vp.fptoui.v32i64.v32f64(<32 x double> %va, <32 x i1> %m, i32 %evl)
  ret <32 x i64> %v
}

define <32 x i64> @vfptoui_v32i64_v32f64_unmasked(<32 x double> %va, i32 zeroext %evl) {
; CHECK-LABEL: vfptoui_v32i64_v32f64_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a2, 16
; CHECK-NEXT:    mv a1, a0
; CHECK-NEXT:    bltu a0, a2, .LBB26_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    li a1, 16
; CHECK-NEXT:  .LBB26_2:
; CHECK-NEXT:    vsetvli zero, a1, e64, m8, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v8, v8
; CHECK-NEXT:    addi a1, a0, -16
; CHECK-NEXT:    sltu a0, a0, a1
; CHECK-NEXT:    addi a0, a0, -1
; CHECK-NEXT:    and a0, a0, a1
; CHECK-NEXT:    vsetvli zero, a0, e64, m8, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v16, v16
; CHECK-NEXT:    ret
  %v = call <32 x i64> @llvm.vp.fptoui.v32i64.v32f64(<32 x double> %va, <32 x i1> shufflevector (<32 x i1> insertelement (<32 x i1> undef, i1 true, i32 0), <32 x i1> undef, <32 x i32> zeroinitializer), i32 %evl)
  ret <32 x i64> %v
}