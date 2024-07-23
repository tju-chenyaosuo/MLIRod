; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 -mattr=+v,+experimental-zvksh \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 -mattr=+v,+experimental-zvksh \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK

declare <vscale x 8 x i32> @llvm.riscv.vsm3me.nxv8i32.nxv8i32(
  <vscale x 8 x i32>,
  <vscale x 8 x i32>,
  <vscale x 8 x i32>,
  iXLen)

define <vscale x 8 x i32> @intrinsic_vsm3me_vv_nxv8i32_nxv8i32(<vscale x 8 x i32> %0, <vscale x 8 x i32> %1, iXLen %2) nounwind {
; CHECK-LABEL: intrinsic_vsm3me_vv_nxv8i32_nxv8i32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m4, ta, ma
; CHECK-NEXT:    vsm3me.vv v8, v8, v12
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 8 x i32> @llvm.riscv.vsm3me.nxv8i32.nxv8i32(
    <vscale x 8 x i32> undef,
    <vscale x 8 x i32> %0,
    <vscale x 8 x i32> %1,
    iXLen %2)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vsm3me.nxv16i32.nxv16i32(
  <vscale x 16 x i32>,
  <vscale x 16 x i32>,
  <vscale x 16 x i32>,
  iXLen)

define <vscale x 16 x i32> @intrinsic_vsm3me_vv_nxv16i32_nxv16i32(<vscale x 16 x i32> %0, <vscale x 16 x i32> %1, iXLen %2) nounwind {
; CHECK-LABEL: intrinsic_vsm3me_vv_nxv16i32_nxv16i32:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m8, ta, ma
; CHECK-NEXT:    vsm3me.vv v8, v8, v16
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 16 x i32> @llvm.riscv.vsm3me.nxv16i32.nxv16i32(
    <vscale x 16 x i32> undef,
    <vscale x 16 x i32> %0,
    <vscale x 16 x i32> %1,
    iXLen %2)

  ret <vscale x 16 x i32> %a
}