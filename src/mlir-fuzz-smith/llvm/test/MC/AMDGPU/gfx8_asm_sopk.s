// RUN: llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s

s_movk_i32 s5, 0x3141
// CHECK: [0x41,0x31,0x05,0xb0]

s_movk_i32 s101, 0x3141
// CHECK: [0x41,0x31,0x65,0xb0]

s_movk_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0x66,0xb0]

s_movk_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0x67,0xb0]

s_movk_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0x6a,0xb0]

s_movk_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0x6b,0xb0]

s_movk_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0x6c,0xb0]

s_movk_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0x6d,0xb0]

s_movk_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0x6e,0xb0]

s_movk_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0x6f,0xb0]

s_movk_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0x7b,0xb0]

s_movk_i32 m0, 0x3141
// CHECK: [0x41,0x31,0x7c,0xb0]

s_movk_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0x7e,0xb0]

s_movk_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0x7f,0xb0]

s_movk_i32 s5, 0xc1d1
// CHECK: [0xd1,0xc1,0x05,0xb0]

s_cmovk_i32 s5, 0x3141
// CHECK: [0x41,0x31,0x85,0xb0]

s_cmovk_i32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb0]

s_cmovk_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb0]

s_cmovk_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb0]

s_cmovk_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb0]

s_cmovk_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb0]

s_cmovk_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb0]

s_cmovk_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb0]

s_cmovk_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb0]

s_cmovk_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb0]

s_cmovk_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb0]

s_cmovk_i32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb0]

s_cmovk_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb0]

s_cmovk_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb0]

s_cmovk_i32 s5, 0xc1d1
// CHECK: [0xd1,0xc1,0x85,0xb0]

s_cmpk_eq_i32 s1, 0x3141
// CHECK: [0x41,0x31,0x01,0xb1]

s_cmpk_eq_i32 s101, 0x3141
// CHECK: [0x41,0x31,0x65,0xb1]

s_cmpk_eq_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0x66,0xb1]

s_cmpk_eq_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0x67,0xb1]

s_cmpk_eq_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0x6a,0xb1]

s_cmpk_eq_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0x6b,0xb1]

s_cmpk_eq_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0x6c,0xb1]

s_cmpk_eq_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0x6d,0xb1]

s_cmpk_eq_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0x6e,0xb1]

s_cmpk_eq_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0x6f,0xb1]

s_cmpk_eq_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0x7b,0xb1]

s_cmpk_eq_i32 m0, 0x3141
// CHECK: [0x41,0x31,0x7c,0xb1]

s_cmpk_eq_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0x7e,0xb1]

s_cmpk_eq_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0x7f,0xb1]

s_cmpk_eq_i32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x01,0xb1]

s_cmpk_lg_i32 s1, 0x3141
// CHECK: [0x41,0x31,0x81,0xb1]

s_cmpk_lg_i32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb1]

s_cmpk_lg_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb1]

s_cmpk_lg_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb1]

s_cmpk_lg_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb1]

s_cmpk_lg_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb1]

s_cmpk_lg_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb1]

s_cmpk_lg_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb1]

s_cmpk_lg_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb1]

s_cmpk_lg_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb1]

s_cmpk_lg_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb1]

s_cmpk_lg_i32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb1]

s_cmpk_lg_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb1]

s_cmpk_lg_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb1]

s_cmpk_lg_i32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x81,0xb1]

s_cmpk_gt_i32 s1, 0x3141
// CHECK: [0x41,0x31,0x01,0xb2]

s_cmpk_gt_i32 s101, 0x3141
// CHECK: [0x41,0x31,0x65,0xb2]

s_cmpk_gt_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0x66,0xb2]

s_cmpk_gt_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0x67,0xb2]

s_cmpk_gt_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0x6a,0xb2]

s_cmpk_gt_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0x6b,0xb2]

s_cmpk_gt_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0x6c,0xb2]

s_cmpk_gt_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0x6d,0xb2]

s_cmpk_gt_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0x6e,0xb2]

s_cmpk_gt_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0x6f,0xb2]

s_cmpk_gt_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0x7b,0xb2]

s_cmpk_gt_i32 m0, 0x3141
// CHECK: [0x41,0x31,0x7c,0xb2]

s_cmpk_gt_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0x7e,0xb2]

s_cmpk_gt_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0x7f,0xb2]

s_cmpk_gt_i32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x01,0xb2]

s_cmpk_ge_i32 s1, 0x3141
// CHECK: [0x41,0x31,0x81,0xb2]

s_cmpk_ge_i32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb2]

s_cmpk_ge_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb2]

s_cmpk_ge_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb2]

s_cmpk_ge_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb2]

s_cmpk_ge_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb2]

s_cmpk_ge_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb2]

s_cmpk_ge_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb2]

s_cmpk_ge_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb2]

s_cmpk_ge_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb2]

s_cmpk_ge_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb2]

s_cmpk_ge_i32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb2]

s_cmpk_ge_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb2]

s_cmpk_ge_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb2]

s_cmpk_ge_i32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x81,0xb2]

s_cmpk_lt_i32 s1, 0x3141
// CHECK: [0x41,0x31,0x01,0xb3]

s_cmpk_lt_i32 s101, 0x3141
// CHECK: [0x41,0x31,0x65,0xb3]

s_cmpk_lt_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0x66,0xb3]

s_cmpk_lt_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0x67,0xb3]

s_cmpk_lt_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0x6a,0xb3]

s_cmpk_lt_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0x6b,0xb3]

s_cmpk_lt_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0x6c,0xb3]

s_cmpk_lt_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0x6d,0xb3]

s_cmpk_lt_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0x6e,0xb3]

s_cmpk_lt_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0x6f,0xb3]

s_cmpk_lt_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0x7b,0xb3]

s_cmpk_lt_i32 m0, 0x3141
// CHECK: [0x41,0x31,0x7c,0xb3]

s_cmpk_lt_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0x7e,0xb3]

s_cmpk_lt_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0x7f,0xb3]

s_cmpk_lt_i32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x01,0xb3]

s_cmpk_le_i32 s1, 0x3141
// CHECK: [0x41,0x31,0x81,0xb3]

s_cmpk_le_i32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb3]

s_cmpk_le_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb3]

s_cmpk_le_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb3]

s_cmpk_le_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb3]

s_cmpk_le_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb3]

s_cmpk_le_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb3]

s_cmpk_le_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb3]

s_cmpk_le_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb3]

s_cmpk_le_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb3]

s_cmpk_le_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb3]

s_cmpk_le_i32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb3]

s_cmpk_le_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb3]

s_cmpk_le_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb3]

s_cmpk_le_i32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x81,0xb3]

s_cmpk_eq_u32 s1, 0x3141
// CHECK: [0x41,0x31,0x01,0xb4]

s_cmpk_eq_u32 s101, 0x3141
// CHECK: [0x41,0x31,0x65,0xb4]

s_cmpk_eq_u32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0x66,0xb4]

s_cmpk_eq_u32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0x67,0xb4]

s_cmpk_eq_u32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0x6a,0xb4]

s_cmpk_eq_u32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0x6b,0xb4]

s_cmpk_eq_u32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0x6c,0xb4]

s_cmpk_eq_u32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0x6d,0xb4]

s_cmpk_eq_u32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0x6e,0xb4]

s_cmpk_eq_u32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0x6f,0xb4]

s_cmpk_eq_u32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0x7b,0xb4]

s_cmpk_eq_u32 m0, 0x3141
// CHECK: [0x41,0x31,0x7c,0xb4]

s_cmpk_eq_u32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0x7e,0xb4]

s_cmpk_eq_u32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0x7f,0xb4]

s_cmpk_eq_u32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x01,0xb4]

s_cmpk_lg_u32 s1, 0x3141
// CHECK: [0x41,0x31,0x81,0xb4]

s_cmpk_lg_u32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb4]

s_cmpk_lg_u32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb4]

s_cmpk_lg_u32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb4]

s_cmpk_lg_u32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb4]

s_cmpk_lg_u32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb4]

s_cmpk_lg_u32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb4]

s_cmpk_lg_u32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb4]

s_cmpk_lg_u32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb4]

s_cmpk_lg_u32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb4]

s_cmpk_lg_u32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb4]

s_cmpk_lg_u32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb4]

s_cmpk_lg_u32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb4]

s_cmpk_lg_u32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb4]

s_cmpk_lg_u32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x81,0xb4]

s_cmpk_gt_u32 s1, 0x3141
// CHECK: [0x41,0x31,0x01,0xb5]

s_cmpk_gt_u32 s101, 0x3141
// CHECK: [0x41,0x31,0x65,0xb5]

s_cmpk_gt_u32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0x66,0xb5]

s_cmpk_gt_u32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0x67,0xb5]

s_cmpk_gt_u32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0x6a,0xb5]

s_cmpk_gt_u32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0x6b,0xb5]

s_cmpk_gt_u32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0x6c,0xb5]

s_cmpk_gt_u32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0x6d,0xb5]

s_cmpk_gt_u32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0x6e,0xb5]

s_cmpk_gt_u32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0x6f,0xb5]

s_cmpk_gt_u32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0x7b,0xb5]

s_cmpk_gt_u32 m0, 0x3141
// CHECK: [0x41,0x31,0x7c,0xb5]

s_cmpk_gt_u32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0x7e,0xb5]

s_cmpk_gt_u32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0x7f,0xb5]

s_cmpk_gt_u32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x01,0xb5]

s_cmpk_ge_u32 s1, 0x3141
// CHECK: [0x41,0x31,0x81,0xb5]

s_cmpk_ge_u32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb5]

s_cmpk_ge_u32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb5]

s_cmpk_ge_u32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb5]

s_cmpk_ge_u32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb5]

s_cmpk_ge_u32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb5]

s_cmpk_ge_u32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb5]

s_cmpk_ge_u32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb5]

s_cmpk_ge_u32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb5]

s_cmpk_ge_u32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb5]

s_cmpk_ge_u32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb5]

s_cmpk_ge_u32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb5]

s_cmpk_ge_u32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb5]

s_cmpk_ge_u32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb5]

s_cmpk_ge_u32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x81,0xb5]

s_cmpk_lt_u32 s1, 0x3141
// CHECK: [0x41,0x31,0x01,0xb6]

s_cmpk_lt_u32 s101, 0x3141
// CHECK: [0x41,0x31,0x65,0xb6]

s_cmpk_lt_u32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0x66,0xb6]

s_cmpk_lt_u32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0x67,0xb6]

s_cmpk_lt_u32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0x6a,0xb6]

s_cmpk_lt_u32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0x6b,0xb6]

s_cmpk_lt_u32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0x6c,0xb6]

s_cmpk_lt_u32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0x6d,0xb6]

s_cmpk_lt_u32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0x6e,0xb6]

s_cmpk_lt_u32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0x6f,0xb6]

s_cmpk_lt_u32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0x7b,0xb6]

s_cmpk_lt_u32 m0, 0x3141
// CHECK: [0x41,0x31,0x7c,0xb6]

s_cmpk_lt_u32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0x7e,0xb6]

s_cmpk_lt_u32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0x7f,0xb6]

s_cmpk_lt_u32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x01,0xb6]

s_cmpk_le_u32 s1, 0x3141
// CHECK: [0x41,0x31,0x81,0xb6]

s_cmpk_le_u32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb6]

s_cmpk_le_u32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb6]

s_cmpk_le_u32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb6]

s_cmpk_le_u32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb6]

s_cmpk_le_u32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb6]

s_cmpk_le_u32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb6]

s_cmpk_le_u32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb6]

s_cmpk_le_u32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb6]

s_cmpk_le_u32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb6]

s_cmpk_le_u32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb6]

s_cmpk_le_u32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb6]

s_cmpk_le_u32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb6]

s_cmpk_le_u32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb6]

s_cmpk_le_u32 s1, 0xc1d1
// CHECK: [0xd1,0xc1,0x81,0xb6]

s_addk_i32 s5, 0x3141
// CHECK: [0x41,0x31,0x05,0xb7]

s_addk_i32 s101, 0x3141
// CHECK: [0x41,0x31,0x65,0xb7]

s_addk_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0x66,0xb7]

s_addk_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0x67,0xb7]

s_addk_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0x6a,0xb7]

s_addk_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0x6b,0xb7]

s_addk_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0x6c,0xb7]

s_addk_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0x6d,0xb7]

s_addk_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0x6e,0xb7]

s_addk_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0x6f,0xb7]

s_addk_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0x7b,0xb7]

s_addk_i32 m0, 0x3141
// CHECK: [0x41,0x31,0x7c,0xb7]

s_addk_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0x7e,0xb7]

s_addk_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0x7f,0xb7]

s_addk_i32 s5, 0xc1d1
// CHECK: [0xd1,0xc1,0x05,0xb7]

s_mulk_i32 s5, 0x3141
// CHECK: [0x41,0x31,0x85,0xb7]

s_mulk_i32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb7]

s_mulk_i32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb7]

s_mulk_i32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb7]

s_mulk_i32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb7]

s_mulk_i32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb7]

s_mulk_i32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb7]

s_mulk_i32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb7]

s_mulk_i32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb7]

s_mulk_i32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb7]

s_mulk_i32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb7]

s_mulk_i32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb7]

s_mulk_i32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb7]

s_mulk_i32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb7]

s_mulk_i32 s5, 0xc1d1
// CHECK: [0xd1,0xc1,0x85,0xb7]

s_cbranch_i_fork s[2:3], 12609
// CHECK: [0x41,0x31,0x02,0xb8]

s_cbranch_i_fork s[4:5], 12609
// CHECK: [0x41,0x31,0x04,0xb8]

s_cbranch_i_fork s[100:101], 12609
// CHECK: [0x41,0x31,0x64,0xb8]

s_cbranch_i_fork flat_scratch, 12609
// CHECK: [0x41,0x31,0x66,0xb8]

s_cbranch_i_fork vcc, 12609
// CHECK: [0x41,0x31,0x6a,0xb8]

s_cbranch_i_fork tba, 12609
// CHECK: [0x41,0x31,0x6c,0xb8]

s_cbranch_i_fork tma, 12609
// CHECK: [0x41,0x31,0x6e,0xb8]

s_cbranch_i_fork ttmp[10:11], 12609
// CHECK: [0x41,0x31,0x7a,0xb8]

s_cbranch_i_fork exec, 12609
// CHECK: [0x41,0x31,0x7e,0xb8]

s_cbranch_i_fork s[2:3], 49617
// CHECK: [0xd1,0xc1,0x02,0xb8]

s_getreg_b32 s5, 0x3141
// CHECK: [0x41,0x31,0x85,0xb8]

s_getreg_b32 s101, 0x3141
// CHECK: [0x41,0x31,0xe5,0xb8]

s_getreg_b32 flat_scratch_lo, 0x3141
// CHECK: [0x41,0x31,0xe6,0xb8]

s_getreg_b32 flat_scratch_hi, 0x3141
// CHECK: [0x41,0x31,0xe7,0xb8]

s_getreg_b32 vcc_lo, 0x3141
// CHECK: [0x41,0x31,0xea,0xb8]

s_getreg_b32 vcc_hi, 0x3141
// CHECK: [0x41,0x31,0xeb,0xb8]

s_getreg_b32 tba_lo, 0x3141
// CHECK: [0x41,0x31,0xec,0xb8]

s_getreg_b32 tba_hi, 0x3141
// CHECK: [0x41,0x31,0xed,0xb8]

s_getreg_b32 tma_lo, 0x3141
// CHECK: [0x41,0x31,0xee,0xb8]

s_getreg_b32 tma_hi, 0x3141
// CHECK: [0x41,0x31,0xef,0xb8]

s_getreg_b32 ttmp11, 0x3141
// CHECK: [0x41,0x31,0xfb,0xb8]

s_getreg_b32 m0, 0x3141
// CHECK: [0x41,0x31,0xfc,0xb8]

s_getreg_b32 exec_lo, 0x3141
// CHECK: [0x41,0x31,0xfe,0xb8]

s_getreg_b32 exec_hi, 0x3141
// CHECK: [0x41,0x31,0xff,0xb8]

s_getreg_b32 s5, 0xc1d1
// CHECK: [0xd1,0xc1,0x85,0xb8]

s_setreg_b32 0x3141, s1
// CHECK: [0x41,0x31,0x01,0xb9]

s_setreg_b32 0xc1d1, s1
// CHECK: [0xd1,0xc1,0x01,0xb9]

s_setreg_b32 0x3141, s101
// CHECK: [0x41,0x31,0x65,0xb9]

s_setreg_b32 0x3141, flat_scratch_lo
// CHECK: [0x41,0x31,0x66,0xb9]

s_setreg_b32 0x3141, flat_scratch_hi
// CHECK: [0x41,0x31,0x67,0xb9]

s_setreg_b32 0x3141, vcc_lo
// CHECK: [0x41,0x31,0x6a,0xb9]

s_setreg_b32 0x3141, vcc_hi
// CHECK: [0x41,0x31,0x6b,0xb9]

s_setreg_b32 0x3141, tba_lo
// CHECK: [0x41,0x31,0x6c,0xb9]

s_setreg_b32 0x3141, tba_hi
// CHECK: [0x41,0x31,0x6d,0xb9]

s_setreg_b32 0x3141, tma_lo
// CHECK: [0x41,0x31,0x6e,0xb9]

s_setreg_b32 0x3141, tma_hi
// CHECK: [0x41,0x31,0x6f,0xb9]

s_setreg_b32 0x3141, ttmp11
// CHECK: [0x41,0x31,0x7b,0xb9]

s_setreg_b32 0x3141, m0
// CHECK: [0x41,0x31,0x7c,0xb9]

s_setreg_b32 0x3141, exec_lo
// CHECK: [0x41,0x31,0x7e,0xb9]

s_setreg_b32 0x3141, exec_hi
// CHECK: [0x41,0x31,0x7f,0xb9]

s_setreg_imm32_b32 0x3141, 0x11213141
// CHECK: [0x41,0x31,0x00,0xba,0x41,0x31,0x21,0x11]

s_setreg_imm32_b32 0xc1d1, 0x11213141
// CHECK: [0xd1,0xc1,0x00,0xba,0x41,0x31,0x21,0x11]

s_setreg_imm32_b32 0x3141, 0xa1b1c1d1
// CHECK: [0x41,0x31,0x00,0xba,0xd1,0xc1,0xb1,0xa1]