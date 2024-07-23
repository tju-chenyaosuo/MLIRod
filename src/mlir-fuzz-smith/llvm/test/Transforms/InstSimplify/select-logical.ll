; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

define i1 @logical_and_of_or_commute0(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_and_of_or_commute0(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %ynot = xor i1 %y, -1
  %xory = select i1 %x, i1 true, i1 %y
  %xorynot = select i1 %x, i1 true, i1 %ynot
  %and = select i1 %xory, i1 %xorynot, i1 false
  ret i1 %and
}

define <2 x i1> @logical_and_of_or_commute1(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @logical_and_of_or_commute1(
; CHECK-NEXT:    ret <2 x i1> [[X:%.*]]
;
  %ynot = xor <2 x i1> %y, <i1 -1, i1 poison>
  %xory = select <2 x i1> %y, <2 x i1> <i1 true, i1 true>, <2 x i1> %x
  %xorynot = select <2 x i1> %x, <2 x i1> <i1 true, i1 true>, <2 x i1> %ynot
  %and = select <2 x i1> %xory, <2 x i1> %xorynot, <2 x i1> zeroinitializer
  ret <2 x i1> %and
}

define i1 @logical_and_of_or_commute2(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_and_of_or_commute2(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %ynot = xor i1 %y, -1
  %xory = select i1 %x, i1 true, i1 %y
  %xorynot = select i1 %ynot, i1 true, i1 %x
  %and = select i1 %xory, i1 %xorynot, i1 false
  ret i1 %and
}

define i1 @logical_and_of_or_commute3(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_and_of_or_commute3(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %ynot = xor i1 %y, -1
  %xory = select i1 %y, i1 true, i1 %x
  %xorynot = select i1 %ynot, i1 true, i1 %x
  %and = select i1 %xory, i1 %xorynot, i1 false
  ret i1 %and
}

define i1 @logical_and_of_or_commute4(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_and_of_or_commute4(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %ynot = xor i1 %y, -1
  %xory = select i1 %x, i1 true, i1 %y
  %xorynot = select i1 %x, i1 true, i1 %ynot
  %and = select i1 %xorynot, i1 %xory, i1 false
  ret i1 %and
}

define i1 @logical_and_of_or_commute5(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_and_of_or_commute5(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %ynot = xor i1 %y, -1
  %xory = select i1 %y, i1 true, i1 %x
  %xorynot = select i1 %x, i1 true, i1 %ynot
  %and = select i1 %xorynot, i1 %xory, i1 false
  ret i1 %and
}

define i1 @logical_and_of_or_commute6(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_and_of_or_commute6(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %ynot = xor i1 %y, -1
  %xory = select i1 %x, i1 true, i1 %y
  %xorynot = select i1 %ynot, i1 true, i1 %x
  %and = select i1 %xorynot, i1 %xory, i1 false
  ret i1 %and
}

define i1 @logical_and_of_or_commute7(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_and_of_or_commute7(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %ynot = xor i1 %y, -1
  %xory = select i1 %y, i1 true, i1 %x
  %xorynot = select i1 %ynot, i1 true, i1 %x
  %and = select i1 %xorynot, i1 %xory, i1 false
  ret i1 %and
}

; negative test - wrong logic op

define i1 @logical_and_of_or_and(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_and_of_or_and(
; CHECK-NEXT:    [[XANDY:%.*]] = select i1 [[Y:%.*]], i1 [[X:%.*]], i1 false
; CHECK-NEXT:    [[YNOT:%.*]] = xor i1 [[Y]], true
; CHECK-NEXT:    [[XORYNOT:%.*]] = select i1 [[YNOT]], i1 true, i1 [[X]]
; CHECK-NEXT:    [[AND:%.*]] = select i1 [[XORYNOT]], i1 [[XANDY]], i1 false
; CHECK-NEXT:    ret i1 [[AND]]
;
  %xandy = select i1 %y, i1 %x, i1 false
  %ynot = xor i1 %y, -1
  %xorynot = select i1 %ynot, i1 true, i1 %x
  %and = select i1 %xorynot, i1 %xandy, i1 false
  ret i1 %and
}

; negative test - must have common operands

define i1 @logical_and_of_or_no_common_op(i1 %x, i1 %y, i1 %z) {
; CHECK-LABEL: @logical_and_of_or_no_common_op(
; CHECK-NEXT:    [[XORZ:%.*]] = select i1 [[X:%.*]], i1 true, i1 [[Z:%.*]]
; CHECK-NEXT:    [[YNOT:%.*]] = xor i1 [[Y:%.*]], true
; CHECK-NEXT:    [[XORYNOT:%.*]] = select i1 [[X]], i1 true, i1 [[YNOT]]
; CHECK-NEXT:    [[AND:%.*]] = select i1 [[XORYNOT]], i1 [[XORZ]], i1 false
; CHECK-NEXT:    ret i1 [[AND]]
;
  %xorz = select i1 %x, i1 true, i1 %z
  %ynot = xor i1 %y, -1
  %xorynot = select i1 %x, i1 true, i1 %ynot
  %and = select i1 %xorynot, i1 %xorz, i1 false
  ret i1 %and
}

; !(X | Y) && X --> false

define i1 @or_not_and(i1 %x, i1 %y) {
; CHECK-LABEL: @or_not_and(
; CHECK-NEXT:    ret i1 false
;
  %l.and = or i1 %x, %y
  %not = xor i1 %l.and, true
  %r = select i1 %not, i1 %x, i1 false
  ret i1 %r
}

; vector case !(X | Y) && X --> false

define <2 x i1> @or_not_and_vector(<2 x i1>  %x, <2 x i1>  %y) {
; CHECK-LABEL: @or_not_and_vector(
; CHECK-NEXT:    ret <2 x i1> zeroinitializer
;
  %l.and = or <2 x i1> %x, %y
  %not = xor <2 x i1> %l.and, <i1 true, i1 true>
  %r = select <2 x i1>  %not, <2 x i1>  %x, <2 x i1> <i1 false, i1 false>
  ret <2 x i1>  %r
}

; vector case !(X | Y) && X --> false

define <2 x i1> @or_not_and_vector_poison1(<2 x i1>  %x, <2 x i1>  %y) {
; CHECK-LABEL: @or_not_and_vector_poison1(
; CHECK-NEXT:    ret <2 x i1> zeroinitializer
;
  %l.and = or <2 x i1> %x, %y
  %not = xor <2 x i1> %l.and, <i1 poison, i1 true>
  %r = select <2 x i1>  %not, <2 x i1>  %x, <2 x i1> <i1 false, i1 false>
  ret <2 x i1>  %r
}

; vector case !(X | Y) && X --> false

define <2 x i1> @or_not_and_vector_poison2(<2 x i1>  %x, <2 x i1>  %y) {
; CHECK-LABEL: @or_not_and_vector_poison2(
; CHECK-NEXT:    ret <2 x i1> zeroinitializer
;
  %l.and = or <2 x i1> %x, %y
  %not = xor <2 x i1> %l.and, <i1 true, i1 true>
  %r = select <2 x i1>  %not, <2 x i1>  %x, <2 x i1> <i1 poison, i1 false>
  ret <2 x i1>  %r
}


; !(X || Y) && X --> false

define i1 @logical_or_not_and(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_or_not_and(
; CHECK-NEXT:    ret i1 false
;
  %l.and = select i1 %x, i1 true, i1 %y
  %not = xor i1 %l.and, true
  %r = select i1 %not, i1 %x, i1 false
  ret i1 %r
}

; !(X || Y) && Y --> false

define i1 @logical_or_not_and_commute_or(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_or_not_and_commute_or(
; CHECK-NEXT:    ret i1 false
;
  %l.and = select i1 %x, i1 true, i1 %y
  %not = xor i1 %l.and, true
  %r = select i1 %not, i1 %y, i1 false
  ret i1 %r
}

; X && !(X || Y) --> false

define i1 @logical_or_not_commute_and(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_or_not_commute_and(
; CHECK-NEXT:    ret i1 false
;
  %l.and = select i1 %x, i1 true, i1 %y
  %not = xor i1 %l.and, true
  %r = select i1 %x, i1 %not, i1 false
  ret i1 %r
}

; Y && !(X || Y) --> false

define i1 @logical_or_not_commute_and_commute_or(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_or_not_commute_and_commute_or(
; CHECK-NEXT:    ret i1 false
;
  %l.and = select i1 %x, i1 true, i1 %y
  %not = xor i1 %l.and, true
  %r = select i1 %y, i1 %not, i1 false
  ret i1 %r
}

; vector case !(X || Y) && X --> false

define <3 x i1> @logical_or_not_and_vector1(<3 x i1> %x, <3 x i1> %y) {
; CHECK-LABEL: @logical_or_not_and_vector1(
; CHECK-NEXT:    ret <3 x i1> zeroinitializer
;
  %l.and = select <3 x i1> %x, <3 x i1> <i1 true, i1 true, i1 true>, <3 x i1> %y
  %not = xor <3 x i1> %l.and, <i1 true, i1 true, i1 true>
  %r = select <3 x i1> %not, <3 x i1> %x, <3 x i1> <i1 false, i1 false, i1 false>
  ret <3 x i1> %r
}

; TODO: this could transform to false
; vector case !(X || Y) && X --> false

define <3 x i1> @logical_or_not_and_vector1_poison1(<3 x i1> %x, <3 x i1> %y) {
; CHECK-LABEL: @logical_or_not_and_vector1_poison1(
; CHECK-NEXT:    [[L_AND:%.*]] = select <3 x i1> [[X:%.*]], <3 x i1> <i1 true, i1 true, i1 poison>, <3 x i1> [[Y:%.*]]
; CHECK-NEXT:    [[NOT:%.*]] = xor <3 x i1> [[L_AND]], <i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[R:%.*]] = select <3 x i1> [[NOT]], <3 x i1> [[X]], <3 x i1> zeroinitializer
; CHECK-NEXT:    ret <3 x i1> [[R]]
;
  %l.and = select <3 x i1> %x, <3 x i1> <i1 true, i1 true, i1 poison>, <3 x i1> %y
  %not = xor <3 x i1> %l.and, <i1 true, i1 true, i1 true>
  %r = select <3 x i1> %not, <3 x i1> %x, <3 x i1> <i1 false, i1 false, i1 false>
  ret <3 x i1> %r
}

; vector case !(X || Y) && X --> false

define <3 x i1> @logical_or_not_and_vector1_poison2(<3 x i1> %x, <3 x i1> %y) {
; CHECK-LABEL: @logical_or_not_and_vector1_poison2(
; CHECK-NEXT:    ret <3 x i1> zeroinitializer
;
  %l.and = select <3 x i1> %x, <3 x i1> <i1 true, i1 true, i1 true>, <3 x i1> %y
  %not = xor <3 x i1> %l.and, <i1 true, i1 poison, i1 true>
  %r = select <3 x i1> %not, <3 x i1> %x, <3 x i1> <i1 false, i1 false, i1 false>
  ret <3 x i1> %r
}

; vector case !(X || Y) && X --> false

define <3 x i1> @logical_or_not_and_vector1_poison3(<3 x i1> %x, <3 x i1> %y) {
; CHECK-LABEL: @logical_or_not_and_vector1_poison3(
; CHECK-NEXT:    ret <3 x i1> zeroinitializer
;
  %l.and = select <3 x i1> %x, <3 x i1> <i1 true, i1 true, i1 true>, <3 x i1> %y
  %not = xor <3 x i1> %l.and, <i1 true, i1 true, i1 true>
  %r = select <3 x i1> %not, <3 x i1> %x, <3 x i1> <i1 poison, i1 false, i1 false>
  ret <3 x i1> %r
}

; negative test - must have common operands

define i1 @logical_not_or_and_negative1(i1 %x, i1 %y, i1 %z) {
; CHECK-LABEL: @logical_not_or_and_negative1(
; CHECK-NEXT:    [[OR:%.*]] = or i1 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[R:%.*]] = select i1 [[OR]], i1 false, i1 [[Z:%.*]]
; CHECK-NEXT:    ret i1 [[R]]
;
  %or = or i1 %x, %y
  %r = select i1 %or, i1 false, i1 %z
  ret i1 %r
}

; !(x && y) || x --> true

define i1 @logical_nand_logical_or_common_op_commute1(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_nand_logical_or_common_op_commute1(
; CHECK-NEXT:    ret i1 true
;
  %and = select i1 %x, i1 %y, i1 false
  %nand = xor i1 %and, -1
  %or = select i1 %nand, i1 true, i1 %x
  ret i1 %or
}

define <2 x i1> @logical_nand_logical_or_common_op_commute2(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @logical_nand_logical_or_common_op_commute2(
; CHECK-NEXT:    ret <2 x i1> <i1 true, i1 true>
;
  %and = select <2 x i1> %y, <2 x i1> %x, <2 x i1> zeroinitializer
  %nand = xor <2 x i1> %and, <i1 -1, i1 -1>
  %or = select <2 x i1> %nand, <2 x i1> <i1 -1, i1 -1>, <2 x i1> %x
  ret <2 x i1> %or
}

define <2 x i1> @logical_nand_logical_or_common_op_commute3(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @logical_nand_logical_or_common_op_commute3(
; CHECK-NEXT:    ret <2 x i1> <i1 true, i1 true>
;
  %and = select <2 x i1> %x, <2 x i1> %y, <2 x i1> zeroinitializer
  %nand = xor <2 x i1> %and, <i1 -1, i1 poison>
  %or = select <2 x i1> %x, <2 x i1> <i1 -1, i1 poison>, <2 x i1> %nand
  ret <2 x i1> %or
}

define i1 @logical_nand_logical_or_common_op_commute4(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_nand_logical_or_common_op_commute4(
; CHECK-NEXT:    ret i1 true
;
  %and = select i1 %y, i1 %x, i1 false
  %nand = xor i1 %and, -1
  %or = select i1 %x, i1 true, i1 %nand
  ret i1 %or
}

; TODO: This could fold the same as above (we don't match a partial poison vector as logical op).

define <2 x i1> @logical_nand_logical_or_common_op_commute4_poison_vec(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @logical_nand_logical_or_common_op_commute4_poison_vec(
; CHECK-NEXT:    [[AND:%.*]] = select <2 x i1> [[Y:%.*]], <2 x i1> [[X:%.*]], <2 x i1> <i1 false, i1 poison>
; CHECK-NEXT:    [[NAND:%.*]] = xor <2 x i1> [[AND]], <i1 true, i1 true>
; CHECK-NEXT:    [[OR:%.*]] = select <2 x i1> [[X]], <2 x i1> <i1 true, i1 true>, <2 x i1> [[NAND]]
; CHECK-NEXT:    ret <2 x i1> [[OR]]
;
  %and = select <2 x i1> %y, <2 x i1> %x, <2 x i1> <i1 0, i1 poison>
  %nand = xor <2 x i1> %and, <i1 -1, i1 -1>
  %or = select <2 x i1> %x, <2 x i1> <i1 -1, i1 -1>, <2 x i1> %nand
  ret <2 x i1> %or
}

; negative test - need common operand

define i1 @logical_nand_logical_or(i1 %x, i1 %y, i1 %z) {
; CHECK-LABEL: @logical_nand_logical_or(
; CHECK-NEXT:    [[AND:%.*]] = select i1 [[X:%.*]], i1 [[Y:%.*]], i1 false
; CHECK-NEXT:    [[NAND:%.*]] = xor i1 [[AND]], true
; CHECK-NEXT:    [[OR:%.*]] = select i1 [[NAND]], i1 true, i1 [[Z:%.*]]
; CHECK-NEXT:    ret i1 [[OR]]
;
  %and = select i1 %x, i1 %y, i1 false
  %nand = xor i1 %and, -1
  %or = select i1 %nand, i1 true, i1 %z
  ret i1 %or
}

; (X | Y) ? false : X --> false

define i1 @or_select_false_x_case1(i1 %x, i1 %y) {
; CHECK-LABEL: @or_select_false_x_case1(
; CHECK-NEXT:    ret i1 false
;
  %or = or i1 %x, %y
  %r = select i1 %or, i1 false, i1 %x
  ret i1 %r
}

; (X | Y) ? false : X --> false

define i1 @or_select_false_x_case2(i1 %x, i1 %y) {
; CHECK-LABEL: @or_select_false_x_case2(
; CHECK-NEXT:    ret i1 false
;
  %or = or i1 %x, %y
  %r = select i1 %or, i1 false, i1 %y
  ret i1 %r
}

; vector case (X | Y) ? false : X --> false

define <2 x i1> @or_select_false_x_vector(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @or_select_false_x_vector(
; CHECK-NEXT:    ret <2 x i1> zeroinitializer
;
  %or = or <2 x i1> %x, %y
  %r = select <2 x i1> %or, <2 x i1> <i1 false, i1 false>, <2 x i1> %x
  ret <2 x i1> %r
}

; vector poison case (X | Y) ? false : X --> false

define <2 x i1> @or_select_false_x_vector_poison(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @or_select_false_x_vector_poison(
; CHECK-NEXT:    ret <2 x i1> zeroinitializer
;
  %or = or <2 x i1> %x, %y
  %r = select <2 x i1> %or, <2 x i1> <i1 poison, i1 false>, <2 x i1> %x
  ret <2 x i1> %r
}

; (X || Y) ? false : X --> false

define i1 @logical_or_select_false_x_case1(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_or_select_false_x_case1(
; CHECK-NEXT:    ret i1 false
;
  %or = select i1 %x, i1 true, i1 %y
  %r = select i1 %or, i1 false, i1 %x
  ret i1 %r
}

; (X || Y) ? false : X --> false

define i1 @logical_or_select_false_x_case2(i1 %x, i1 %y) {
; CHECK-LABEL: @logical_or_select_false_x_case2(
; CHECK-NEXT:    ret i1 false
;
  %or = select i1 %y, i1 true, i1 %x
  %r = select i1 %or, i1 false, i1 %x
  ret i1 %r
}

; vector case (X || Y) ? false : X --> false

define <2 x i1> @logical_or_select_false_x_vector(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @logical_or_select_false_x_vector(
; CHECK-NEXT:    ret <2 x i1> zeroinitializer
;
  %or = select <2 x i1> %y, <2 x i1> <i1 true, i1 true>, <2 x i1> %x
  %r = select <2 x i1> %or, <2 x i1> <i1 false, i1 false>, <2 x i1> %x
  ret <2 x i1> %r
}

; TODO: this could transform to false
; vector poison case (X || Y) ? false : X --> false

define <2 x i1> @logical_or_select_false_x_vector_poison1(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @logical_or_select_false_x_vector_poison1(
; CHECK-NEXT:    [[OR:%.*]] = select <2 x i1> [[Y:%.*]], <2 x i1> <i1 poison, i1 true>, <2 x i1> [[X:%.*]]
; CHECK-NEXT:    [[R:%.*]] = select <2 x i1> [[OR]], <2 x i1> zeroinitializer, <2 x i1> [[X]]
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %or = select <2 x i1> %y, <2 x i1> <i1 poison, i1 true>, <2 x i1> %x
  %r = select <2 x i1> %or, <2 x i1> <i1 false, i1 false>, <2 x i1> %x
  ret <2 x i1> %r
}

; vector poison case (X || Y) ? false : X --> false

define <2 x i1> @logical_or_select_false_x_vector_poison2(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @logical_or_select_false_x_vector_poison2(
; CHECK-NEXT:    ret <2 x i1> zeroinitializer
;
  %or = select <2 x i1> %y, <2 x i1> <i1 true, i1 true>, <2 x i1> %x
  %r = select <2 x i1> %or, <2 x i1> <i1 poison, i1 false>, <2 x i1> %x
  ret <2 x i1> %r
}

; negative test - must have common operands

define i1 @or_select_false_x_negative(i1 %x, i1 %y, i1 %z) {
; CHECK-LABEL: @or_select_false_x_negative(
; CHECK-NEXT:    [[OR:%.*]] = or i1 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[R:%.*]] = select i1 [[OR]], i1 false, i1 [[Z:%.*]]
; CHECK-NEXT:    ret i1 [[R]]
;
  %or = or i1 %x, %y
  %r = select i1 %or, i1 false, i1 %z
  ret i1 %r
}

; (X || Y) ? X : Y --> X

define i1 @select_or_same_op(i1 %x, i1 %y) {
; CHECK-LABEL: @select_or_same_op(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %or = or i1 %x, %y
  %r = select i1 %or, i1 %x, i1 %y
  ret i1 %r
}


define i1 @select_or_same_op_commute(i1 %x, i1 %y) {
; CHECK-LABEL: @select_or_same_op_commute(
; CHECK-NEXT:    ret i1 [[Y:%.*]]
;
  %or = or i1 %x, %y
  %r = select i1 %or, i1 %y, i1 %x
  ret i1 %r
}


define <2 x i1> @select_or_same_op_vector1(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @select_or_same_op_vector1(
; CHECK-NEXT:    ret <2 x i1> [[X:%.*]]
;
  %or = or <2 x i1> %x, %y
  %r = select <2 x i1> %or, <2 x i1> %x, <2 x i1> %y
  ret <2 x i1> %r
}


define i1 @select_logic_or1_same_op(i1 %x, i1 %y) {
; CHECK-LABEL: @select_logic_or1_same_op(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %or = select i1 %x, i1 true, i1 %y
  %r = select i1 %or, i1 %x, i1 %y
  ret i1 %r
}


define i1 @select_logic_or2_same_op(i1 %x, i1 %y) {
; CHECK-LABEL: @select_logic_or2_same_op(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %or = select i1 %y, i1 true, i1 %x
  %r = select i1 %or, i1 %x, i1 %y
  ret i1 %r
}


define <2 x i1> @select_or_same_op_vector2(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @select_or_same_op_vector2(
; CHECK-NEXT:    ret <2 x i1> [[X:%.*]]
;
  %or = select <2 x i1> %x, <2 x i1> <i1 true, i1 true>, <2 x i1> %y
  %r = select <2 x i1> %or, <2 x i1> %x, <2 x i1> %y
  ret <2 x i1> %r
}

; TODO: this could transform to X
; (X || Y) ? X : Y --> X

define <2 x i1> @select_or_same_op_vector2_poison(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @select_or_same_op_vector2_poison(
; CHECK-NEXT:    [[OR:%.*]] = select <2 x i1> [[X:%.*]], <2 x i1> <i1 true, i1 poison>, <2 x i1> [[Y:%.*]]
; CHECK-NEXT:    [[R:%.*]] = select <2 x i1> [[OR]], <2 x i1> [[X]], <2 x i1> [[Y]]
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %or = select <2 x i1> %x, <2 x i1> <i1 true, i1 poison>, <2 x i1> %y
  %r = select <2 x i1> %or, <2 x i1> %x, <2 x i1> %y
  ret <2 x i1> %r
}

; negative test - must have common operands

define i1 @select_or_same_op_negative(i1 %x, i1 %y, i1 %z) {
; CHECK-LABEL: @select_or_same_op_negative(
; CHECK-NEXT:    [[OR:%.*]] = or i1 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[R:%.*]] = select i1 [[OR]], i1 [[X]], i1 [[Z:%.*]]
; CHECK-NEXT:    ret i1 [[R]]
;
  %or = or i1 %x, %y
  %r = select i1 %or, i1 %x, i1 %z
  ret i1 %r
}

; (X && Y) ? X : Y --> Y

define i1 @select_and_same_op(i1 %x, i1 %y) {
; CHECK-LABEL: @select_and_same_op(
; CHECK-NEXT:    ret i1 [[Y:%.*]]
;
  %a = and i1 %x, %y
  %r = select i1 %a, i1 %x, i1 %y
  ret i1 %r
}


define i1 @select_and_same_op_commute(i1 %x, i1 %y) {
; CHECK-LABEL: @select_and_same_op_commute(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %a = and i1 %x, %y
  %r = select i1 %a, i1 %y, i1 %x
  ret i1 %r
}


define <2 x i1> @select_and_same_op_vector1(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @select_and_same_op_vector1(
; CHECK-NEXT:    ret <2 x i1> [[Y:%.*]]
;
  %a = and <2 x i1> %x, %y
  %r = select <2 x i1> %a, <2 x i1> %x, <2 x i1> %y
  ret <2 x i1> %r
}


define i1 @select_logic_and1_same_op(i1 %x, i1 %y) {
; CHECK-LABEL: @select_logic_and1_same_op(
; CHECK-NEXT:    ret i1 [[Y:%.*]]
;
  %a = select i1 %x, i1 %y, i1 false
  %r = select i1 %a, i1 %x, i1 %y
  ret i1 %r
}


define i1 @select_logic_and2_same_op(i1 %x, i1 %y) {
; CHECK-LABEL: @select_logic_and2_same_op(
; CHECK-NEXT:    ret i1 [[Y:%.*]]
;
  %a = select i1 %y, i1 %x, i1 false
  %r = select i1 %a, i1 %x, i1 %y
  ret i1 %r
}


define <2 x i1> @select_and_same_op_vector2(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @select_and_same_op_vector2(
; CHECK-NEXT:    ret <2 x i1> [[Y:%.*]]
;
  %a = select <2 x i1> %x, <2 x i1> %y, <2 x i1> zeroinitializer
  %r = select <2 x i1> %a, <2 x i1> %x, <2 x i1> %y
  ret <2 x i1> %r
}

; TODO: this could transform to Y
; (X && Y) ? X : Y --> Y

define <2 x i1> @select_and_same_op_vector2_poison(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @select_and_same_op_vector2_poison(
; CHECK-NEXT:    [[A:%.*]] = select <2 x i1> [[X:%.*]], <2 x i1> [[Y:%.*]], <2 x i1> <i1 false, i1 poison>
; CHECK-NEXT:    [[R:%.*]] = select <2 x i1> [[A]], <2 x i1> [[X]], <2 x i1> [[Y]]
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %a = select <2 x i1> %x, <2 x i1> %y, <2 x i1> <i1 false, i1 poison>
  %r = select <2 x i1> %a, <2 x i1> %x, <2 x i1> %y
  ret <2 x i1> %r
}

; negative test - must have common operands

define i1 @select_and_same_op_negative(i1 %x, i1 %y, i1 %z) {
; CHECK-LABEL: @select_and_same_op_negative(
; CHECK-NEXT:    [[A:%.*]] = and i1 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[R:%.*]] = select i1 [[A]], i1 [[X]], i1 [[Z:%.*]]
; CHECK-NEXT:    ret i1 [[R]]
;
  %a = and i1 %x, %y
  %r = select i1 %a, i1 %x, i1 %z
  ret i1 %r
}

define i1 @and_same_op(i1 %x) {
; CHECK-LABEL: @and_same_op(
; CHECK-NEXT:    ret i1 [[X:%.*]]
;
  %r = select i1 %x, i1 %x, i1 false
  ret i1 %r
}

define <2 x i1> @or_same_op(<2 x i1> %x) {
; CHECK-LABEL: @or_same_op(
; CHECK-NEXT:    ret <2 x i1> [[X:%.*]]
;
  %r = select <2 x i1> %x, <2 x i1> <i1 true, i1 true>, <2 x i1> %x
  ret <2 x i1> %r
}

define <2 x i1> @always_true_same_op(<2 x i1> %x) {
; CHECK-LABEL: @always_true_same_op(
; CHECK-NEXT:    ret <2 x i1> <i1 true, i1 true>
;
  %r = select <2 x i1> %x, <2 x i1> %x, <2 x i1> <i1 poison, i1 true>
  ret <2 x i1> %r
}

define i1 @always_false_same_op(i1 %x) {
; CHECK-LABEL: @always_false_same_op(
; CHECK-NEXT:    ret i1 false
;
  %r = select i1 %x, i1 false, i1 %x
  ret i1 %r
}

; (X && Y) || Y --> Y

define i1 @or_and_common_op_commute0(i1 %x, i1 %y) {
; CHECK-LABEL: @or_and_common_op_commute0(
; CHECK-NEXT:    ret i1 [[Y:%.*]]
;
  %a = select i1 %x, i1 %y, i1 false
  %r = select i1 %a, i1 true, i1 %y
  ret i1 %r
}

define <2 x i1> @or_and_common_op_commute1(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @or_and_common_op_commute1(
; CHECK-NEXT:    ret <2 x i1> [[Y:%.*]]
;
  %a = select <2 x i1> %y, <2 x i1> %x, <2 x i1> zeroinitializer
  %r = select <2 x i1> %a, <2 x i1> <i1 true, i1 true>, <2 x i1> %y
  ret <2 x i1> %r
}

define <2 x i1> @or_and_common_op_commute2(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @or_and_common_op_commute2(
; CHECK-NEXT:    ret <2 x i1> [[Y:%.*]]
;
  %a = select <2 x i1> %x, <2 x i1> %y, <2 x i1> zeroinitializer
  %r = select <2 x i1> %y, <2 x i1> <i1 true, i1 true>, <2 x i1> %a
  ret <2 x i1> %r
}

; TODO: this could fold the same as above

define <2 x i1> @or_and_common_op_commute2_poison(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @or_and_common_op_commute2_poison(
; CHECK-NEXT:    [[A:%.*]] = select <2 x i1> [[X:%.*]], <2 x i1> [[Y:%.*]], <2 x i1> <i1 false, i1 poison>
; CHECK-NEXT:    [[R:%.*]] = select <2 x i1> [[Y]], <2 x i1> <i1 true, i1 true>, <2 x i1> [[A]]
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %a = select <2 x i1> %x, <2 x i1> %y, <2 x i1> <i1 0, i1 poison>
  %r = select <2 x i1> %y, <2 x i1> <i1 true, i1 true>, <2 x i1> %a
  ret <2 x i1> %r
}

define <2 x i1> @or_and_common_op_commute3(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @or_and_common_op_commute3(
; CHECK-NEXT:    ret <2 x i1> [[Y:%.*]]
;
  %a = select <2 x i1> %y, <2 x i1> %x, <2 x i1> zeroinitializer
  %r = select <2 x i1> %y, <2 x i1> <i1 poison, i1 true>, <2 x i1> %a
  ret <2 x i1> %r
}

; negative test

define i1 @or_and_not_common_op(i1 %x, i1 %y, i1 %z) {
; CHECK-LABEL: @or_and_not_common_op(
; CHECK-NEXT:    [[A:%.*]] = select i1 [[X:%.*]], i1 [[Y:%.*]], i1 false
; CHECK-NEXT:    [[R:%.*]] = select i1 [[A]], i1 true, i1 [[Z:%.*]]
; CHECK-NEXT:    ret i1 [[R]]
;
  %a = select i1 %x, i1 %y, i1 false
  %r = select i1 %a, i1 true, i1 %z
  ret i1 %r
}

; (X || Y) && Y --> Y

define i1 @and_or_common_op_commute0(i1 %x, i1 %y) {
; CHECK-LABEL: @and_or_common_op_commute0(
; CHECK-NEXT:    ret i1 [[Y:%.*]]
;
  %o = select i1 %x, i1 true, i1 %y
  %r = select i1 %o, i1 %y, i1 false
  ret i1 %r
}

define <2 x i1> @and_or_common_op_commute1(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @and_or_common_op_commute1(
; CHECK-NEXT:    ret <2 x i1> [[Y:%.*]]
;
  %o = select <2 x i1> %y, <2 x i1> <i1 true, i1 true>, <2 x i1> %x
  %r = select <2 x i1> %o, <2 x i1> %y, <2 x i1> zeroinitializer
  ret <2 x i1> %r
}


define <2 x i1> @and_or_common_op_commute2(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @and_or_common_op_commute2(
; CHECK-NEXT:    ret <2 x i1> [[Y:%.*]]
;
  %o = select <2 x i1> %x, <2 x i1> <i1 true, i1 true>, <2 x i1> %y
  %r = select <2 x i1> %y, <2 x i1> %o, <2 x i1> <i1 0, i1 poison>
  ret <2 x i1> %r
}

define <2 x i1> @and_or_common_op_commute3(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @and_or_common_op_commute3(
; CHECK-NEXT:    ret <2 x i1> [[Y:%.*]]
;
  %o = select <2 x i1> %y, <2 x i1> <i1 true, i1 true>, <2 x i1> %x
  %r = select <2 x i1> %y, <2 x i1> %o, <2 x i1> zeroinitializer
  ret <2 x i1> %r
}

; TODO: this could fold the same as above

define <2 x i1> @and_or_common_op_commute3_poison(<2 x i1> %x, <2 x i1> %y) {
; CHECK-LABEL: @and_or_common_op_commute3_poison(
; CHECK-NEXT:    [[O:%.*]] = select <2 x i1> [[Y:%.*]], <2 x i1> <i1 poison, i1 true>, <2 x i1> [[X:%.*]]
; CHECK-NEXT:    [[R:%.*]] = select <2 x i1> [[Y]], <2 x i1> [[O]], <2 x i1> zeroinitializer
; CHECK-NEXT:    ret <2 x i1> [[R]]
;
  %o = select <2 x i1> %y, <2 x i1> <i1 poison, i1 true>, <2 x i1> %x
  %r = select <2 x i1> %y, <2 x i1> %o, <2 x i1> zeroinitializer
  ret <2 x i1> %r
}

; negative test

define i1 @and_or_not_common_op(i1 %x, i1 %y, i1 %z) {
; CHECK-LABEL: @and_or_not_common_op(
; CHECK-NEXT:    [[O:%.*]] = select i1 [[X:%.*]], i1 true, i1 [[Y:%.*]]
; CHECK-NEXT:    [[R:%.*]] = select i1 [[Z:%.*]], i1 [[O]], i1 false
; CHECK-NEXT:    ret i1 [[R]]
;
  %o = select i1 %x, i1 true, i1 %y
  %r = select i1 %z, i1 %o, i1 false
  ret i1 %r
}