#include "textflag.h"

// func dotAVX2(s float32, a, b *float32, n int) float32
TEXT ·dotAVX2(SB), NOSPLIT, $0-36
	VMOVSS s+0(FP), X0 // s in lane 0, the rest zero
	MOVQ   a+8(FP), AX
	MOVQ   b+16(FP), BX
	MOVQ   n+24(FP), CX
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

wide:
	CMPQ        CX, $32
	JLT         one
	VMOVUPS     (AX), Y4
	VMOVUPS     32(AX), Y5
	VMOVUPS     64(AX), Y6
	VMOVUPS     96(AX), Y7
	VFMADD231PS (BX), Y4, Y0
	VFMADD231PS 32(BX), Y5, Y1
	VFMADD231PS 64(BX), Y6, Y2
	VFMADD231PS 96(BX), Y7, Y3
	ADDQ        $128, AX
	ADDQ        $128, BX
	SUBQ        $32, CX
	JMP         wide

one:
	CMPQ        CX, $8
	JLT         fold
	VMOVUPS     (AX), Y4
	VFMADD231PS (BX), Y4, Y0
	ADDQ        $32, AX
	ADDQ        $32, BX
	SUBQ        $8, CX
	JMP         one

fold:
	VADDPS       Y1, Y0, Y0
	VADDPS       Y3, Y2, Y2
	VADDPS       Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

tail:
	TESTQ       CX, CX
	JZ          done
	VMOVSS      (AX), X4
	VFMADD231SS (BX), X4, X0
	ADDQ        $4, AX
	ADDQ        $4, BX
	DECQ        CX
	JMP         tail

done:
	VMOVSS     X0, ret+32(FP)
	VZEROUPPER
	RET

// func axpyAVX2(w float32, x, y *float32, n int)
TEXT ·axpyAVX2(SB), NOSPLIT, $0-32
	VBROADCASTSS w+0(FP), Y0
	MOVQ         x+8(FP), AX
	MOVQ         y+16(FP), BX
	MOVQ         n+24(FP), CX

wide:
	CMPQ        CX, $32
	JLT         one
	VMOVUPS     (BX), Y1
	VMOVUPS     32(BX), Y2
	VMOVUPS     64(BX), Y3
	VMOVUPS     96(BX), Y4
	VFMADD231PS (AX), Y0, Y1
	VFMADD231PS 32(AX), Y0, Y2
	VFMADD231PS 64(AX), Y0, Y3
	VFMADD231PS 96(AX), Y0, Y4
	VMOVUPS     Y1, (BX)
	VMOVUPS     Y2, 32(BX)
	VMOVUPS     Y3, 64(BX)
	VMOVUPS     Y4, 96(BX)
	ADDQ        $128, AX
	ADDQ        $128, BX
	SUBQ        $32, CX
	JMP         wide

one:
	CMPQ        CX, $8
	JLT         tail
	VMOVUPS     (BX), Y1
	VFMADD231PS (AX), Y0, Y1
	VMOVUPS     Y1, (BX)
	ADDQ        $32, AX
	ADDQ        $32, BX
	SUBQ        $8, CX
	JMP         one

tail:
	TESTQ       CX, CX
	JZ          done
	VMOVSS      (BX), X1
	VFMADD231SS (AX), X0, X1
	VMOVSS      X1, (BX)
	ADDQ        $4, AX
	ADDQ        $4, BX
	DECQ        CX
	JMP         tail

done:
	VZEROUPPER
	RET

// func cpuid(leaf, sub uint32) (a, b, c, d uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL leaf+0(FP), AX
	MOVL sub+4(FP), CX
	CPUID
	MOVL AX, a+8(FP)
	MOVL BX, b+12(FP)
	MOVL CX, c+16(FP)
	MOVL DX, d+20(FP)
	RET

// func xgetbv() uint32
TEXT ·xgetbv(SB), NOSPLIT, $0-4
	MOVL   $0, CX
	XGETBV
	MOVL   AX, ret+0(FP)
	RET

// STEP is one of a polynomial's: p·d, rounded, plus the coefficient at off
#define STEP(off, d, p) \
	VMULPS       d, p, p    \
	VBROADCASTSS off(DX), Y8 \
	VADDPS       Y8, p, p

// func geluAVX2(x *float32, n int, k *float32)
// gelu over n floats, a multiple of 8: no multiply fused with its add, and the
// minima and maxima's operands in the order that keeps a NaN
TEXT ·geluAVX2(SB), NOSPLIT, $0-24
	MOVQ         x+0(FP), AX
	MOVQ         n+8(FP), CX
	MOVQ         k+16(FP), DX
	VPCMPEQD     Y15, Y15, Y15
	VPSRLD       $1, Y15, Y15 // every bit but the sign
	VXORPS       Y14, Y14, Y14
	VBROADCASTSS (DX), Y13    // the tail's end
	VBROADCASTSS 4(DX), Y12   // the split
	VBROADCASTSS 8(DX), Y11   // the near piece's middle
	VBROADCASTSS 56(DX), Y10  // the far piece's

loop:
	CMPQ         CX, $8
	JLT          done
	VMOVUPS      (AX), Y0
	VANDPS       Y15, Y0, Y1
	VMINPS       Y1, Y13, Y1  // a
	VMINPS       Y1, Y12, Y2  // a up to the split
	VMAXPS       Y1, Y12, Y3  // a from the split
	VSUBPS       Y11, Y2, Y4
	VBROADCASTSS 12(DX), Y6
	STEP(16, Y4, Y6)
	STEP(20, Y4, Y6)
	STEP(24, Y4, Y6)
	STEP(28, Y4, Y6)
	STEP(32, Y4, Y6)
	STEP(36, Y4, Y6)
	STEP(40, Y4, Y6)
	STEP(44, Y4, Y6)
	STEP(48, Y4, Y6)
	STEP(52, Y4, Y6)
	VSUBPS       Y2, Y12, Y2
	VMULPS       Y2, Y6, Y6
	VSUBPS       Y10, Y3, Y4
	VBROADCASTSS 60(DX), Y7
	STEP(64, Y4, Y7)
	STEP(68, Y4, Y7)
	STEP(72, Y4, Y7)
	STEP(76, Y4, Y7)
	STEP(80, Y4, Y7)
	STEP(84, Y4, Y7)
	STEP(88, Y4, Y7)
	STEP(92, Y4, Y7)
	STEP(96, Y4, Y7)
	STEP(100, Y4, Y7)
	VSUBPS       Y3, Y13, Y3
	VMULPS       Y3, Y7, Y7
	VADDPS       Y6, Y7, Y6   // the tail
	VMULPS       Y1, Y6, Y6
	VMAXPS       Y14, Y0, Y0  // max(x, 0), zero for a NaN
	VSUBPS       Y6, Y0, Y0
	VMOVUPS      Y0, (AX)
	ADDQ         $32, AX
	SUBQ         $8, CX
	JMP          loop

done:
	VZEROUPPER
	RET
