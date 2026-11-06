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
