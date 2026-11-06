#include "textflag.h"

// func dotNEON(s float32, a, b *float32, n int) float32
TEXT ·dotNEON(SB), NOSPLIT, $0-36
	FMOVS s+0(FP), F0 // s in lane 0, the rest zero
	MOVD  a+8(FP), R0
	MOVD  b+16(FP), R1
	MOVD  n+24(FP), R2
	VEOR  V1.B16, V1.B16, V1.B16
	VEOR  V2.B16, V2.B16, V2.B16
	VEOR  V3.B16, V3.B16, V3.B16

wide:
	CMP    $16, R2
	BLT    one
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VFMLA  V4.S4, V8.S4, V0.S4
	VFMLA  V5.S4, V9.S4, V1.S4
	VFMLA  V6.S4, V10.S4, V2.S4
	VFMLA  V7.S4, V11.S4, V3.S4
	SUB    $16, R2
	B      wide

one:
	CMP    $4, R2
	BLT    fold
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R1), [V8.S4]
	VFMLA  V4.S4, V8.S4, V0.S4
	SUB    $4, R2
	B      one

fold:
	VFADD  V1.S4, V0.S4, V0.S4
	VFADD  V3.S4, V2.S4, V2.S4
	VFADD  V2.S4, V0.S4, V0.S4
	VFADDP V0.S4, V0.S4, V0.S4
	VFADDP V0.S4, V0.S4, V0.S4

tail:
	CBZ     R2, done
	FMOVS.P 4(R0), F4
	FMOVS.P 4(R1), F8
	FMADDS  F8, F0, F4, F0
	SUB     $1, R2
	B       tail

done:
	FMOVS F0, ret+32(FP)
	RET

// func axpyNEON(w float32, x, y *float32, n int)
TEXT ·axpyNEON(SB), NOSPLIT, $0-32
	FMOVS w+0(FP), F0
	VDUP  V0.S[0], V1.S4
	MOVD  x+8(FP), R0
	MOVD  y+16(FP), R1
	MOVD  n+24(FP), R2

wide:
	CMP    $16, R2
	BLT    one
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1   (R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VFMLA  V1.S4, V4.S4, V8.S4
	VFMLA  V1.S4, V5.S4, V9.S4
	VFMLA  V1.S4, V6.S4, V10.S4
	VFMLA  V1.S4, V7.S4, V11.S4
	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R1)
	SUB    $16, R2
	B      wide

one:
	CMP    $4, R2
	BLT    tail
	VLD1.P 16(R0), [V4.S4]
	VLD1   (R1), [V8.S4]
	VFMLA  V1.S4, V4.S4, V8.S4
	VST1.P [V8.S4], 16(R1)
	SUB    $4, R2
	B      one

tail:
	CBZ     R2, done
	FMOVS.P 4(R0), F4
	FMOVS   (R1), F8
	FMADDS  F4, F8, F0, F8
	FMOVS.P F8, 4(R1)
	SUB     $1, R2
	B       tail

done:
	RET
