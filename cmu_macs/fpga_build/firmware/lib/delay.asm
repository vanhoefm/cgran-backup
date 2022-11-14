;--------------------------------------------------------
; File Created by SDCC : FreeWare ANSI-C Compiler
; Version 2.6.0 #4309 (Nov 10 2006)
; This file generated Wed Jan 23 16:41:18 2008
;--------------------------------------------------------
	.module delay
	.optsdcc -mmcs51 --model-small
	
;--------------------------------------------------------
; Public variables in this module
;--------------------------------------------------------
	.globl _mdelay
	.globl _udelay
;--------------------------------------------------------
; special function registers
;--------------------------------------------------------
	.area RSEG    (DATA)
;--------------------------------------------------------
; special function bits
;--------------------------------------------------------
	.area RSEG    (DATA)
;--------------------------------------------------------
; overlayable register banks
;--------------------------------------------------------
	.area REG_BANK_0	(REL,OVR,DATA)
	.ds 8
;--------------------------------------------------------
; internal ram data
;--------------------------------------------------------
	.area DSEG    (DATA)
;--------------------------------------------------------
; overlayable items in internal ram 
;--------------------------------------------------------
	.area OSEG    (OVR,DATA)
;--------------------------------------------------------
; indirectly addressable internal ram data
;--------------------------------------------------------
	.area ISEG    (DATA)
;--------------------------------------------------------
; bit data
;--------------------------------------------------------
	.area BSEG    (BIT)
;--------------------------------------------------------
; paged external ram data
;--------------------------------------------------------
	.area PSEG    (PAG,XDATA)
;--------------------------------------------------------
; external ram data
;--------------------------------------------------------
	.area XSEG    (XDATA)
;--------------------------------------------------------
; external initialized ram data
;--------------------------------------------------------
	.area HOME    (CODE)
	.area GSINIT0 (CODE)
	.area GSINIT1 (CODE)
	.area GSINIT2 (CODE)
	.area GSINIT3 (CODE)
	.area GSINIT4 (CODE)
	.area GSINIT5 (CODE)
	.area GSINIT  (CODE)
	.area GSFINAL (CODE)
	.area CSEG    (CODE)
;--------------------------------------------------------
; global & static initialisations
;--------------------------------------------------------
	.area HOME    (CODE)
	.area GSINIT  (CODE)
	.area GSFINAL (CODE)
	.area GSINIT  (CODE)
;--------------------------------------------------------
; Home
;--------------------------------------------------------
	.area HOME    (CODE)
	.area CSEG    (CODE)
;--------------------------------------------------------
; code
;--------------------------------------------------------
	.area CSEG    (CODE)
;------------------------------------------------------------
;Allocation info for local variables in function 'udelay1'
;------------------------------------------------------------
;------------------------------------------------------------
;	delay.c:27: udelay1 (void) _naked
;	-----------------------------------------
;	 function udelay1
;	-----------------------------------------
_udelay1:
;	naked function: no prologue.
;	delay.c:31: _endasm;
;	genInline
	 ; lcall that got us here took 4 bus cycles
	        ret ; 4 bus cycles
;	Peephole 300	removed redundant label 00101$
;	naked function: no epilogue.
;------------------------------------------------------------
;Allocation info for local variables in function 'udelay'
;------------------------------------------------------------
;usecs                     Allocated to registers r2 
;------------------------------------------------------------
;	delay.c:38: udelay (unsigned char usecs)
;	-----------------------------------------
;	 function udelay
;	-----------------------------------------
_udelay:
	ar2 = 0x02
	ar3 = 0x03
	ar4 = 0x04
	ar5 = 0x05
	ar6 = 0x06
	ar7 = 0x07
	ar0 = 0x00
	ar1 = 0x01
;	genReceive
	mov	r2,dpl
;	delay.c:40: do {
;	genAssign
00101$:
;	delay.c:41: udelay1 ();
;	genCall
	lcall	_udelay1
;	delay.c:42: } while (--usecs != 0);
;	genMinus
;	genMinusDec
;	genCmpEq
;	gencjneshort
;	Peephole 112.b	changed ljmp to sjmp
;	Peephole 198.b	optimized misc jump sequence
;	Peephole 257	optimized decrement with compare
	djnz	r2,00101$
;	Peephole 200.b	removed redundant sjmp
;	Peephole 300	removed redundant label 00108$
;	Peephole 300	removed redundant label 00109$
;	Peephole 300	removed redundant label 00104$
	ret
;------------------------------------------------------------
;Allocation info for local variables in function 'mdelay1'
;------------------------------------------------------------
;------------------------------------------------------------
;	delay.c:54: mdelay1 (void) _naked
;	-----------------------------------------
;	 function mdelay1
;	-----------------------------------------
_mdelay1:
;	naked function: no prologue.
;	delay.c:65: _endasm;
;	genInline
	        mov dptr,#(-1200 & 0xffff)
002$:
	        inc dptr ; 3 bus cycles
	        mov a, dpl ; 2 bus cycles
	        orl a, dph ; 2 bus cycles
	        jnz 002$ ; 3 bus cycles
	        ret
;	Peephole 300	removed redundant label 00101$
;	naked function: no epilogue.
;------------------------------------------------------------
;Allocation info for local variables in function 'mdelay'
;------------------------------------------------------------
;msecs                     Allocated to registers r2 r3 
;------------------------------------------------------------
;	delay.c:69: mdelay (unsigned int msecs)
;	-----------------------------------------
;	 function mdelay
;	-----------------------------------------
_mdelay:
;	genReceive
	mov	r2,dpl
	mov	r3,dph
;	delay.c:71: do {
;	genAssign
00101$:
;	delay.c:72: mdelay1 ();
;	genCall
	lcall	_mdelay1
;	delay.c:73: } while (--msecs != 0);
;	genMinus
;	genMinusDec
	dec	r2
	cjne	r2,#0xff,00108$
	dec	r3
00108$:
;	genCmpEq
;	gencjneshort
;	Peephole 112.b	changed ljmp to sjmp
;	Peephole 198.a	optimized misc jump sequence
	cjne	r2,#0x00,00101$
	cjne	r3,#0x00,00101$
;	Peephole 200.b	removed redundant sjmp
;	Peephole 300	removed redundant label 00109$
;	Peephole 300	removed redundant label 00110$
;	Peephole 300	removed redundant label 00104$
	ret
	.area CSEG    (CODE)
	.area CONST   (CODE)
