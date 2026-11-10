/* 211872577 Gal Shoshan */

.section .data


.section .text
.globl strlen_sse42
.type strlen_sse42, @function

strlen_sse42:
   # size_t strlen_sse42(const char *str): rdi = str, result in rax.
    # Every load is aligned (16 B, or 64 B in the main loop), so no load ever
    # crosses into a page the string does not touch: reading past the NUL is safe.
    pxor      %xmm0, %xmm0              # xmm0 = 0: NUL comparand / empty PCMPISTRI operand
    movq      %rdi, %rax
    andq      $-16, %rax                # rax = str rounded down to 16
    movl      %edi, %ecx
    andl      $15, %ecx                 # ecx = misalignment of str
 
    # Head: NUL mask of the aligned block, shifted to drop bytes before str.
    movdqa    (%rax), %xmm1
    pcmpeqb   %xmm0, %xmm1
    pmovmskb  %xmm1, %edx               # bit i = 1 if byte i == 0
    shrl      %cl, %edx
    bsfl      %edx, %edx                # ZF = 1 if str has no NUL in this block
    jz        .Lfirst64
    movl      %edx, %eax
    ret
 
    # Rest of the first 64 bytes with SSE4.2 PCMPISTRI. imm 0x08 = unsigned bytes,
    # EQUAL_EACH vs. an empty string -> ECX = index of first NUL, ZF = 1 if one exists.
.Lfirst64:
    pcmpistri $0x08, 16(%rax), %xmm0
    jz        .Lhit16
    pcmpistri $0x08, 32(%rax), %xmm0
    jz        .Lhit32
    pcmpistri $0x08, 48(%rax), %xmm0
    jz        .Lhit48
    addq      $64, %rax
    andq      $-64, %rax                # 64-byte aligned; all bytes below rax are non-NUL
 
    # Main loop: 64 bytes per iteration. Unsigned min of the four blocks is 0
    # iff some byte is NUL, so one compare + one branch covers a whole cache line.
    .p2align 4
.Lloop:
    movdqa    (%rax), %xmm1
    movdqa    16(%rax), %xmm2
    movdqa    32(%rax), %xmm3
    movdqa    48(%rax), %xmm4
    pminub    %xmm2, %xmm1
    pminub    %xmm4, %xmm3
    pminub    %xmm3, %xmm1
    pcmpeqb   %xmm0, %xmm1
    pmovmskb  %xmm1, %edx
    testl     %edx, %edx
    jnz       .Lfound
    addq      $64, %rax
    jmp       .Lloop
 
    # A NUL is in this cache line: PCMPISTRI pinpoints the block and its index.
.Lfound:
    pcmpistri $0x08, (%rax), %xmm0
    jz        .Ldone
    pcmpistri $0x08, 16(%rax), %xmm0
    jz        .Lhit16
    pcmpistri $0x08, 32(%rax), %xmm0
    jz        .Lhit32
    pcmpistri $0x08, 48(%rax), %xmm0
.Lhit48:
    addq      $16, %rax
.Lhit32:
    addq      $16, %rax
.Lhit16:
    addq      $16, %rax
.Ldone:
    addq      %rcx, %rax                # address of the NUL
    subq      %rdi, %rax                # length = NUL address - str
    ret