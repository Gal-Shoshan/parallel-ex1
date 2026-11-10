/* 211872577 Gal Shoshan */
 
long decode_c_version(long x, long y, long z) {
    unsigned long t = (unsigned long)y - (unsigned long)z;  /* subq  %rdx, %rsi  */
    unsigned long p = (unsigned long)x * t;                 /* imulq %rsi, %rdi  */
    unsigned long mask = 0UL - (t & 1);                     /* salq/sarq $63: all 1s if t is odd, else 0 */
    return (long)(p ^ mask);                                /* xorq  %rdi, %rax  */
}