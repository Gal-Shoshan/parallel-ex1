/* 211872577 Gal Shoshan */
#include <xmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pmmintrin.h>

#define MAX_STR 255

float formula1(float *x, unsigned int length) {
    /* sqrt(1 + cbrt(sum sqrt(x_k)) / prod(x_k^2 + 1)) in one pass.
       Four independent sum/product accumulators (16 floats per iteration) keep
       several sqrtps/mulps/addps in flight instead of waiting on one dependency chain. */
    const __m128 one = _mm_set1_ps(1.0f);
    __m128 s0 = _mm_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    __m128 p0 = one, p1 = one, p2 = one, p3 = one;
    unsigned int i = 0;
    const unsigned int n16 = length & ~15u, n4 = length & ~3u;

    for (; i < n16; i += 16) {
        __m128 a = _mm_loadu_ps(x + i);
        __m128 b = _mm_loadu_ps(x + i + 4);
        __m128 c = _mm_loadu_ps(x + i + 8);
        __m128 d = _mm_loadu_ps(x + i + 12);
        s0 = _mm_add_ps(s0, _mm_sqrt_ps(a));
        s1 = _mm_add_ps(s1, _mm_sqrt_ps(b));
        s2 = _mm_add_ps(s2, _mm_sqrt_ps(c));
        s3 = _mm_add_ps(s3, _mm_sqrt_ps(d));
        p0 = _mm_mul_ps(p0, _mm_add_ps(_mm_mul_ps(a, a), one));
        p1 = _mm_mul_ps(p1, _mm_add_ps(_mm_mul_ps(b, b), one));
        p2 = _mm_mul_ps(p2, _mm_add_ps(_mm_mul_ps(c, c), one));
        p3 = _mm_mul_ps(p3, _mm_add_ps(_mm_mul_ps(d, d), one));
    }

      /* remaining full vectors */
    for (; i < n4; i += 4) {                          
        __m128 a = _mm_loadu_ps(x + i);
        s0 = _mm_add_ps(s0, _mm_sqrt_ps(a));
        p0 = _mm_mul_ps(p0, _mm_add_ps(_mm_mul_ps(a, a), one));
    }

    __m128 s = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
    __m128 p = _mm_mul_ps(_mm_mul_ps(p0, p1), _mm_mul_ps(p2, p3));

    /* 0-3 leftover elements, lane 0 only */
    for (; i < length; i++) {                           
        __m128 a = _mm_load_ss(x + i);
        s = _mm_add_ss(s, _mm_sqrt_ss(a));
        p = _mm_mul_ss(p, _mm_add_ss(_mm_mul_ss(a, a), one));
    }

    /* Horizontal reductions: fold high half onto low, then odd lane onto even. */
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    p = _mm_mul_ps(p, _mm_movehl_ps(p, p));
    p = _mm_mul_ss(p, _mm_movehdup_ps(p));

    __m128 r = _mm_set_ss(cbrtf(_mm_cvtss_f32(s)));
    r = _mm_sqrt_ss(_mm_add_ss(one, _mm_div_ss(r, p)));
    return _mm_cvtss_f32(r);
}