/* 211872577 Gal Shoshan */

#include <emmintrin.h>   // SSE2
#include <smmintrin.h>   // SSE4.1
#include <string.h>

#define MAX_STR 256

int hamming_dist(char str1[MAX_STR], char str2[MAX_STR]) {
    /* Distance = len(longer) - (#equal positions before the shorter string ends):
       every position past the shorter string's end is a mismatch.
       All loads are 16 bytes at offsets 0, 16, ..., 240, so they stay inside the buffers. */
       const __m128i zero = _mm_setzero_si128();
       const __m128i lane = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
       __m128i eq_cnt = zero;              /* per-lane count of equal bytes (at most 16 each) */
       int len = MAX_STR;                  /* length of the longer string */
    
       for (int i = 0; i < MAX_STR; i += 16) {
           __m128i a  = _mm_loadu_si128((const __m128i *)(str1 + i));
           __m128i b  = _mm_loadu_si128((const __m128i *)(str2 + i));
           __m128i eq = _mm_cmpeq_epi8(a, b);
           /* min(a, b) == 0 exactly where either string has its NUL */
           int nul = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(a, b), zero));
    
           if (nul) {
               int k = __builtin_ctz(nul);                 /* shorter string ends at i + k */
               unsigned bit = 1u << k;
               /* count equal bytes only in lanes < k */
               eq_cnt = _mm_sub_epi8(eq_cnt,
                        _mm_and_si128(eq, _mm_cmpgt_epi8(_mm_set1_epi8((char)k), lane)));
    
               unsigned ma = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(a, zero));
               unsigned mb = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(b, zero));
               if (ma & mb & bit) {                        /* equal lengths */
                   len = i + k;
                   break;
               }
               /* Scan the longer string for its NUL, starting just after position k. */
               const char *s = (ma & bit) ? str2 : str1;
               unsigned m = ((ma & bit) ? mb : ma) & ~(2 * bit - 1);
               while (!m && (i += 16) < MAX_STR)
                   m = (unsigned)_mm_movemask_epi8(
                           _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(s + i)), zero));
               len = m ? i + __builtin_ctz(m) : MAX_STR;
               break;
           }
           eq_cnt = _mm_sub_epi8(eq_cnt, eq);              /* eq lanes are -1: subtracting adds 1 */
       }
    
       /* Horizontal sum of the 16 byte counters. */
       __m128i sum = _mm_sad_epu8(eq_cnt, zero);
       int equal = _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 2);
       return len - equal;
}
