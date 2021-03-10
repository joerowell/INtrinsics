#ifndef INCLUDED_INTRINSICS_H
#define INCLUDED_INTRINSICS_H
#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

/**
 * Intrinsics.h. This is the C version of the Intrinsics wrapper.
 * The point of this header (compared to the other one) is that it allows type punning
 * [explicitly in the standard], which means that the compiler should generate nicer code.
 * All of the
 */
typedef union ArrType
{
  int8_t v8[32];
  int16_t v16[16];
  int32_t v32[8];
  int64_t v64[4];
  __m128i m128[2];
  __m256i m256;

} ArrType;


ArrType m256_shuffle_epi8(const ArrType * const a, const ArrType * const b);

ArrType m256_shuffle_epi8(const ArrType * const a, const ArrType * const b) {
    ArrType c;
#ifdef __AVX2__
    c.m256 = _mm256_shuffle_epi8(a->m256, b->m256);
#elif defined(__SSE3__)
    c.m128[0] = _mm_shuffle_epi8(a->m128[0], b->m128[0]);
    c.m128[1] = _mm_shuffle_epi8(a->m128[1], b->m128[1]);
#else
    // This loop would be nicer written as two separate loops,
    // but we're hoping to build a pattern that a sensible compiler
    // can recognise as reasonable (i.e something that tightly matches to
    // the _mm256_shuffle_epi8 semantics)

    // This function has weird semantics, because the _mm256_shuffle_epi8 is
    // a single-lane shuffle. Essentially, it does not allow you to move across
    // 2 128-bit chunks at once: instead, the shuffle is localised to each
    // 128-bit vector. This means it is faster, but also weird.
    //
    // The algorithm at
    // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_shuffle_epi8&expand=5156
    // is somewhat descriptive: our variant is similar, but it's entirely
    // branchless with a sensible compiler. Originally we used the branchy
    // version like the original algorithm, but it generated very poor object
    // code. The idea is as follows: view b as an array of bytes of size 32. If
    // the leading bit of some byte b[i] is not set, we skip it (i.e we set the
    // index c[i] = 0). Otherwise, we set c[i] = a[pos], where pos consists of
    // the final 4 bits of b[i]. Note that this is only 4 bits, because we have
    // at most 16 options to choose from, and so we need exactly 4 bits. The
    // same is true for the second part: it's just shifted by 16 (i.e over the
    // other lane). You can write this as two one-liners if you'd prefer, but
    // this is somewhat neater. and the object code is approximately the same.

    // The (1^(...)) trick is a trick for negating the final bit of the number.
    // Here's how it works: if flag == 0 then the shift gives 1: 1-1  = 0, and so the
    // ^ gives 1. By contrast, if flag == 1 then the shift gives 2: 2-1 = 1, and so the ^ gives 0.
    // This can be generalised to n-many bits by changing the 1 to n.
    for (unsigned int i = 0; i < 16; i++)
    {
      const int16_t flag = (b->v8[i] & 0x80) >> 7;
      assert(flag == 0 || flag == 1);
      const unsigned pos = b->v8[i] & 0x0F;
      c.v8[i]               = (int8_t)(1 ^ ((1u << flag) - 1)) * a->v8[pos];

      const unsigned flag2 = (b->v8[i + 16] & 0x80) >> 7;
      assert(flag2 == 0 || flag2 == 1);

      const unsigned pos2 = b->v8[i + 16] & 0x0F;
      c.v8[i + 16]           = (int8_t)(1 ^ ((1u << flag2) - 1)) * a->v8[pos2 + 16];
    }
#endif
    return c;
}
#endif
