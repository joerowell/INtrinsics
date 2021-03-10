#ifndef INCLUDED_INTRINSICS_H
#define INCLUDED_INTRINSICS_H
#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

/***
 * Intrinsics. This header file provides a collection of hand-written versions
 * of some x86-64 intrinsics. The reason for this is that intrinsics tend to
 * need a particular instruction set (e.g AVX), but these are not always
 * available. This can be a problem if you're targeting a different instruction
 * set, or something that isn't an x86-64 platform.
 *
 * Therefore, this file defines some functions that implement the same
 * functionality as some x86_64 intrinsics, but in plain C++. For each function,
 * we tested if gcc 10.2 generated good object code [where appropriate, we
 * include a Godbolt link] -- if it does not, we provide manual calls to the
 * intrinsics themselves. These are only resolved if the relevant macro (e.g
 * __AVX2__) are present at compile-time: this means that the intrinsics will be
 * used if available, but in all other cases we fall back to the manual C++
 * implementation. Note: this header set does absolutely no checking to see if
 * the runtime platform supports these intrinsics. This may be added later if
 * necessary.
 *
 * The general naming convention is the following:
 * m<size of type>_<name of operation>_<size of elements>.
 *
 * For example, if you were to write a function that permuted 16 16-bit integers
 * in a 256-bit vector, it would be named m256_permute16x16_epi16.
 *
 * This tightly matches the Intel naming scheme, but it also prevents us from
 * having a code-base that's littered with the already reserved underscores.
 * Every function in this class accepts the ArrType union as its arguments. This type is meant to be
 * used as a near 1:1 correspondence with 256-bit __m256i type. To make this type ever more useful,
 * we provide many different members for access.
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

/**
 * It's sometimes very useful to have named masks and bit patterns.
 * Rather than have these as inline binary literals, we provide these with
 * friendly names.
 * Note: this is a struct to prevent it from being extended elsewhere.
 * This struct is by design at the top of the namespace, because future
 * methods may not have a chance to see them.
 *
 */

// To prevent complaints about overflow, we denote each of these
// as uint8_t.  This isn't wrong: the CPU will operate on these
// in 32-bit registers anyway.
// Note that, in practice, the distinction doesn't matter; these are
// just bits and bytes after-all. However, the compiler will warn, and we
// don't want that.

// Individual bit masks
static uint8_t int8_zero_bit_mask    = 0b00000001;
static uint8_t int8_first_bit_mask   = 0b00000010;
static uint8_t int8_second_bit_mask  = 0b00000100;
static uint8_t int8_third_bit_mask   = 0b00001000;
static uint8_t int8_fourth_bit_mask  = 0b00010000;
static uint8_t int8_fifth_bit_mask   = 0b00100000;
static uint8_t int8_sixth_bit_mask   = 0b01000000;
static uint8_t int8_seventh_bit_mask = 0b10000000;

// Individual pair masks
static uint8_t int8_zero_pair_mask   = 0b00000011;
static uint8_t int8_first_pair_mask  = 0b00001100;
static uint8_t int8_second_pair_mask = 0b00110000;
static uint8_t int8_third_pair_mask  = 0b11000000;

// Quad masks.
static uint8_t int8_zero_quad_mask  = 0b00001111;
static uint8_t int8_first_quad_mask = 0b11110000;

static uint8_t int8_zero_bit_shift    = 0;
static uint8_t int8_first_bit_shift   = 1;
static uint8_t int8_second_bit_shift  = 2;
static uint8_t int8_third_bit_shift   = 3;
static uint8_t int8_fourth_bit_shift  = 4;
static uint8_t int8_fifth_bit_shift   = 5;
static uint8_t int8_sixth_bit_shift   = 6;
static uint8_t int8_seventh_bit_shift = 0;

static uint8_t int8_zero_pair_shift  = 0;
static uint8_t int8_first_pair_shift = 2;
static uint8_t int8_second_pair_shift = 4;
static uint8_t int8_third_pair_shift = 6;
static uint8_t int8_zero_quad_shift  = 0;
static uint8_t int8_first_quad_shift = 4;

/**
 * Forward declarations.
 * Note: each and all of these functions must be inline. 
 * The reason is a little-bit hacky, and relies upon an understanding of the linker and 
 * intrinsics. If you want to understand, read on -- otherwise, feel free to skip this.
 *
 * With that warning: here are where things get crazy.
 * Some x86 instructions, rather than polluting the instruction cache, actually encode 
 * integer arguments in their instruction codes: in Intel parlance, these arguments are called immediates. 
 * This sounds insane, but it's actually very sensible -- this is no different to writing code like this:
 * void add_1(const int a) {return a + 1;}
 * void add_2(const int a) {return a + 2;} 
 * ....
 *
 * And so on and so forth. If you want to guarantee statically what a particular operation will do, enforcing this is not a bad idea.
 * However, it really stumps us when it comes to writing code. We don't want to have to write out all of the variants of the instructions with fixed elements -- we 
 * want to write a singular function that we can call with the immediate as a parameter -- provided we know it at compile-time.
 *
 * Here's the problem, though: when the compiler encounters the definition of a particular function, it has no way of knowing whether the parameter is known at compile-time or not: it can
 * only see the local definition, not the usage. This is a big problem, and it prevents us from writing code that's neat.
 * The C++ programmers amongst you will think "ah, a template". This would be a good place to use a template -- but unfortunately, there's no such system in C.
 * However, if we think about why a template works here, it seems to be exactly what we want -- the compiler only instantiates a template when it sees a definition (or a call) that supplies a 
 * compile-time as the template argument. How can we mimic this without templates?
 *
 * The answer is simple: tell the compiler 'hey, only instantiate this at the call site'. By doing this, we let the compiler 'see' the parameters that have been passed in -- it will then know whether
 * the parameter is a compile-time constant or not, and we'll be on our merry way. While 'inline' is just a hint, it also tells the linker when to do its work (i.e one definition per translation unit)
 * which almost mimics what we want.
 */

inline ArrType m256_shuffle_epi8(const ArrType *const a, const ArrType *const b);
inline ArrType m256_permute4x64_epi64(const ArrType * const a, const uint8_t imm8);
/**
 * e_sign. Implements the signum function in a branchless fashion on the input value,
 * value.
 *
 * In particular, this function returns:
 * 1 if value > 0
 * 0 if value == 0
 * -1 otherwise.
 *
 *  This doesn't actually correspond to an Intel Intrinsic, but it's useful in implementing
 *  other functionality later.
 */
inline int16_t e_sign(const int16_t value);
inline ArrType m256_hadd_epi16(const ArrType *const a, const ArrType *b);

// Actual implementations follow.

inline ArrType m256_shuffle_epi8(const ArrType *const a, const ArrType *const b)
{
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
    c.v8[i]            = (int8_t)(1 ^ ((1u << flag) - 1)) * a->v8[pos];

    const unsigned flag2 = (b->v8[i + 16] & 0x80) >> 7;
    assert(flag2 == 0 || flag2 == 1);

    const unsigned pos2 = b->v8[i + 16] & 0x0F;
    c.v8[i + 16]        = (int8_t)(1 ^ ((1u << flag2) - 1)) * a->v8[pos2 + 16];
  }
#endif
  return c;
}

inline int16_t e_sign(const int16_t value)
{
  // The cast to int16_t is vital: if you don't cast here, it'll do
  // int32_t comparison, which in turn means you'll get results you
  // didn't intend for.
  return ((int16_t)0 < value) - (value < (int16_t)0);
}

inline ArrType m256_hadd_epi16(const ArrType *const a, const ArrType *b)
{
  ArrType c;
  // Note; this is compile-time dispatch.
  // This function also makes use of reinterpret cast: the C++ standard dictates that
  // these should not generate extra machine instructions, so there's no overhead here.
#if defined(__AVX2__)
  c.m256 = _mm256_hadd_epi16(a->m256, b->m256);
#elif defined(__SSE3__)
  c.m128[0] = _mm_hadd_epi16(a->m128[0], b->m128[0]);
  c.m128[1] = _mm_hadd_epi16(a->m128[1], b->m128[1]);
#else
  // A clever compiler will unroll this into two separate batches of mov instructions:
  // this just makes it easier to check the semantics.
  for (unsigned int i = 0; i < 16; i += 8)
  {
    c.v16[i + 0] = a->v16[i + 0] + a->v16[i + 1];
    c.v16[i + 1] = a->v16[i + 2] + a->v16[i + 3];
    c.v16[i + 2] = a->v16[i + 4] + a->v16[i + 5];
    c.v16[i + 3] = a->v16[i + 6] + a->v16[i + 7];
    c.v16[i + 4] = b->v16[i + 0] + b->v16[i + 1];
    c.v16[i + 5] = b->v16[i + 2] + b->v16[i + 3];
    c.v16[i + 6] = b->v16[i + 4] + b->v16[i + 5];
    c.v16[i + 7] = b->v16[i + 6] + b->v16[i + 7];
  }
#endif
  return c;
}

inline ArrType m256_permute4x64_epi64(const ArrType * const a, const uint8_t imm8) {
    ArrType b;
#ifdef __AVX2__
    b.m256 = _mm256_permute4x64_epi64(a->m256, imm8);
#else
    // This function works as follows: grabs the index from each of the bytes of
    // imm8 We isolate these bytes by bitwise ops, and then shift if necessary
    // Note; these are constexpr variables, which means that these masks are
    // computed at compile-time. As a result, this is just a series of mov
    // instructions, which occur at a rate of approximately 4 per clock: a
    // clever compiler will interleave these movs to hide the latency.
     uint8_t zero   = (imm8 & int8_zero_pair_mask) >> int8_zero_pair_shift;
     uint8_t first  = (imm8 & int8_first_pair_mask) >> int8_first_pair_shift;
     uint8_t second = (imm8 & int8_second_pair_mask) >> int8_second_pair_shift;
    // This doesn't require a shift because the bytes are already in the
    // bottom-most byte.
     uint8_t third = (imm8 & int8_third_pair_mask) >> int8_third_pair_shift;
    // Finally, we do the permutation and return.
    b.v64[0] = a->v64[zero];
    b.v64[1] = a->v64[first];
    b.v64[2] = a->v64[second];
    b.v64[3] = a->v64[third];
#endif
    return b;
  }
#endif
