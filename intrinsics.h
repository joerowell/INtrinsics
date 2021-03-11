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
 * functionality as some x86_64 intrinsics, but in plain C. For each function,
 * we tested if gcc 10.2 generated good object code [where appropriate, we
 * include a Godbolt link] -- if it does not, we provide manual calls to the
 * intrinsics themselves. These are only resolved if the relevant macro (e.g
 * __AVX2__) are present at compile-time: this means that the intrinsics will be
 * used if available, but in all other cases we fall back to the manual C
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
const static uint8_t int8_zero_bit_mask    = 0b00000001;
const static uint8_t int8_first_bit_mask   = 0b00000010;
const static uint8_t int8_second_bit_mask  = 0b00000100;
const static uint8_t int8_third_bit_mask   = 0b00001000;
const static uint8_t int8_fourth_bit_mask  = 0b00010000;
const static uint8_t int8_fifth_bit_mask   = 0b00100000;
const static uint8_t int8_sixth_bit_mask   = 0b01000000;
const static uint8_t int8_seventh_bit_mask = 0b10000000;

// Individual pair masks
const static uint8_t int8_zero_pair_mask   = 0b00000011;
const static uint8_t int8_first_pair_mask  = 0b00001100;
const static uint8_t int8_second_pair_mask = 0b00110000;
const static uint8_t int8_third_pair_mask  = 0b11000000;

// Quad masks.
const static uint8_t int8_zero_quad_mask  = 0b00001111;
const static uint8_t int8_first_quad_mask = 0b11110000;

const static uint8_t int8_zero_bit_shift    = 0;
const static uint8_t int8_first_bit_shift   = 1;
const static uint8_t int8_second_bit_shift  = 2;
const static uint8_t int8_third_bit_shift   = 3;
const static uint8_t int8_fourth_bit_shift  = 4;
const static uint8_t int8_fifth_bit_shift   = 5;
const static uint8_t int8_sixth_bit_shift   = 6;
const static uint8_t int8_seventh_bit_shift = 0;

const static uint8_t int8_zero_pair_shift   = 0;
const static uint8_t int8_first_pair_shift  = 2;
const static uint8_t int8_second_pair_shift = 4;
const static uint8_t int8_third_pair_shift  = 6;
const static uint8_t int8_zero_quad_shift   = 0;
const static uint8_t int8_first_quad_shift  = 4;

/**
 * Forward declarations.
 * Note: each and all of these functions must be inline.
 * The reason is a little-bit hacky, and relies upon an understanding of the linker and
 * intrinsics. If you want to understand, read on -- otherwise, feel free to skip this.
 *
 * With that warning: here are where things get crazy.
 * Some x86 instructions, rather than polluting the instruction cache, actually encode
 * integer arguments in their instruction codes: in Intel parlance, these arguments are called
 * immediates. This sounds insane, but it's actually very sensible -- this is no different to
 * writing code like this: void add_1(const int a) {return a + 1;} void add_2(const int a) {return a
 * + 2;}
 * ....
 *
 * And so on and so forth. If you want to guarantee statically what a particular operation will do,
 * enforcing this is not a bad idea. However, it really stumps us when it comes to writing code. We
 * don't want to have to write out all of the variants of the instructions with fixed elements -- we
 * want to write a singular function that we can call with the immediate as a parameter -- provided
 * we know it at compile-time.
 *
 * Here's the problem, though: when the compiler encounters the definition of a particular function,
 * it has no way of knowing whether the parameter is known at compile-time or not: it can only see
 * the local definition, not the usage. This is a big problem, and it prevents us from writing code
 * that's neat. The C++ programmers amongst you will think "ah, a template". This would be a good
 * place to use a template -- but unfortunately, there's no such system in C. However, if we think
 * about why a template works here, it seems to be exactly what we want -- the compiler only
 * instantiates a template when it sees a definition (or a call) that supplies a compile-time as the
 * template argument. How can we mimic this without templates?
 *
 * The answer is simple: tell the compiler 'hey, only instantiate this at the call site'. By doing
 * this, we let the compiler 'see' the parameters that have been passed in -- it will then know
 * whether the parameter is a compile-time constant or not, and we'll be on our merry way. While
 * 'inline' is just a hint, it also tells the linker when to do its work (i.e one definition per
 * translation unit) which almost mimics what we want.
 */

inline ArrType m256_shuffle_epi8(const ArrType *const a, const ArrType *const b);
inline ArrType m256_permute4x64_epi64(const ArrType *const a, const uint8_t imm8);
inline ArrType m256_srli_epi16(const ArrType *const a, const uint8_t imm8);
inline ArrType m256_slli_epi16(const ArrType *const a, const uint8_t imm8);
inline ArrType m256_abs_epi16(const ArrType *const a);

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

inline ArrType m256_permute4x64_epi64(const ArrType *const a, const uint8_t imm8)
{
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
  b.v64[0]          = a->v64[zero];
  b.v64[1]          = a->v64[first];
  b.v64[2]          = a->v64[second];
  b.v64[3]          = a->v64[third];
#endif
  return b;
}

/**
 * m256_srli_epi16. Given an array reference, a, we shift each int16_t in a by to the right by
 * imm8 many positions. If imm8 > 16, we replace by zeros. This corresponds exactly to the
 * _mm256_slli_epi16 intrinsic.
 * GCC 10.2 compiles this exactly to the right intrinsic, so there's no need for any clever
 * tricks.
 */
inline ArrType m256_srli_epi16(const ArrType *const a, const uint8_t imm8)
{
  ArrType b;
  // Right-shift is not well-defined for signed values.
  //
  // In particular,
  // the C and C++ standards take the view that shifting a value should not change its sign.
  // This means that here we need to explicitly tell the compiler that we want to shift in 0s
  // by throwing away the 'signedness' for the shift operation. If we don't do this, then the
  // compiler *will* generate vectorised code, but the semantics of the operation are completely
  // different (i.e the leading bits now depend on the sign).
  // This is the sort of bug that would be nigh-on impossible to find, so this
  // comment is meant to draw your attention to it.
  for (unsigned int i = 0; i < 16; i++)
  {
    b.v16[i] = (int16_t)((uint16_t)(a->v16[i]) >> imm8);
  }
  return b;
}

/**
 * m256_slli_epi16. Given an array reference, a, we shift each int16_t in a by to the left by
 * imm8 many positions. If imm8 > 16, we replace by zeros. This corresponds exactly to the
 * _mm256_slli_epi16 intrinsic.
 * GCC 10.2 compiles this exactly to the right intrinsic, so there's no need for any clever
 * tricks.
 */
inline ArrType m256_slli_epi16(const ArrType *const a, const uint8_t imm8)
{
  ArrType b;
  // Because left-shift is well-defined, the compiler will actually just implement
  // this as either two 128-bit shifts or one 256-bit shift. In other words, this is
  // an easy thing for the compiler to optimise.
  // Note; this will cause constants to exist in your instruction cache. This is because
  // imm8 needs to be an immediate for this to compile properly. If you do not know why this
  // matters, don't worry. (short answer: bigger entry -- fewer entries -- not great). But, this
  // is unfixable.
  for (unsigned int i = 0; i < 16; i++)
  {
    b.v16[i] = a->v16[i] << imm8;
  }

  return b;
}

/**
 * m256_abs_epi16. This function accepts an array reference, a, and applies the abs function
 * to each 16-bit integer in a, returning the result (denoted as b).
 *
 * This exactly mimics the _mm256_abs_epi16 intrinsic.
 * In particular, for each i = (0, 15) we have that b[i] = abs(a[i]) upon return.
 * Since GCC 10.2 appears to have issues generating the right object code
 * (see https://godbolt.org/z/7YnKfo), we provide AVX2 and SSE3 intrinsic delegation.
 * a) If __AVX2__ is defined, we use the _mm256_abs_epi16 intrinsics.
 * b) If not, and if __SSSE4__ is defined, we use the _mm_abs_epi16 over each lane of a.
 * c) If not, we use the hand-rolled version as explained below.
 *
 */
inline ArrType m256_abs_epi16(const ArrType *const a)
{
  ArrType b;

#ifdef __AVX2__
  b.m256 = _mm256_abs_epi16(a->m256);
#elif defined(__SSE3__)
  b.m128[0]         = _mm_abs_epi16(a->m128[0]);
  b.m128[1]         = _mm_abs_epi16(a->m128[1]);
#else
  // This is a branchless implementation of the ABS function.
  // The idea is taken from the following excellent blog post:
  // https://hbfs.wordpress.com/2008/08/05/branchless-equivalents-of-simple-functions/
  // The idea is that you can generate a mask that represents the sign-bit of the function
  // by using the behaviour of the shift. If you want to be really clever (as we do here), you can
  // type pun with a union. In C++ this would be disallowed, but C is more flexible.
  union pun
  {
    // let us suppose long is twice as wide as int
    long w;
    // should be hi,lo on a big endian machine
    struct
    {
      int lo, hi;
    };
  };

  // The point is this: if a[i] is positive, then the leading bit is 0: so the shift
  // generates an all zero mask. This means the second line does nothing.
  // By contrast, if a[i] is negative, then the shift generates the all 1 mask.
  // When xoring against this, we will get the two's complement of a[i]: subtracting
  // sign-extend in this context implicitly converts it to 1, which is exactly how you
  // convert between two's complement numbers. Cool, eh?
  // BTW: this is branchless because the if would have maximum entropy (i.e it's a 50/50 chance
  // if your number is positive or negative -- not fun.)
  for (unsigned int i = 0; i < 16; i++)
  {
    const union pun t           = {.w = a->v16[i]};
    const int16_t signed_extend = t.hi;
    b.v16[i]                    = (a->v16[i] ^ signed_extend) - signed_extend;
  }
#endif
  return b;
}

/**
 * get_randomness. This function is not SIMD in nature: instead, given two sources of randomness,
 * gstate_1 and g_state2, it produces a 128-bit random value.
 *
 * In it's naive form, this is based on Lehmer's generator, which is (arguably) the simplest
 * random number generator that can pass the Big Crush tests. It's also remarkably simple
 * to write -- see e.g
 * https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/
 *
 * However, in some situations it can be a little bit slow compared to the available alternatives.
 * As a result -- and where applicable -- we delegate to the aes_enc function, which is really
 * fast, to produce randomness. This trick was first brought to my attention by working on
 * https://github.com/lducas/AVX2-BDGL-bucketer.
 *
 * WARNING WARNING WARNING: this should _not_ be used for any situation
 * where you need true randomness. It may be fast -- it may even appear reasonable --
 * but please, don't use this for anything that needs anything sensible.
 * Here we *just* use it for speed: it's fast and small, but beyond that there's no reason
 * to use it. You have been warned.
 */

inline ArrType get_randomness(ArrType *const gstate_1, ArrType *const gstate_2) noexcept
{
  ArrType out;
#ifdef __AES_NI__
  out.m128[0] = _mm_aesenc_si128(gstate_1->m128[0], gstate_1->m128[0]);
#else
  gstate_1->m128[0] = (__m128i)((__uint128_t)gstate_1->m128[0] * 0xda942042e4dd85b5);
  gstate_2->m128[0] *= (__m128i)((__uint128_t)gstate_2->m128[0] * 0xda942042e4dd85b5);
  out.v64[0] = gstate_1->v64[1];
  out.v64[1] = gstate_2->v64[0];
#endif
  return out;
}

/***
 *
 * m256_xor_epi64. This function accepts two array references, a and b,
 * and returns the xor of a and b in a third array, denoted as c.
 * This function exactly mimics the behaviour of the _mm256_xor_si256 function.
 * In particular, after
 * this function is called, c has the following layout: for all i in {0, 1, 2,
 * 3}, c[i] = a[i] ^ b[i].
 * Note that we allow a == b to be the same pair:
 * this is because xoring two vectors is a common use
 * case. As gcc 10.2 has no trouble producing vectorised object code for this
 * function, we do not explicitly delegate to the intrinsics.
 */
inline ArrType m256_xor_epi64(const ArrType *const a, const ArrType *const b)
{
  ArrType c;
  // Simply xor them together!
  for (unsigned i = 0; i < 4; i++)
  {
    c.v64[i] = a->v64[i] ^ b->v64[i];
  }
  return c;
}

/***
 * m256_or_epi64.
 * This function accepts two array references, a and b,
 * and returns the bitwise OR of a and b, denoted as c.
 * This exactly mimics the behaviour of the _mm256_or_si256 function.
 *
 * In particular, after this function is called, c has the following layout: For
 * all i in {0, 1, 2, 3}, c[i] = a[i] | b[i]
 * This function will only work if a != b.
 * We do not allow (a, b) to be the same pair: this is because a | a = a.
 *
 * As gcc 10.2 has no trouble producing vectorised object code for this function, we do not
 * explicitly delegate to the intrinsics.
 */
inline ArrType m256_or_epi64(const ArrType *const a, const ArrType *const b)
{
  ArrType c;
  // Simply OR them together!
  for (unsigned i = 0; i < 4; i++)
  {
    c.v64[i] = a->v64[i] | b->v64[i];
  }
  return c;
}

/***
 * m256_and_epi64.
 * This function accepts two array references, a and b,
 * and returns the bitwise AND of a and b, denoted as c.
 * This exactly mimics the behaviour of the _mm256_and_si256 function.
 *
 * In particular, after this function is called, c has the following layout: For
 * all i in {0, 1, 2, 3}, c[i] = a[i] & b[i]
 * This function will only work if a != b.
 * We do not allow (a, b) to be the same pair: this is because a & a = a.
 *
 * As gcc 10.2 has no trouble producing vectorised object code for this function, we do not
 * explicitly delegate to the intrinsics.
 */
inline ArrType m256_and_epi64(const ArrType *const a, const ArrType *const b) noexcept
{
  // Function pre-condition: we check that a != b because that would be the equivalent of a
  // no-op.
  assert(a != b);

  ArrType c;
  // Simply AND them together!
  for (unsigned i = 0; i < 4; i++)
  {
    c.v64[i] = a->v64[i] & b->v64[i];
  }
  return c;
}

/***
 * m256_cmpgt_epi16. This function accepts two array references, a and b, and returns
 * the comparison mask between a and b, denoted as c.
 *
 * This exactly mimics the behaviour of the _mm256_cmpgt_epi16
 * function. In particular, after this function is called, c has the following
 * layout: For all i = 0, ..., 15:
 *
 * c[i] = 0xFFFF if a[i] > b[i]
 * c[i] = 0 otherwise
 *
 * This function will only work if a and b are not equal.
 * As GCC 10.2 has trouble producing vectorised object code (see https://godbolt.org/z/W9nqq7)
 * for this function,
 * where possible we delegate to the relevant
 * intrinsics. In particular, this function:
 *
 * a) If __AVX2__ is defined, it uses the AVX256 function _mm256_cmpgt_epi16.
 * b) If __AVX2__ is not defined, but __SSE2__ is, it uses the _mm_cmpgt_epi16
 * function twice.
 * c) Otherwise, use the hand-written variant. This does generate vectorised
 * code, but it isn't quite one instruction.
 */
inline ArrType m256_cmpgt_epi16(const ArrType *const a, const ArrType *const b)
{
  // Note: this isn't strictly necessary, but it'd be faster just
  // to zero-out the whole array and so we disallow it.
  // If you want to zero out the whole array, then you can use m256_xor_epi64
  // or similar.
  assert(a != b);
  ArrType c;
#ifdef __AVX2__
  c.m256 = _mm256_cmpgt_epi16(a->m256, b->m256);
#elif defined(__SSE2__)
  c.m128[0]  = _mm_cmpgt_epi16(a->m128[0], b->m128[0]);
  c.m128[1]  = _mm_cmpgt_epi16(a->m128[1], b->m128[1]);
#else
  // A sensible compiler will unroll this and produce somewhat good vectorised
  // code, but not quite optimal code (as of GCC 10.2).
  for (unsigned int i = 0; i < 16; i++)
  {
    // This works as follows:
    // a[i] > b[i] evaluates to 0 or 1.
    // If 0, then c[i] = 0.
    // If 1, then c[i] = 0xFFFF,
    // which matches the semantics of cmpgt_epi16 exactly.
    c.v16[i] = (a->v16[i] > b->v16[i]) * 0xFFFF;
  }
#endif
  return c;
}

  /***
    * m256_testz_si256.
    */
   inline bool m256_testz_si256(const ArrType * const a, const ArrType * const b)
   {
 #ifdef __AVX2__
     return _mm256_testz_si256(a->m256, b->m256);
 #else
     const auto c = m256_and_epi64(a, b);
     // Sum and pop-cnt all of the elements in c.
     // Note: builtin_popcountl will compile to a CPU instruction iff SSE4.2 or later
     // is available, but if not the compiler has its own dedicated software routines
     // for this.
     int total0, total1, total2, total3;
       total0 = __builtin_popcountl((uint64_t)c.v64[0]);
       total1 = __builtin_popcountl((uint64_t)c.v64[1]);
       total2 = __builtin_popcountl((uint64_t)c.v64[2]);
       total3 = __builtin_popcountl((uint64_t)c.v64[3]);
 
     return (total0 + total1 + total2 + total3) == 0;
 #endif
   }

   /***
   * m256_add_epi16. This function accepts 2 array references, a and b, and pairwises sums a and b,
   * returning the result, denoted as c. This function exactly mimics the _mm256_add_epi16 function.
   * In particular, after this function is called, c
   * has the following layout: For all i = 0, ..., 15: c[i] = a[i] + b[i]
   *
   * This function appears to be trivially vectorisable on GCC 10.2, and as a
   * result we do not introduce other intrinsics into this function.
   */
  inline ArrType m256_add_epi16(const ArrType * const a,
                                const ArrType * const b)
  {
    ArrType c;
    // Note; this very trivial for-loop should be trivial for the compiler to
    // optimise, especially if it knows the size of the arrays ahead of time
    // (which it does!)
    for (unsigned int i = 0; i < 16; i++)
    {
      c.v16[i] = a->v16[i] + b->v16[i];
    }
    return c;
  }


   /***
    * m256_sub_epi16. This function accepts 2 array references, a and b, and pairwises sums a and b,
    * returning the result, denoted as c. This function exactly mimics the _mm256_add_epi16 function.
    * In particular, after this function is called, c
    * has the following layout: For all i = 0, ..., 15: c[i] = a[i] - b[i]
    *
    * This function appears to be trivially vectorisable on GCC 10.2, and as a
    * result we do not introduce other intrinsics into this function.
    */
   inline ArrType m256_sub_epi16(const ArrType * const a, const ArrType * const b) 
   {
     ArrType c;
     // Note; this very trivial for-loop should be trivial for the compiler to
     // optimise, especially if it knows the size of the arrays ahead of time
     // (which it does!)
     for (unsigned int i = 0; i < 16; i++)
     {
       c.v16[i] = a->v16[i] - b->v16[i];
     }
 
     return c;
   }
 
   /**
    * s256_sign_epi16.
    * Applies a negation to the packed signed 16-bit integers in a according to
    * the elements in b and returns the result in an array c.
    *
    * More particularly: for any i in [0 ... 15], after this function is called c
    * contains the following:
    *
    * c[i] = a[i]  if b[i] > 0
    * c[i] = 0     if b[i] == 0
    * c[i] = -a[i] otherwise
    *
    * You can view this as a signed inclusion.
    *
    * Because GCC 10.2 seems to struggle with producing the right object code (see
    * https://godbolt.org/z/3nasvK), we provide an AVX2 overload if it is defined by the compiler. In
    * particular: a) if __AVX2__ is defined, we use the _mm256_sign_epi16
    * intrinsic. b) else if __SSE3__ is defined, we use the _mm_sign_epi16
    * intrinsic over each half of a and b.
    * c) else, we use use our hand-written
    * version.
    *
    * Note: our hand-written version is not bad. Rather cleverly, the compiler will vectorise the
    * call to e_sign and apply it across all of the masks and then applies a multiplication. However,
    * it's not quite as short as the _mm256_sign_epi16 version.
    */
   inline ArrType m256_sign_epi16(const ArrType * const a, const ArrType * const b)
   {
     // pre-conditions for the function to work.
     assert(a != b);
     ArrType c;
 #ifdef __AVX2__
     c.m256 = _mm256_sign_epi16(a->m256, b->m256);
 #elif defined(__SSE3__)
     c.m128[0] = _mm_sign_epi16(a->m128[0], b->m128[0]);
     c.m128[1] = _mm_sign_epi16(a->m128[1], b->m128[1]);
 #else
     // This function is rather simple: we extract the signs of each b[i] and
     // multiply a[i] by them.
     for (unsigned int i = 0; i < 16; i++)
     {
       c.v16[i] = a->v16[i] * e_sign(b->v16[i]);
     }
 #endif
     return c;
   }

 /**
   * m256_broadcastsi128_si256. Given a uint128_t `value` as input, this function returns an array,
   * `a`, where a[0:7] = value and a[8:15] = value. This is exactly the same semantics as the
   * __mm256_broadcastsi128_si256 intrinsic.
   *
   * GCC 10.2 has trouble producing vectorised code for this function: see e.g
   * https://godbolt.org/z/covYEd. As a result, similarly to elsewhere in this file, we delegate to
   * the appropriate intrinsics where available: a) If __AVX2__ is defined, we use the
   * _mm256_broadcastsi128_si256 intrinsic. b) If __AVX2__ is not defined, but __SSE2__ is, we use
   * two _mm_store_si128 instructions over the upper and lower halves of a. c) Otherwise we use our
   * hand-rolled version.
   */
  inline ArrType m256_broadcastsi128_si256(const ArrType * const value)
  {
    ArrType a;
#ifdef __AVX2__
    a.m256 = _mm256_broadcastsi128_si256(value->m128[0]);
#elif defined(__SSE2__)
    a.m128[0] = value->m128[0];
    a.m128[1] = value->m128[0];
#else
    // the simplest way to do this is just to use an memcpy.
    // GCC compiles this to 4 mov instructions, which is exactly the behaviour we want.
    memcpy(&a.v16[0], value, sizeof(*value));
    memcpy(&a.v16[8], value, sizeof(*value));
#endif
    return a;
  }

  /**
   * mm_extract_epi64. Given a uint128_t `value` as input and a template parameter `pos`, this
   * function returns the 64-bit in value[pos*64:(pos*64) + 63]. This compiles down into a
   * a series of shifts and xors, which is relatively fast.
   *
   * GCC 10.2 generates decent code (see lines 8 and 9 of https://godbolt.org/z/9nEcEe), but it
   * isn't quite the vectorised instructions we'd like. As a result, we delegate to the pextract
   * intrinsic if it's available. This intrinsic is available if SSE4.1 is available.
   */
  inline int64_t mm_extract_epi64(const ArrType * const arr, const uint8_t imm8)
  {
#ifdef __AVX__  // SSE4.1 doesn't have a supported macro: the lowest other macro is AVX.
    return _mm_extract_epi64(arr->m128[0], imm8);
#else
    return arr->v64[imm8];
#endif
  }

#endif
