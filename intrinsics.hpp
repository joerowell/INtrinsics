#ifndef __INTRINSICS__
#define __INTRINSICS__
#include <array>
#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <iostream>

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
 * All of these functions live inside the CPP_INTRIN struct. This is a struct to
 * prevent arbitrary extensions at a future location -- it's easier to have them
 * all here.
 *
 * The general naming convention is the following:
 * m<size of type>_<name of operation>_<size of elements>.
 *
 * For example, if you were to write a function that permuted 16 16-bit integers
 * in a 256-bit vector, it would be named m256_permute16x16_epi16.
 *
 * This tightly matches the Intel naming scheme, but it also prevents us from
 * having a code-base that's littered with the already reserved underscores.
 */

struct CPP_INTRIN
{
  /**
   * It's sometimes very useful to have named masks and bit patterns.
   * Rather than have these as inline binary literals, we provide these with
   * friendly names.
   * Note: this is a struct to prevent it from being extended elsewhere.
   * This struct is by design at the top of the namespace, because future
   * methods may not have a chance to see them.
   *
   * The general naming convention is this:
   */
  struct BitPatterns
  {
    // To prevent complaints about overflow, we denote each of these
    // as uint8_t.  This isn't wrong: the CPU will operate on these
    // in 32-bit registers anyway.
    // Note that, in practice, the distinction doesn't matter; these are
    // just bits and bytes after-all. However, the compiler will warn, and we
    // don't want that.

    // Individual bit masks
    constexpr static uint8_t int8_zero_bit_mask    = 0b00000001;
    constexpr static uint8_t int8_first_bit_mask   = 0b00000010;
    constexpr static uint8_t int8_second_bit_mask  = 0b00000100;
    constexpr static uint8_t int8_third_bit_mask   = 0b00001000;
    constexpr static uint8_t int8_fourth_bit_mask  = 0b00010000;
    constexpr static uint8_t int8_fifth_bit_mask   = 0b00100000;
    constexpr static uint8_t int8_sixth_bit_mask   = 0b01000000;
    constexpr static uint8_t int8_seventh_bit_mask = 0b10000000;

    // Individual pair masks
    constexpr static uint8_t int8_zero_pair_mask   = 0b00000011;
    constexpr static uint8_t int8_first_pair_mask  = 0b00001100;
    constexpr static uint8_t int8_second_pair_mask = 0b00110000;
    constexpr static uint8_t int8_third_pair_mask  = 0b11000000;

    // Quad masks.
    constexpr static uint8_t int8_zero_quad_mask  = 0b00001111;
    constexpr static uint8_t int8_first_quad_mask = 0b11110000;

    constexpr static uint8_t int8_zero_bit_shift    = 0;
    constexpr static uint8_t int8_first_bit_shift   = 1;
    constexpr static uint8_t int8_second_bit_shift  = 2;
    constexpr static uint8_t int8_third_bit_shift   = 3;
    constexpr static uint8_t int8_fourth_bit_shift  = 4;
    constexpr static uint8_t int8_fifth_bit_shift   = 5;
    constexpr static uint8_t int8_sixth_bit_shift   = 6;
    constexpr static uint8_t int8_seventh_bit_shift = 0;

    constexpr static uint8_t int8_zero_pair_shift   = 0;
    constexpr static uint8_t int8_first_pair_shift  = 2;
    constexpr static uint8_t int8_second_pair_shift = 4;
    constexpr static uint8_t int8_third_pair_shift  = 6;

    constexpr static uint8_t int8_zero_quad_shift  = 0;
    constexpr static uint8_t int8_first_quad_shift = 4;
  };

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
  static inline constexpr int16_t e_sign(const int16_t value) noexcept
  {
    // The cast to int16_t is vital: if you don't cast here, it'll do
    // int32_t comparison, which in turn means you'll get results you
    // didn't intend for.
    return (int16_t(0) < value) - (value < int16_t(0));
  }

  /***
   * m256_hadd_epi16. This function accepts two array references, a and b
   * and returns the packed horizontal addition of
   * each 128-bit lane inside a and b, storing the result in a return variable c.

   * This exactly mimics the behaviour of the
   * _mm256_hadd_epi16 function. In particular, after this function is called, c
   * has the following layout: Let i = 0, 4 , 8, 12. For even multiples of i,
   * the following holds for any j in {0, 2, 4, 6} c[i+j] = a[i+j] + a[i+j+1];
   *
   * For odd multiples of i, the following holds for any j in {0, 2, 4, 6}
   * c[i+j] = b[i+j] + b[i+j+1]
   *
   * As GCC 10.2 has trouble producing vectorised
   * object code for this function (see https://godbolt.org/z/bWbWod),
   * where possible we delegate to the relevant
   * intrinsics. In particular, this function:
   *
   * a) If __AVX2__ is defined, it uses the AVX256 function _mm256_hadd_epi16.
   * b) If __AVX2__ is not defined, but __SSE3__ is, it uses the _mm_hadd_epi16
   * function twice. c) Otherwise, use the hand-written variant.
   *
   * This checking is done solely at compile-time.
   * TODO Does this make sense? Maybe we could afford one CPUID check elsewhere?
   */
  static inline std::array<int16_t, 16> m256_hadd_epi16(const std::array<int16_t, 16> &a,
                                                        const std::array<int16_t, 16> &b) noexcept
  {
    std::array<int16_t, 16> c;
    // Note; this is compile-time dispatch.
    // This function also makes use of reinterpret cast: the C++ standard dictates that
    // these should not generate extra machine instructions, so there's no overhead here.
#if defined(__AVX2__)
    _mm256_store_si256(reinterpret_cast<__m256i *>(&c),
                       _mm256_hadd_epi16(*reinterpret_cast<const __m256i *>(&a),
                                         *reinterpret_cast<const __m256i *>(&b)));
#elif defined(__SSE3__)
    // Split the input arrays into two, separate chunks.
    _mm_store_si128(reinterpret_cast<__m128i *>(&c),
                    _mm_hadd_epi16(*reinterpret_cast<const __m128i *>(&a),
                                   *reinterpret_cast<const __m128i *>(&b)));
    _mm_store_si128(reinterpret_cast<__m128i *>(&c[8]),
                    _mm_hadd_epi16(*reinterpret_cast<const __m128i *>(&a[8]),
                                   *reinterpret_cast<const __m128i *>(&b[8])));
#else
    // A clever compiler will unroll this into two separate batches of mov instructions:
    // this just makes it easier to check the semantics.
    for (unsigned int i = 0; i < 16; i += 8)
    {
      c[i + 0] = a[i + 0] + a[i + 1];
      c[i + 1] = a[i + 2] + a[i + 3];
      c[i + 2] = a[i + 4] + a[i + 5];
      c[i + 3] = a[i + 6] + a[i + 7];
      c[i + 4] = b[i + 0] + b[i + 1];
      c[i + 5] = b[i + 2] + b[i + 3];
      c[i + 6] = b[i + 4] + b[i + 5];
      c[i + 7] = b[i + 6] + b[i + 7];
    }

#endif
    return c;
  }

  /***
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
  static inline std::array<int64_t, 4> m256_xor_epi64(const std::array<int64_t, 4> &a,
                                                      const std::array<int64_t, 4> &b)
  {
    std::array<int64_t, 4> c;
    // Simply xor them together!
    for (unsigned i = 0; i < 4; i++)
    {
      c[i] = a[i] ^ b[i];
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
  static inline std::array<int64_t, 4> m256_or_epi64(const std::array<int64_t, 4> &a,
                                                     const std::array<int64_t, 4> &b) noexcept
  {
    // Function pre-condition: we check that a != b because that would be the equivalent
    // of a no-op
    assert(a != b);

    std::array<int64_t, 4> c;
    // Simply OR them together!
    for (unsigned i = 0; i < 4; i++)
    {
      c[i] = a[i] | b[i];
    }
    return c;
  }

  /***
   * m256_or_epi64.
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
  static inline std::array<int64_t, 4> m256_and_epi64(const std::array<int64_t, 4> &a,
                                                      const std::array<int64_t, 4> &b) noexcept
  {
    // Function pre-condition: we check that a != b because that would be the equivalent of a
    // no-op.
    assert(a != b);

    std::array<int64_t, 4> c;
    // Simply AND them together!
    for (unsigned i = 0; i < 4; i++)
    {
      c[i] = a[i] & b[i];
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
  static inline std::array<int16_t, 16> m256_cmpgt_epi16(const std::array<int16_t, 16> &a,
                                                         const std::array<int16_t, 16> &b) noexcept
  {
    // Note: this isn't strictly necessary, but it'd be faster just
    // to zero-out the whole array and so we disallow it.
    // If you want to zero out the whole array, then you can use s256_xor_epi64
    // or similar.
    assert(a != b);
    std::array<int16_t, 16> c;
#ifdef __AVX2__
    _mm256_store_si256(reinterpret_cast<__m256i *>(&c),
                       _mm256_cmpgt_epi16(*reinterpret_cast<const __m256i *>(&a),
                                          *reinterpret_cast<const __m256i *>(&b)));
#elif defined(__SSE2__)
    _mm_store_si128(reinterpret_cast<__m128i *>(&c),
                    _mm_cmpgt_epi16(*reinterpret_cast<const __m128i *>(&a),
                                    *reinterpret_cast<const __m128i *>(&b)));
    _mm_store_si128(reinterpret_cast<__m128i *>(&c[8]),
                    _mm_cmpgt_epi16(*reinterpret_cast<const __m128i *>(&a[8]),
                                    *reinterpret_cast<const __m128i *>(&b[8])));
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
      c[i] = (a[i] > b[i]) * 0xFFFF;
    }
#endif
    return c;
  }

  /**
   * m256_shuffle_epi8.
   *
   * This function accepts to array references, a and b. It shuffle the bytes in
   * a according to the mask in b and returns the result, denoted as c.
   *
   * This corresponds exactly to the behaviour of _mm256_shuffle_epi8.
   *
   * The best way to understand
   * what this function does is to view it as applying _mm_shuffle_epi8 twice:
   * once to the lower lane of a, and once to the upper lane of a. The
   * description of this function can be found at:
   * https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=shuffle_epi8&expand=5153.
   *
   * This function will only work if a and b are not equal.
   *
   * As GCC 10.2 has trouble producing vectorised
   * object code for this function (see https://godbolt.org/z/419Whz),
   * where possible we delegate to the relevant
   * intrinsics. In particular, this function:
   *
   * a) If __AVX2__ is defined, it uses the AVX256 function _mm256_shuffle_epi8.
   * b) If __AVX2__ is not defined, but __SSE2__ is, it uses the
   * _mm_shuffle_epi8 function twice.
   * c) Otherwise, use the hand-written
   * variant.
   */
  static inline std::array<int8_t, 32> m256_shuffle_epi8(const std::array<int8_t, 32> &a,
                                                         const std::array<int8_t, 32> &b) noexcept
  {
    // Correctness pre-conditions.
    assert(a != b);
    std::array<int8_t, 32> c;

#ifdef __AVX2__
    _mm256_store_si256(reinterpret_cast<__m256i *>(&c),
                       _mm256_shuffle_epi8(*reinterpret_cast<const __m256i *>(&a),
                                           *reinterpret_cast<const __m256i *>(&b)));
    return c;
#elif defined(__SSE3__)
    _mm_store_si128(reinterpret_cast<__m128i *>(&c),
                    _mm_shuffle_epi8(*reinterpret_cast<const __m128i *>(&a),
                                     *reinterpret_cast<const __m128i *>(&b)));
    _mm_store_si128(reinterpret_cast<__m128i *>(&c + 8),
                    _mm_shuffle_epi8(*reinterpret_cast<const __m128i *>(&a + 8),
                                     *reinterpret_cast<const __m128i *>(&b + 8)));
    return c;
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
    for (unsigned int i = 0; i < 16; i++)
    {
      const unsigned flag = b[i] & 0x80;
      const unsigned pos  = b[i] & 0x0F;
      c[i]                = int16_t(~flag) * a[pos];

      const unsigned flag2 = b[i + 16] & 0x80;
      const unsigned pos2  = b[i + 16] & 0x0F;
      c[i + 16]            = int16_t(~flag2) * a[pos2 + 16];
    }
    return c;
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
  static std::array<int16_t, 16> m256_add_epi16(const std::array<int16_t, 16> &a,
                                                const std::array<int16_t, 16> &b) noexcept
  {
    std::array<int16_t, 16> c;
    // Note; this very trivial for-loop should be trivial for the compiler to
    // optimise, especially if it knows the size of the arrays ahead of time
    // (which it does!)
    for (unsigned int i = 0; i < 16; i++)
    {
      c[i] = a[i] + b[i];
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
  static std::array<int16_t, 16> m256_sub_epi16(const std::array<int16_t, 16> &a,
                                                const std::array<int16_t, 16> &b) noexcept
  {
    std::array<int16_t, 16> c;
    // Note; this very trivial for-loop should be trivial for the compiler to
    // optimise, especially if it knows the size of the arrays ahead of time
    // (which it does!)
    for (unsigned int i = 0; i < 16; i++)
    {
      c[i] = a[i] - b[i];
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
  static inline std::array<int16_t, 16> m256_sign_epi16(const std::array<int16_t, 16> &a,
                                                        const std::array<int16_t, 16> &b) noexcept
  {
    // pre-conditions for the function to work.
    assert(a != b);
    std::array<int16_t, 16> c;
#ifdef __AVX2__
    _mm256_store_si256(reinterpret_cast<__m256i *>(&c),
                       _mm256_sign_epi16(*reinterpret_cast<const __m256i *>(&a),
                                         *reinterpret_cast<const __m256i *>(&b)));
#elif defined(__SSE3__)
    _mm_store_si128(reinterpret_cast<__m128i *>(&c),
                    _mm_sign_epi16(*reinterpret_cast<const __m128i *>(&a),
                                   *reinterpret_cast<const __m128i *>(&b)));
    _mm_store_si128(reinterpret_cast<__m128i *>(&c[8]),
                    _mm_sign_epi16(*reinterpret_cast<const __m128i *>(&a[8]),
                                   *reinterpret_cast<const __m128i *>(&b[8])));
#else
    // This function is rather simple: we extract the signs of each b[i] and
    // multiply a[i] by them.
    for (unsigned int i = 0; i < 16; i++)
    {
      c[i] = a[i] * e_sign(b[i]);
    }
#endif
    return c;
  }

  /**
   * m256_permute4x64_epi64. Permutes the array a according to the bitmask imm8
   * and returns the result, denoted as b. This corresponds exactly to the _mm256_permute4x64_epi64
   * intrinsic.
   *
   * Because GCC 10.2 seems to struggle with producing the right object code(see
   * https://godbolt.org/z/jGb46h), we provide an AVX2 overload if it is defined by the compiler. In
   * particular: a) if __AVX2__ is defined, we use the _mm256_permute4x64_epi64
   * intrinsic. b) else, we use our hand-written version.
   *
   * Implementor's notes: this function relies on
   * binary literals. This feature was only officially available in standard C++
   * in C++14 and later. However, this has been implemented in GCC for far
   * longer than this.
   */
  template <int8_t imm8>
  static std::array<int64_t, 4> m256_permute4x64_epi64(const std::array<int64_t, 4> &a) noexcept
  {
    // As in the contract, the size needs to be 4.
    std::array<int64_t, 4> b;
#ifdef __AVX2__
    _mm256_store_si256(reinterpret_cast<__m256i *>(&b),
                       _mm256_permute4x64_epi64(*reinterpret_cast<const __m256i *>(&a), imm8));
#else
    // This function works as follows: grabs the index from each of the bytes of
    // imm8 We isolate these bytes by bitwise ops, and then shift if necessary
    // Note; these are constexpr variables, which means that these masks are
    // computed at compile-time. As a result, this is just a series of mov
    // instructions, which occur at a rate of approximately 4 per clock: a
    // clever compiler will interleave these movs to hide the latency.
    constexpr unsigned zero =
        (imm8 & BitPatterns::int8_zero_pair_mask) >> BitPatterns::int8_zero_pair_shift;
    constexpr unsigned first =
        (imm8 & BitPatterns::int8_first_pair_mask) >> BitPatterns::int8_first_pair_shift;
    constexpr unsigned second =
        (imm8 & BitPatterns::int8_second_pair_mask) >> BitPatterns::int8_second_pair_shift;
    // This doesn't require a shift because the bytes are already in the
    // bottom-most byte.
    constexpr unsigned third =
        (imm8 & BitPatterns::int8_third_pair_mask) >> BitPatterns::int8_third_pair_shift;

    // These asserts are just to make sure that the code statically does the
    // right thing.
    static_assert(zero < 4, "Error: zero >= size.");
    static_assert(first < 4, "Error: first >= size.");
    static_assert(second < 4, "Error: second >= size.");
    static_assert(third < 4, "Error: third >= size.");

    // Finally, we do the permutation and return.
    b[0] = a[zero];
    b[1] = a[first];
    b[2] = a[second];
    b[3] = a[third];
#endif
    return b;
  }

  /***
   * m256_permute4x64_epi16. This function is a special wrapper around the
   * permutation function, moving each element around under the control of the
   * immediate imm8. Because C++ has slightly different aliasing rules to the
   * Intel intrinsics, it's often necessary to convert between array types to
   * get the desired semantics. This is a real pain, because it requires
   * allocations and that'll slow down the code dramatically.
   *
   * As a result, for this one special case we define a different wrapper, which
   * allows us to do 64-bit operations on a 16-bit vector. This treats each
   * batch of 4 16-bit entries as one larger 64-bit entry. The semantics are
   * exactly the same as permute4x64_epi64.
   *
   * Similarly to above, we add an AVX2 guard. In particular, if __AVX2__ is defined, we use
   * the _mm256_permute4x64_epi64 instruction. Otherwise we just use our handwritten variant.
   */
  template <int8_t imm8>
  static std::array<int16_t, 16> m256_permute4x64_epi16(const std::array<int16_t, 16> &a)
  {
    // If the right intrinsic is available, we just use that.
    // The intrinsics, being a circuit, isn't as tightly constrained as we are!
    std::array<int16_t, 16> b;
#ifdef __AVX2__
    _mm256_store_si256(reinterpret_cast<__m256i *>(&b),
                       _mm256_permute4x64_epi64(*reinterpret_cast<const __m256i *>(&a[0]), imm8));
#else
    // Do the same as in permute4x64_epi64: produce the masks.
    // Here we do something different though: we use each mask as an indicator
    // for which stride of 4 16-bit entries we want.
    constexpr unsigned zero =
        4 * ((imm8 & BitPatterns::int8_zero_pair_mask) >> BitPatterns::int8_zero_pair_shift);
    constexpr unsigned first =
        4 * ((imm8 & BitPatterns::int8_first_pair_mask) >> BitPatterns::int8_first_pair_shift);
    constexpr unsigned second =
        4 * ((imm8 & BitPatterns::int8_second_pair_mask) >> BitPatterns::int8_second_pair_shift);
    // This doesn't require a shift because the bytes are already in the
    // bottom-most byte.
    constexpr unsigned third =
        4 * ((imm8 & BitPatterns::int8_third_pair_mask) >> BitPatterns::int8_third_pair_shift);

    // These asserts are just to make sure that the code statically does the
    // right thing.
    static_assert(zero < 16, "Error: zero >= 16.");
    static_assert(first < 16, "Error: first >= 16.");
    static_assert(second < 16, "Error: second >=16.");
    static_assert(third < 16, "Error: third >= 16.");

    b[0] = a[zero + 0];
    b[1] = a[zero + 1];
    b[2] = a[zero + 2];
    b[3] = a[zero + 3];

    b[4] = a[first + 0];
    b[5] = a[first + 1];
    b[6] = a[first + 2];
    b[7] = a[first + 3];

    b[8]  = a[second + 0];
    b[9]  = a[second + 1];
    b[10] = a[second + 2];
    b[11] = a[second + 3];

    b[12] = a[third + 0];
    b[13] = a[third + 1];
    b[14] = a[third + 2];
    b[15] = a[third + 3];

#endif
    return b;
  }
};

#endif
