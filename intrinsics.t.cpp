extern "C"
{
#include "intrinsics.h"
  union ArrType;
  ArrType m256_shuffle_epi8(const ArrType *const a, const ArrType *const b);
}

#include "gtest/gtest.h"
#include <bitset>
#include <cstdlib>

TEST(testIntrin, testShuffle_epi8)
{
  // This function tests the shuffling functionality in this m256_shuffle_epi8.
  // To begin, we define a random array of 8-bit ints.
  ArrType a;
  ArrType b;

  for (unsigned i = 0; i < 32; i++)
  {
    a.v8[i] = rand();
  }

  // Now we define a general mask too.
  for (unsigned i = 0; i < 32; i++)
  {
    b.v8[i] = rand();
  }

  // Now we shuffle.
  auto c = m256_shuffle_epi8(&a, &b);
  for (unsigned int i = 0; i < 32; i++)
  {
    if (((b.v8[i] & 0x80) >> 7) == 1)
    {
      // the value of c[i] should be 0
      ASSERT_EQ(c.v8[i], 0);
    }
    else
    {
      // Here we actually need to check on the value of c.
      // The 'mask' is from b[i] & 0x0F -- it only uses the final 4 bits.
      // However, if the value of `i` is greater than 15 we also add 16 to account
      // for the offset.
      const unsigned int index =
          static_cast<unsigned>(((i > 15) * 16)) + static_cast<unsigned>((b.v8[i] & 0x0F));
      ASSERT_EQ(c.v8[i], a.v8[index]);
    }
  }

  // We now do exactly the same thing, but over 16-bit entries.
  ArrType a1;
  ArrType b1;
  for (unsigned i = 0; i < 16; i++)
  {
    a1.v16[i] = rand();
    b1.v16[i] = rand();
  }

  const auto c1 = m256_shuffle_epi8(&a1, &b1);

  for (unsigned int i = 0; i < 32; i++)
  {
    if (((b1.v8[i] & 0x80) >> 7) == 1)
    {
      // the value of c[i] should be 0
      ASSERT_EQ(c1.v8[i], 0);
    }
    else
    {
      // Here we actually need to check on the value of c.
      // The 'mask' is from b[i] & 0x0F -- it only uses the final 4 bits.
      // However, if the value of `i` is greater than 15 we also add 16 to account
      // for the offset.
      const unsigned int index =
          static_cast<unsigned>(((i > 15) * 16)) + static_cast<unsigned>((b1.v8[i] & 0x0F));
      ASSERT_EQ(c1.v8[i], a1.v8[index]);
    }
  }
}

TEST(testIntrin, testPermu4x64_epi64)
{
  // The values of this doesn't matter.
  ArrType a, b, c, d;
  for (unsigned int i = 0; i < 4; i++)
  {
    a.v64[i] = rand();
  }

  for (unsigned int i = 0; i < 16; i++)
  {
    c.v16[i] = rand();
  }
  unsigned first, second, third, fourth;

  // In a nod to formal verification, we try all possible masks.
  // This means we can guarantee the permute function works.
  // Since we're permuting in 4 ways, there's 4! = 24 possible combinations.
  // Because the mask is required to be constexpr, we can't write this in a
  // loop: we need to do it manually. Never fear: we can use the preprocessor to
  // do very old-fashioned metaprogramming. In particular, we define the
  // following macro function that will replace the code for us.

#define CHECK_PERMU(a, b, x)                                                                       \
  first  = (x & 0b00000011) >> 0;                                                                  \
  second = (x & 0b00001100) >> 2;                                                                  \
  third  = (x & 0b00110000) >> 4;                                                                  \
  fourth = (x & 0b11000000) >> 6;                                                                  \
  b      = m256_permute4x64_epi64(&a, x);                                                          \
  EXPECT_EQ(b.v64[0], a.v64[first]);                                                               \
  EXPECT_EQ(b.v64[1], a.v64[second]);                                                              \
  EXPECT_EQ(b.v64[2], a.v64[third]);                                                               \
  EXPECT_EQ(b.v64[3], a.v64[fourth]);

  // And now we just unroll.
  // There's probably a for-loop preprocessor macro, but this is the edge of my
  // expertise.
  CHECK_PERMU(a, b, 0);
  CHECK_PERMU(a, b, 1);
  CHECK_PERMU(a, b, 2);
  CHECK_PERMU(a, b, 3);
  CHECK_PERMU(a, b, 4);
  CHECK_PERMU(a, b, 5);
  CHECK_PERMU(a, b, 6);
  CHECK_PERMU(a, b, 7);
  CHECK_PERMU(a, b, 8);
  CHECK_PERMU(a, b, 9);
  CHECK_PERMU(a, b, 10);
  CHECK_PERMU(a, b, 11);
  CHECK_PERMU(a, b, 12);
  CHECK_PERMU(a, b, 13);
  CHECK_PERMU(a, b, 14);
  CHECK_PERMU(a, b, 15);
  CHECK_PERMU(a, b, 16);
  CHECK_PERMU(a, b, 17);
  CHECK_PERMU(a, b, 18);
  CHECK_PERMU(a, b, 19);
  CHECK_PERMU(a, b, 20);
  CHECK_PERMU(a, b, 21);
  CHECK_PERMU(a, b, 22);
  CHECK_PERMU(a, b, 23);
#undef CHECK_PERMU

// Since we're here, we may as well check the 16-bit one too.
#define CHECK_PERMU(a, b, x)                                                                       \
  first  = 4 * ((x & 0b00000011) >> 0);                                                            \
  second = 4 * ((x & 0b00001100) >> 2);                                                            \
  third  = 4 * ((x & 0b00110000) >> 4);                                                            \
  fourth = 4 * ((x & 0b11000000) >> 6);                                                            \
  b      = m256_permute4x64_epi64(&a, x);                                                          \
  EXPECT_EQ(b.v16[0], a.v16[first]);                                                               \
  EXPECT_EQ(b.v16[1], a.v16[first + 1]);                                                           \
  EXPECT_EQ(b.v16[2], a.v16[first + 2]);                                                           \
  EXPECT_EQ(b.v16[3], a.v16[first + 3]);                                                           \
  EXPECT_EQ(b.v16[4], a.v16[second]);                                                              \
  EXPECT_EQ(b.v16[5], a.v16[second + 1]);                                                          \
  EXPECT_EQ(b.v16[6], a.v16[second + 2]);                                                          \
  EXPECT_EQ(b.v16[7], a.v16[second + 3]);                                                          \
  EXPECT_EQ(b.v16[8], a.v16[third]);                                                               \
  EXPECT_EQ(b.v16[9], a.v16[third + 1]);                                                           \
  EXPECT_EQ(b.v16[10], a.v16[third + 2]);                                                          \
  EXPECT_EQ(b.v16[11], a.v16[third + 3]);                                                          \
  EXPECT_EQ(b.v16[12], a.v16[fourth]);                                                             \
  EXPECT_EQ(b.v16[13], a.v16[fourth + 1]);                                                         \
  EXPECT_EQ(b.v16[14], a.v16[fourth + 2]);                                                         \
  EXPECT_EQ(b.v16[15], a.v16[fourth + 3]);

  CHECK_PERMU(c, d, 0);
  CHECK_PERMU(c, d, 1);
  CHECK_PERMU(c, d, 2);
  CHECK_PERMU(c, d, 3);
  CHECK_PERMU(c, d, 4);
  CHECK_PERMU(c, d, 5);
  CHECK_PERMU(c, d, 6);
  CHECK_PERMU(c, d, 7);
  CHECK_PERMU(c, d, 8);
  CHECK_PERMU(c, d, 9);
  CHECK_PERMU(c, d, 10);
  CHECK_PERMU(c, d, 11);
  CHECK_PERMU(c, d, 12);
  CHECK_PERMU(c, d, 13);
  CHECK_PERMU(c, d, 14);
  CHECK_PERMU(c, d, 15);
  CHECK_PERMU(c, d, 16);
  CHECK_PERMU(c, d, 17);
  CHECK_PERMU(c, d, 18);
  CHECK_PERMU(c, d, 19);
  CHECK_PERMU(c, d, 20);
  CHECK_PERMU(c, d, 21);
  CHECK_PERMU(c, d, 22);
  CHECK_PERMU(c, d, 23);
#undef CHECK_PERMU
}

TEST(testIntrin, testz_and_si256)
{
  // Generate two random vectors for the failure case.
  ArrType a, b;
  for (unsigned i = 0; i < 16; i++)
  {
    a.v16[i] = 1;
    b.v16[i] = 0;
  }

  ASSERT_EQ(m256_testz_si256(&a, &b), true);
  ArrType a2, b2;

  for (unsigned int i = 0; i < 16; i++)
  {
    a2.v16[i] = b2.v16[i] = rand();
  }

  ASSERT_EQ(m256_testz_si256(&a2, &b2), false);
}

// Tests for the 16-bit addition function.
TEST(testIntrin, testAdd_epi16)
{

  ArrType a;
  ArrType b;
  ArrType c;

  for (unsigned i = 0; i < 16; i++)
  {
    a.v16[i] = rand();
    b.v16[i] = rand();
  }

  // Now we can actually test that it adds up as we expect.
  c = m256_add_epi16(&a, &b);

  for (unsigned int i = 0; i < 16; i++)
  {
    // Cast to prevent int promotion
    ASSERT_EQ(c.v16[i], int16_t(a.v16[i] + b.v16[i]));
  }
}

// Tests for addition functions.
TEST(testIntrin, testSub_epi16)
{
  ArrType a, b, c;

  for (unsigned i = 0; i < 16; i++)
  {
    a.v16[i] = rand();
    b.v16[i] = rand();
  }

  // Now we can actually test that these subtract as we expect.
  c = m256_sub_epi16(&a, &b);

  for (unsigned int i = 0; i < 16; i++)
  {
    // Cast to prevent int promotion
    EXPECT_EQ(c.v16[i], int16_t(a.v16[i] - b.v16[i]));
  }
}

TEST(testIntrin, testSignEPI16)
{
  ArrType a, b, c;

  for (unsigned i = 0; i < 16; i++)
  {
    a.v16[i] = rand();
    b.v16[i] = rand();
  }

  // However, with that passed we can call it properly.
  c = m256_sign_epi16(&a, &b);

  // note: this loop just checks for bit equality. You can interpret them
  // however you like :) The point here is we're explicitly checking the
  // semantics of the sign_epi16 function.
  for (unsigned int i = 0; i < 16; i++)
  {
    if (b.v16[i] < 0)
    {
      EXPECT_EQ(c.v16[i], -a.v16[i]);
    }
    else if (b.v16[i] == 0)
    {
      EXPECT_EQ(c.v16[i], 0);
    }
    else
    {
      EXPECT_EQ(c.v16[i], a.v16[i]);
    }
  }
}

TEST(testIntrin, testAndEPI16)
{
  ArrType a;
  ArrType b;
  for (unsigned int i = 0; i < 4; i++)
  {
    a.v64[i] = static_cast<int64_t>(rand());
    b.v64[i] = static_cast<int64_t>(rand());
  }

  // Here we death test: the call to the regular and_si256 should terminate if
  // the size isn't 4.
  ArrType c = m256_and_epi64(&a, &b);

  for (unsigned int i = 0; i < 4; i++)
  {
    ASSERT_EQ(c.v64[i], a.v64[i] & b.v64[i]);
  }
}

TEST(testIntrin, testXorEPI16)
{
  ArrType a;
  ArrType b;
  for (unsigned int i = 0; i < 4; i++)
  {
    a.v64[i] = static_cast<int64_t>(rand());
    b.v64[i] = static_cast<int64_t>(rand());
  }

  ArrType c = m256_xor_epi64(&a, &b);

  for (unsigned int i = 0; i < 4; i++)
  {
    ASSERT_EQ(c.v64[i], a.v64[i] ^ b.v64[i]);
  }
}

TEST(testIntrin, testOrEpi16)
{
  ArrType a;
  ArrType b;
  for (unsigned int i = 0; i < 4; i++)
  {
    a.v64[i] = static_cast<int64_t>(rand());
    b.v64[i] = static_cast<int64_t>(rand());
  }

  ArrType c = m256_or_epi64(&a, &b);

  for (unsigned int i = 0; i < 4; i++)
  {
    ASSERT_EQ(c.v64[i], a.v64[i] | b.v64[i]);
  }
}

TEST(testIntrin, testHaddEpi16)
{
  ArrType a, b;

  for (unsigned int i = 0; i < 16; i++)
  {
    a.v16[i] = rand();
    b.v16[i] = rand();
  }

  ArrType c = m256_hadd_epi16(&a, &b);
  for (unsigned int i = 0; i < 16; i += 8)
  {
    EXPECT_EQ(int16_t(c.v16[i + 0]), int16_t(a.v16[i + 0] + a.v16[i + 1]));
    EXPECT_EQ(int16_t(c.v16[i + 1]), int16_t(a.v16[i + 2] + a.v16[i + 3]));
    EXPECT_EQ(int16_t(c.v16[i + 2]), int16_t(a.v16[i + 4] + a.v16[i + 5]));
    EXPECT_EQ(int16_t(c.v16[i + 3]), int16_t(a.v16[i + 6] + a.v16[i + 7]));
    EXPECT_EQ(int16_t(c.v16[i + 4]), int16_t(b.v16[i + 0] + b.v16[i + 1]));
    EXPECT_EQ(int16_t(c.v16[i + 5]), int16_t(b.v16[i + 2] + b.v16[i + 3]));
    EXPECT_EQ(int16_t(c.v16[i + 6]), int16_t(b.v16[i + 4] + b.v16[i + 5]));
    EXPECT_EQ(int16_t(c.v16[i + 7]), int16_t(b.v16[i + 6] + b.v16[i + 7]));
  }
}

TEST(testIntrin, testSLLIepi16)
{
  ArrType a, b;
  for (unsigned int i = 0; i < 16; i++)
  {
    a.v16[i] = rand();
  }

#define CHECK_LSHIFT(a, b, x)                                                                      \
  b = m256_slli_epi16(&a, x);                                                                      \
  EXPECT_EQ(b.v16[0], int16_t(a.v16[0] << x));                                                     \
  EXPECT_EQ(b.v16[1], int16_t(a.v16[1] << x));                                                     \
  EXPECT_EQ(b.v16[2], int16_t(a.v16[2] << x));                                                     \
  EXPECT_EQ(b.v16[3], int16_t(a.v16[3] << x));                                                     \
  EXPECT_EQ(b.v16[4], int16_t(a.v16[4] << x));                                                     \
  EXPECT_EQ(b.v16[5], int16_t(a.v16[5] << x));                                                     \
  EXPECT_EQ(b.v16[6], int16_t(a.v16[6] << x));                                                     \
  EXPECT_EQ(b.v16[7], int16_t(a.v16[7] << x));                                                     \
  EXPECT_EQ(b.v16[8], int16_t(a.v16[8] << x));                                                     \
  EXPECT_EQ(b.v16[9], int16_t(a.v16[9] << x));                                                     \
  EXPECT_EQ(b.v16[10], int16_t(a.v16[10] << x));                                                   \
  EXPECT_EQ(b.v16[11], int16_t(a.v16[11] << x));                                                   \
  EXPECT_EQ(b.v16[12], int16_t(a.v16[12] << x));                                                   \
  EXPECT_EQ(b.v16[13], int16_t(a.v16[13] << x));                                                   \
  EXPECT_EQ(b.v16[14], int16_t(a.v16[14] << x));                                                   \
  EXPECT_EQ(b.v16[15], int16_t(a.v16[15] << x));

  CHECK_LSHIFT(a, b, 0);
  CHECK_LSHIFT(a, b, 1);
  CHECK_LSHIFT(a, b, 2);
  CHECK_LSHIFT(a, b, 3);
  CHECK_LSHIFT(a, b, 4);

  CHECK_LSHIFT(a, b, 5);
  CHECK_LSHIFT(a, b, 6);
  CHECK_LSHIFT(a, b, 7);
  CHECK_LSHIFT(a, b, 8);
  CHECK_LSHIFT(a, b, 9);

  CHECK_LSHIFT(a, b, 10);
  CHECK_LSHIFT(a, b, 11);
  CHECK_LSHIFT(a, b, 12);
  CHECK_LSHIFT(a, b, 13);
  CHECK_LSHIFT(a, b, 14);
  CHECK_LSHIFT(a, b, 15);
  CHECK_LSHIFT(a, b, 16);
}

TEST(testIntrin, testSRLIepi16)
{
  ArrType a, b;
  for (unsigned int i = 0; i < 16; i++)
  {
    a.v16[i] = rand();
  }

#define CHECK_RSHIFT(a, b, x)                                                                      \
  b = m256_srli_epi16(&a, x);                                                                      \
  EXPECT_EQ(b.v16[0], int16_t(((uint16_t)a.v16[0]) >> x));                                         \
  EXPECT_EQ(b.v16[1], int16_t(((uint16_t)a.v16[1]) >> x));                                         \
  EXPECT_EQ(b.v16[2], int16_t(((uint16_t)a.v16[2]) >> x));                                         \
  EXPECT_EQ(b.v16[3], int16_t(((uint16_t)a.v16[3]) >> x));                                         \
  EXPECT_EQ(b.v16[4], int16_t(((uint16_t)a.v16[4]) >> x));                                         \
  EXPECT_EQ(b.v16[5], int16_t(((uint16_t)a.v16[5]) >> x));                                         \
  EXPECT_EQ(b.v16[6], int16_t(((uint16_t)a.v16[6]) >> x));                                         \
  EXPECT_EQ(b.v16[7], int16_t(((uint16_t)a.v16[7]) >> x));                                         \
  EXPECT_EQ(b.v16[8], int16_t(((uint16_t)a.v16[8]) >> x));                                         \
  EXPECT_EQ(b.v16[9], int16_t(((uint16_t)a.v16[9]) >> x));                                         \
  EXPECT_EQ(b.v16[10], int16_t(((uint16_t)a.v16[10]) >> x));                                       \
  EXPECT_EQ(b.v16[11], int16_t(((uint16_t)a.v16[11]) >> x));                                       \
  EXPECT_EQ(b.v16[12], int16_t(((uint16_t)a.v16[12]) >> x));                                       \
  EXPECT_EQ(b.v16[13], int16_t(((uint16_t)a.v16[13]) >> x));                                       \
  EXPECT_EQ(b.v16[14], int16_t(((uint16_t)a.v16[14]) >> x));                                       \
  EXPECT_EQ(b.v16[15], int16_t(((uint16_t)a.v16[15]) >> x));

  CHECK_RSHIFT(a, b, 0);
  CHECK_RSHIFT(a, b, 1);
  CHECK_RSHIFT(a, b, 2);
  CHECK_RSHIFT(a, b, 3);
  CHECK_RSHIFT(a, b, 4);

  CHECK_RSHIFT(a, b, 5);
  CHECK_RSHIFT(a, b, 6);
  CHECK_RSHIFT(a, b, 7);
  CHECK_RSHIFT(a, b, 8);
  CHECK_RSHIFT(a, b, 9);

  CHECK_RSHIFT(a, b, 10);
  CHECK_RSHIFT(a, b, 11);
  CHECK_RSHIFT(a, b, 12);
  CHECK_RSHIFT(a, b, 13);
  CHECK_RSHIFT(a, b, 14);
  CHECK_RSHIFT(a, b, 15);
  CHECK_RSHIFT(a, b, 16);
}

TEST(testIntrin, testAbsepi16)
{
  ArrType a;
  for (unsigned i = 0; i < 16; i++)
  {
    a.v16[i] = rand();
  }

  auto b = m256_abs_epi16(&a);
  for (unsigned int i = 0; i < 16; i++)
  {
    EXPECT_EQ(b.v16[i], std::abs(a.v16[i]));
  }
}

TEST(testIntrin, testRand)
{
  ArrType a, b;
  a.v64[0] = static_cast<uint64_t>(rand());
  a.v64[1] = static_cast<uint64_t>(rand());
  b.v64[0] = static_cast<uint64_t>(rand());
  b.v64[1] = static_cast<uint64_t>(rand());
  auto k   = get_randomness(&a, &b);
  EXPECT_NE((__uint128_t)k.m128[0], (__uint128_t)a.m128[0]);
  EXPECT_NE((__uint128_t)k.m128[0], (__uint128_t)b.m128[0]);
}

TEST(testIntrin, testBroadcast)
{
  // Generate a random __uint128_t to use as our broadcast value.
  ArrType a;
  a.m128[0] = (__m128i) static_cast<__uint128_t>(rand());
  auto c    = m256_broadcastsi128_si256(&a);

  // Firstly we check for consistency in the array: in particular, a[i] = a[i+8] for i in {0, 7}
  for (unsigned int i = 0; i < 8; i++)
  {
    ASSERT_EQ(c.v16[i], c.v16[i + 8]);
  }

  // And then we check that the values are what we actually expect when converted to a uint128_t
  __uint128_t out;
  memcpy(&out, &c.m128[0], sizeof(out));
  ASSERT_EQ(out, (__uint128_t)a.m128[0]);
}

TEST(testIntrin, testExtract)
{
  ArrType a;
  a.m128[0] = (__m128i) static_cast<__uint128_t>(rand());
  auto c    = mm_extract_epi64(&a, 0);
  auto d    = mm_extract_epi64(&a, 1);
  // Extract the 64-bit quantities manually.
  uint64_t c1 = ((__uint128_t)a.m128[0]) & 0xFFFFFFFFFFFFFFFF;
  uint64_t c2 = (((__uint128_t)a.m128[0]) >> 64) & 0xFFFFFFFFFFFFFFFF;
  ASSERT_EQ(c1, c);
  ASSERT_EQ(c2, d);
}
