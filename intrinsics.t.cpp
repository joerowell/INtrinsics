#include "intrinsics.hpp"
#include "gtest/gtest.h"
#include <cstdlib>

/***
 * Intrinsics.t.cpp.
 * This file contains the test drivers for each of the methods in the
 * intrinsics.hpp file. This file is also used for CI on the high-level git
 * repo: as a result, any changes to this file will automatically be reflected
 * during the CI stage.
 */

TEST(testIntrin, testPermu4x64_epi64)
{
  // The values of this doesn't matter.
  std::array<int64_t, 4> a = {static_cast<int64_t>(rand()), static_cast<int64_t>(rand()),
                              static_cast<int64_t>(rand()), static_cast<int64_t>(rand())};
  std::array<int64_t, 4> b;

  std::array<int16_t, 16> c;
  std::array<int16_t, 16> d;

  for (unsigned int i = 0; i < 16; i++)
  {
    c[i] = rand();
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
  b      = CPP_INTRIN::m256_permute4x64_epi64<x>(a);                                               \
  EXPECT_EQ(b[0], a[first]);                                                                       \
  EXPECT_EQ(b[1], a[second]);                                                                      \
  EXPECT_EQ(b[2], a[third]);                                                                       \
  EXPECT_EQ(b[3], a[fourth]);

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
  b      = CPP_INTRIN::m256_permute4x64_epi16<x>(a);                                               \
  EXPECT_EQ(b[0], a[first]);                                                                       \
  EXPECT_EQ(b[1], a[first + 1]);                                                                   \
  EXPECT_EQ(b[2], a[first + 2]);                                                                   \
  EXPECT_EQ(b[3], a[first + 3]);                                                                   \
  EXPECT_EQ(b[4], a[second]);                                                                      \
  EXPECT_EQ(b[5], a[second + 1]);                                                                  \
  EXPECT_EQ(b[6], a[second + 2]);                                                                  \
  EXPECT_EQ(b[7], a[second + 3]);                                                                  \
  EXPECT_EQ(b[8], a[third]);                                                                       \
  EXPECT_EQ(b[9], a[third + 1]);                                                                   \
  EXPECT_EQ(b[10], a[third + 2]);                                                                  \
  EXPECT_EQ(b[11], a[third + 3]);                                                                  \
  EXPECT_EQ(b[12], a[fourth]);                                                                     \
  EXPECT_EQ(b[13], a[fourth + 1]);                                                                 \
  EXPECT_EQ(b[14], a[fourth + 2]);                                                                 \
  EXPECT_EQ(b[15], a[fourth + 3]);

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

// Tests for the 16-bit addition function.
TEST(testIntrin, testAdd_epi16)
{
  std::array<int16_t, 16> a;
  std::array<int16_t, 16> b;
  std::array<int16_t, 16> c;

  for (unsigned i = 0; i < 16; i++)
  {
    a[i] = rand();
    b[i] = rand();
  }

  // Now we can actually test that it adds up as we expect.
  c = CPP_INTRIN::m256_add_epi16(a, b);

  for (unsigned int i = 0; i < 16; i++)
  {
    // Cast to prevent int promotion
    ASSERT_EQ(c[i], int16_t(a[i] + b[i]));
  }
}

// Tests for addition functions.
TEST(testIntrin, testSub_epi16)
{
  std::array<int16_t, 16> a;
  std::array<int16_t, 16> b;
  std::array<int16_t, 16> c;

  std::array<int32_t, 8> a1;
  std::array<int32_t, 8> b1;

  for (unsigned i = 0; i < 16; i++)
  {
    a[i] = rand();
    b[i] = rand();
  }

  // Now we can actually test that it adds up as we expect.
  c = CPP_INTRIN::m256_sub_epi16(a, b);

  for (unsigned int i = 0; i < 16; i++)
  {
    // Cast to prevent int promotion
    EXPECT_EQ(c[i], int16_t(a[i] - b[i]));
  }
}

TEST(testIntrin, testSignEPI16)
{
  std::array<int16_t, 16> a;
  std::array<int16_t, 16> b;
  std::array<int16_t, 16> c;

  for (unsigned i = 0; i < 16; i++)
  {
    a[i] = rand();
    b[i] = rand();
  }

  // However, with that passed we can call it properly.
  c = CPP_INTRIN::m256_sign_epi16(a, b);

  // note: this loop just checks for bit equality. You can interpret them
  // however you like :) The point here is we're explicitly checking the
  // semantics of the sign_epi16 function.
  for (unsigned int i = 0; i < 16; i++)
  {
    if (b[i] < 0)
    {
      EXPECT_EQ(c[i], -a[i]);
    }
    else if (b[i] == 0)
    {
      EXPECT_EQ(c[i], 0);
    }
    else
    {
      EXPECT_EQ(c[i], a[i]);
    }
  }
}

TEST(testIntrin, testAndEPI16)
{
  std::array<int64_t, 4> a;
  std::array<int64_t, 4> b;
  for (unsigned int i = 0; i < 4; i++)
  {
    a[i] = static_cast<int64_t>(rand());
    b[i] = static_cast<int64_t>(rand());
  }

  std::array<int64_t, 4> c;
  // Here we death test: the call to the regular and_si256 should terminate if
  // the size isn't 4.
  c = CPP_INTRIN::m256_and_epi64(a, b);

  for (unsigned int i = 0; i < 4; i++)
  {
    ASSERT_EQ(c[i], a[i] & b[i]);
  }
}
TEST(testIntrin, testXorEPI16)
{
  std::array<int64_t, 4> a;
  std::array<int64_t, 4> b;
  for (unsigned int i = 0; i < 4; i++)
  {
    a[i] = static_cast<int64_t>(rand());
    b[i] = static_cast<int64_t>(rand());
  }

  std::array<int64_t, 4> c;
  // Here we death test: the call to the regular and_si256 should terminate if
  // the size isn't 4.
  c = CPP_INTRIN::m256_xor_epi64(a, b);

  for (unsigned int i = 0; i < 4; i++)
  {
    ASSERT_EQ(c[i], a[i] ^ b[i]);
  }
}

TEST(testIntrin, testOrEPI16)
{
  std::array<int64_t, 4> a;
  std::array<int64_t, 4> b;
  for (unsigned int i = 0; i < 4; i++)
  {
    a[i] = static_cast<int64_t>(rand());
    b[i] = static_cast<int64_t>(rand());
  }

  std::array<int64_t, 4> c;
  // Here we death test: the call to the regular and_si256 should terminate if
  // the size isn't 4.
  c = CPP_INTRIN::m256_or_epi64(a, b);

  for (unsigned int i = 0; i < 4; i++)
  {
    ASSERT_EQ(c[i], a[i] | b[i]);
  }
}
TEST(testIntrin, testHaddEpi16)
{
  std::array<int16_t, 16> a;
  std::array<int16_t, 16> b;
  std::array<int16_t, 16> c;

  for (unsigned int i = 0; i < 16; i++)
  {
    a[i] = rand();
    b[i] = rand();
  }

  c = CPP_INTRIN::m256_hadd_epi16(a, b);

  for (unsigned int i = 0; i < 16; i += 8)
  {
    EXPECT_EQ(int16_t(c[i + 0]), int16_t(a[i + 0] + a[i + 1]));
    EXPECT_EQ(int16_t(c[i + 1]), int16_t(a[i + 2] + a[i + 3]));
    EXPECT_EQ(int16_t(c[i + 2]), int16_t(a[i + 4] + a[i + 5]));
    EXPECT_EQ(int16_t(c[i + 3]), int16_t(a[i + 6] + a[i + 7]));
    EXPECT_EQ(int16_t(c[i + 4]), int16_t(b[i + 0] + b[i + 1]));
    EXPECT_EQ(int16_t(c[i + 5]), int16_t(b[i + 2] + b[i + 3]));
    EXPECT_EQ(int16_t(c[i + 6]), int16_t(b[i + 4] + b[i + 5]));
    EXPECT_EQ(int16_t(c[i + 7]), int16_t(b[i + 6] + b[i + 7]));
  }
}

TEST(testIntrin, testSLLIepi16)
{
  std::array<int16_t, 16> a;
  std::array<int16_t, 16> b;
  for (unsigned int i = 0; i < 16; i++)
  {
    a[i] = rand();
  }

#define CHECK_LSHIFT(a, b, x)                                                                      \
  b = CPP_INTRIN::m256_slli_epi16<x>(a);                                                           \
  EXPECT_EQ(b[0], int16_t(a[0] << x));                                                             \
  EXPECT_EQ(b[1], int16_t(a[1] << x));                                                             \
  EXPECT_EQ(b[2], int16_t(a[2] << x));                                                             \
  EXPECT_EQ(b[3], int16_t(a[3] << x));                                                             \
  EXPECT_EQ(b[4], int16_t(a[4] << x));                                                             \
  EXPECT_EQ(b[5], int16_t(a[5] << x));                                                             \
  EXPECT_EQ(b[6], int16_t(a[6] << x));                                                             \
  EXPECT_EQ(b[7], int16_t(a[7] << x));                                                             \
  EXPECT_EQ(b[8], int16_t(a[8] << x));                                                             \
  EXPECT_EQ(b[9], int16_t(a[9] << x));                                                             \
  EXPECT_EQ(b[10], int16_t(a[10] << x));                                                           \
  EXPECT_EQ(b[11], int16_t(a[11] << x));                                                           \
  EXPECT_EQ(b[12], int16_t(a[12] << x));                                                           \
  EXPECT_EQ(b[13], int16_t(a[13] << x));                                                           \
  EXPECT_EQ(b[14], int16_t(a[14] << x));                                                           \
  EXPECT_EQ(b[15], int16_t(a[15] << x));

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
  std::array<int16_t, 16> a;
  std::array<int16_t, 16> b;
  for (unsigned int i = 0; i < 16; i++)
  {
    a[i] = rand();
  }

#define CHECK_RSHIFT(a, b, x)                                                                      \
  b = CPP_INTRIN::m256_srli_epi16<x>(a);                                                           \
  EXPECT_EQ(b[0], int16_t(((uint16_t)a[0]) >> x));                                                 \
  EXPECT_EQ(b[1], int16_t(((uint16_t)a[1]) >> x));                                                 \
  EXPECT_EQ(b[2], int16_t(((uint16_t)a[2]) >> x));                                                 \
  EXPECT_EQ(b[3], int16_t(((uint16_t)a[3]) >> x));                                                 \
  EXPECT_EQ(b[4], int16_t(((uint16_t)a[4]) >> x));                                                 \
  EXPECT_EQ(b[5], int16_t(((uint16_t)a[5]) >> x));                                                 \
  EXPECT_EQ(b[6], int16_t(((uint16_t)a[6]) >> x));                                                 \
  EXPECT_EQ(b[7], int16_t(((uint16_t)a[7]) >> x));                                                 \
  EXPECT_EQ(b[8], int16_t(((uint16_t)a[8]) >> x));                                                 \
  EXPECT_EQ(b[9], int16_t(((uint16_t)a[9]) >> x));                                                 \
  EXPECT_EQ(b[10], int16_t(((uint16_t)a[10]) >> x));                                               \
  EXPECT_EQ(b[11], int16_t(((uint16_t)a[11]) >> x));                                               \
  EXPECT_EQ(b[12], int16_t(((uint16_t)a[12]) >> x));                                               \
  EXPECT_EQ(b[13], int16_t(((uint16_t)a[13]) >> x));                                               \
  EXPECT_EQ(b[14], int16_t(((uint16_t)a[14]) >> x));                                               \
  EXPECT_EQ(b[15], int16_t(((uint16_t)a[15]) >> x));

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
  std::array<int16_t, 16> a;
  for (unsigned i = 0; i < 16; i++)
  {
    a[i] = rand();
  }

  auto b = CPP_INTRIN::m256_abs_epi16(a);
  for (unsigned int i = 0; i < 16; i++)
  {
    EXPECT_EQ(b[i], std::abs(a[i]));
  }
}

TEST(testIntrin, testRand)
{
  __uint128_t a = static_cast<unsigned>(rand());
  __uint128_t b = static_cast<unsigned>(rand());

  auto k = CPP_INTRIN::get_randomness(a, b);
  EXPECT_NE(k, a);
  EXPECT_NE(k, b);
}
