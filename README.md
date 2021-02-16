# INtrinsics
This header-file only project implements some Intel intrinsic SIMD functions in plain C++. 

This is primarily useful in situations where you want to ensure that some intrinsic functionality exists regardless of the platform that you're compiling for. For speed, where appropriate we delegate to the native intrinsics if support is avaiable. We only delegate to these intrinsics if GCC 10.2 fails to generate good vectorised object code on the plain C++ input: the documentation in ``intrinsics.hpp`` points out this when it occurs. 

This repository utilises [Googletest](https://github.com/google/googletest) with Github actions for CI. In particular, we run tests against AVX2, SSE4 and plain C++ on every commit. These tests are in a "TDD-style": they primarily test observed outcomes rather than the mechanics of how the result was derived. This ensures consistent semantics when compiling with different instruction sets.

## How to understand the code
Every piece of code in this header file is documented and tested. You can find the tests in the ``intrinsics.t.cpp`` file: these tests are also documented where appropriate, and so understanding the code from these tests would be a good first step.

Another excellent resource is the [Intel Intrinsics guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/). This contains detailed descriptions of each intrinsic instruction, and therefore it can be 

More generally: reading this header file requires some implicit understanding of how Intel SIMD intrinsics are implemented. 
This also requires some history:

* In the beginning was MMX. MMX brought SIMD across 64-bits to x86. This is why most SIMD instructions on x86 start with ``_m``.
* Then was SSE. SSE operates over 128-bit chunks, which are referred to as ``lanes``. You can think of a lane as a register of 128-bits.
* Then was AVX/AVX2. These instruction sets extended the SIMD capabilities of x86-64 to 256-bit vectors. 

The salient point is this: AVX2 instructions are often implemented as duplicates of SSE instructions. In particular, it is quite common to find AVX2 instructions that operate over the higher lane (i.e the leading 128-bits) and the lower lane (i.e the trailing 128-bits) separately. A good example of this behaviour in action is the ``_mm256_hadd_epi16`` intrinsic, which double-packs the operands over two separate lanes. When in doubt about semantics, the Intel Intrinsics guide is the authoritative source.
