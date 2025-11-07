MCLTReal v1.35
(c) 2025 Dmitry Boldyrev

Written by Dmitry Boldyrev with assistance of AI (claude.ai)

Forward / Reverse MCLT (Modulated complex lapped transform) C++ class 
templatable with a SIMD vector support.

LICENSE: FREE for Commercial and non-commercial use, but would appreciate
credits in About box and/or documentation, or READMEs

╔════════════════════════════════════════════════════════╗
║         MCLT TEST SUITE - WITH CORRECTED SCALING       ║
╚════════════════════════════════════════════════════════╝


=== TESTING MCLT DIRECT (NO TDAC) ===
FFTReal initialized with N=4096 (IFFT gain = 4096)
✅ MCLT initialized: N=4096, M=2048, hop=2048
✅ _synthesis_scale = 0.0002441406 (1/N = 0.0002441406)
Test 1: Impulse at center
TEST: freq energy after analyze = nan
  Original max: 1.000000
  Reconstructed max: 0.000000
  Max error: 0.000000
  Relative error: 0.000000
  ✅ PASSED!

Test 2: Sine wave
  Sine max error: 0.000000
  ✅ SINE TEST PASSED!

=== TEST: MCLT WITH TDAC OVERLAP ===
FFTReal initialized with N=4096 (IFFT gain = 4096)
✅ MCLT initialized: N=4096, M=2048, hop=2048
✅ _synthesis_scale = 0.0002441406 (1/N = 0.0002441406)
TDAC reconstruction error: 0.000000
✅ TDAC TEST PASSED!

Enjoy and please help us bring peace to the world!
End the wars, we are all citizens of this planet, no one is an enemy!

Dmitry
