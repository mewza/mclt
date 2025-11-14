MCLTReal v1.50
(c) 2025 Dmitry Boldyrev

Written by Dmitry Boldyrev with assistance of AI (claude.ai)

Forward / Reverse MCLT (Modulated complex lapped transform) C++ class 
templatable with a SIMD vector support.

LICENSE: FREE for Commercial and non-commercial use, but would appreciate
credits in About box and/or documentation, or READMEs

Features include:
  
  • Window COLA-based scaling
  • NO extra windowing (window is built into MCLT math)
  • Supports arbitrary HOP sizes with proper WOLA (Weighted Overlap-Add)
  • Uses FFT-to-MCLT mapping (efficient)
  • 3-buffer architecture for flexible overlap
  • Optimized overlap-add for common hop sizes
