/**
 * MCLT Test Suite
 * Tests the fixed MCLT implementation with user-supplied test cases
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <memory>
#include <complex>
#include <vector>
#include "const1.h"
#include "fftreal.h"

// ============================================================================
// MOCK TYPES AND MACROS (since we don't have the full codebase)
// ============================================================================

#define MEM_ALIGN
#define ZITA_LOG(...) printf(__VA_ARGS__); printf("\n"); fflush(stdout)
#define F_MAX(a, b) std::max(a, b)
#define F_MIN(a, b) std::min(a, b)
#define F_ABS(x) std::fabs(x)
#define F_SQRT(x) std::sqrt(x)
#define F_SIN(x) std::sin(x)
#define F_COS(x) std::cos(x)
#define LOG(...) printf(__VA_ARGS__); printf("\n"); fflush(stdout)

template<typename T>
using SimdBase = T;

template<typename T>
constexpr bool IsVector = false;

template<typename T>
struct cmplxT {
    T re, im;
    cmplxT() : re(0), im(0) {}
    cmplxT(T r, T i) : re(r), im(i) {}
    cmplxT operator*(T s) const { return cmplxT(re * s, im * s); }
    cmplxT operator/(T s) const { return cmplxT(re / s, im / s); }
    cmplxT& operator*=(T s) { re *= s; im *= s; return *this; }
};

enum WindowType {
    WINTYPE_SINE = 0,
    WINTYPE_VORBIS = 1,
    WINTYPE_HANN = 2
};

// ============================================================================
// MOCK FFTReal Implementation (with known IFFT gain of N)
// ============================================================================

template<typename T>
class FFTReal {
private:
    int N;
    
public:
    FFTReal(int n) : N(n) {
        ZITA_LOG("FFTReal initialized with N=%d (IFFT gain = %d)", N, N);
    }
    
    // Forward FFT: real input -> packed complex output
    // Gain = 1 (no scaling)
    void do_fft(const T* in, T* out) {
        // Simple DFT for real input
        for (int k = 0; k <= N/2; k++) {
            T re_sum = 0, im_sum = 0;
            for (int n_idx = 0; n_idx < N; n_idx++) {
                T angle = 2.0 * M_PI * k * n_idx / N;
                re_sum += in[n_idx] * cos(angle);
                im_sum += -in[n_idx] * sin(angle);  // Note: negative for FFT convention
            }
            
            if (k == 0 || k == N/2) {
                out[k == 0 ? 0 : N] = re_sum;
                out[k == 0 ? 1 : N+1] = 0;
            } else {
                out[2*k] = re_sum;
                out[2*k+1] = im_sum;
            }
        }
    }
    
    // Inverse FFT: packed complex input -> real output
    // Gain = N (unnormalized IFFT, like FFTReal)
    void do_ifft(const T* in, T* out) {
        // IDFT from packed format
        for (int n_idx = 0; n_idx < N; n_idx++) {
            T sum = 0;
            
            // DC component
            sum += in[0];
            
            // Nyquist component (if N is even)
            sum += in[N] * cos(M_PI * n_idx);
            
            // Other components
            for (int k = 1; k < N/2; k++) {
                T re = in[2*k];
                T im = in[2*k+1];
                T angle = 2.0 * M_PI * k * n_idx / N;
                sum += re * cos(angle) - im * sin(angle);
            }
            
            // Apply gain of N (unnormalized IFFT)
            out[n_idx] = sum * 2.0;  // Factor of 2 because we only summed positive frequencies
        }
    }
};

// ============================================================================
// INCLUDE THE FIXED MCLT IMPLEMENTATION
// ============================================================================

#include "mclt_standalone.h"

// ============================================================================
// TEST SUITE (User's original tests)
// ============================================================================

const int parsize = 2048;  // M = 2048, so N = 4096
using T1 = float;

class MCLTTestSuite {
public:
    void test_fft_roundtrip() {
        ZITA_LOG("\n=== FFT ROUND-TRIP GAIN TEST ===");
        
        FFTReal<T1> fft(parsize * 2);
        const int N = parsize * 2;
        
        MEM_ALIGN T1 time_in[N];
        MEM_ALIGN T1 time_out[N];
        
        // Test with impulse
        memset(time_in, 0, N * sizeof(T1));
        time_in[N/2] = 1.0f;
        
        fft.do_fft(time_in, time_in);  // In-place FFT
        fft.do_ifft(time_in, time_out); // IFFT
        
        float max_out = 0.0f;
        for (int i = 0; i < N; i++) {
            max_out = F_MAX(max_out, F_ABS(time_out[i]));
        }
        
        ZITA_LOG("FFT→IFFT round-trip gain: %.2fx", max_out);
        
        // Test with sine
        for (int i = 0; i < N; i++) {
            time_in[i] = sinf(2.0f * M_PI * 10.0f * i / N);
        }
        
        fft.do_fft(time_in, time_in);
        fft.do_ifft(time_in, time_out);
        
        max_out = 0.0f;
        for (int i = 0; i < N; i++) {
            max_out = F_MAX(max_out, F_ABS(time_out[i]));
        }
        
        ZITA_LOG("Sine round-trip peak: %.6f (input was 1.0)", max_out);
    }
    
    void test_mclt_with_tdac() {
        ZITA_LOG("\n=== TEST: MCLT WITH TDAC OVERLAP ===");
        
        MCLTReal<T1> mclt(parsize * 2);
        const int N = 2 * parsize;
        
        MEM_ALIGN T1 frame1[N], frame2[N], frame3[N];
        MEM_ALIGN cmplxT<T1> freq1[parsize], freq2[parsize];
        MEM_ALIGN T1 output[N];
        
        // Generate continuous sine wave
        for (int i = 0; i < N; i++) {
            float phase = 2.0f * M_PI * 10.0f * i / N;
            frame1[i] = sinf(phase);
            frame2[i] = sinf(phase + M_PI * 10.0f);  // Continue phase from frame1
        }
        
        // Process frames
        mclt.analyze(frame1, freq1);
        mclt.synthesize(freq1);
        mclt.overlap_add_to_buffer(output);  // Warmup frame
        
        mclt.analyze(frame2, freq2);
        mclt.synthesize(freq2);
        mclt.overlap_add_to_buffer(output);  // Real output
        
        // Compare output to ORIGINAL continuous sine (not frame2)
        float max_error = 0.0f;
        for (int i = 0; i < N; i++) {
            float expected = sinf(2.0f * M_PI * 10.0f * (i + N) / N);  // Continue from frame1
            max_error = F_MAX(max_error, F_ABS(output[i] - expected));
        }
        
        ZITA_LOG("TDAC reconstruction error: %.6f", max_error);
        
        if (max_error < 0.01f) {
            ZITA_LOG("✅ TDAC TEST PASSED!");
        } else {
            ZITA_LOG("❌ TDAC TEST FAILED!");
        }
    }
    
    void test_fft_gain() {
        ZITA_LOG("\n=== FFT GAIN TEST ===");
        
        FFTReal<T1> fft(parsize * 2);
        const int N = parsize * 2;
        
        // Test 1: Impulse
        MEM_ALIGN T1 time_in[N];
        MEM_ALIGN T1 freq_buf[N];
        MEM_ALIGN T1 time_out[N];
        
        memset(time_in, 0, N * sizeof(T1));
        time_in[N/2] = 1.0f;  // Unit impulse
        
        // Forward FFT
        fft.do_fft(time_in, freq_buf);
        
        // Measure FFT output energy
        float fft_max = 0.0f;
        for (int i = 0; i < N; i++) {
            fft_max = F_MAX(fft_max, F_ABS(freq_buf[i]));
        }
        
        // Inverse FFT
        fft.do_ifft(freq_buf, time_out);
        
        // Measure IFFT output
        float ifft_max = 0.0f;
        for (int i = 0; i < N; i++) {
            ifft_max = F_MAX(ifft_max, F_ABS(time_out[i]));
        }
        
        ZITA_LOG("Input impulse: 1.0");
        ZITA_LOG("After FFT max: %.6f (gain: %.2fx)", fft_max, fft_max);
        ZITA_LOG("After IFFT max: %.6f (gain: %.2fx)", ifft_max, ifft_max);
        ZITA_LOG("Round-trip gain: %.6fx", ifft_max / 1.0f);
        
        // Test 2: Sine wave
        for (int i = 0; i < N; i++) {
            time_in[i] = sinf(2.0f * M_PI * 10.0f * i / N);
        }
        
        float input_rms = 0.0f;
        for (int i = 0; i < N; i++) {
            input_rms += time_in[i] * time_in[i];
        }
        input_rms = F_SQRT(input_rms / N);
        
        fft.do_fft(time_in, freq_buf);
        fft.do_ifft(freq_buf, time_out);
        
        float output_rms = 0.0f;
        for (int i = 0; i < N; i++) {
            output_rms += time_out[i] * time_out[i];
        }
        output_rms = F_SQRT(output_rms / N);
        
        ZITA_LOG("\nSine wave test:");
        ZITA_LOG("Input RMS: %.6f", input_rms);
        ZITA_LOG("Output RMS: %.6f", output_rms);
        ZITA_LOG("RMS gain: %.6fx", output_rms / input_rms);
    }
    
    void test_mclt_perfect_reconstruction() {
        ZITA_LOG("\n=== TESTING MCLT DIRECT (NO TDAC) ===");
        
        MCLTReal<T1> mclt(parsize * 2);
        
        MEM_ALIGN T1 original[2 * parsize];
        MEM_ALIGN cmplxT<T1> freq[parsize];
        MEM_ALIGN T1 reconstructed[2 * parsize];
        
        // Test 1: Impulse
        memset(original, 0, 2 * parsize * sizeof(T1));
        original[parsize] = 1.0f;
        
        ZITA_LOG("Test 1: Impulse at center");
        
        // Forward
        mclt.analyze(original, freq);
        
        T1 energy_freq = 0.0f;
        for (int i = 0; i < parsize; i++) {
            energy_freq += freq[i].re * freq[i].re + freq[i].im * freq[i].im;
        }
        ZITA_LOG("TEST: freq energy after analyze = %.6f", energy_freq);
       
        // Inverse - DIRECT synthesis without overlap-add
        mclt.synthesize(freq);
        
        // Get the current frame DIRECTLY (bypass overlap_add)
        T1* current = mclt.get_current_frame();
        memcpy(reconstructed, current, 2 * parsize * sizeof(T1));
        
        // Compare
        float max_original = 0.0f, max_recon = 0.0f, max_error = 0.0f;
        for (int i = 0; i < 2 * parsize; i++) {
            max_original = F_MAX(max_original, F_ABS(original[i]));
            max_recon = F_MAX(max_recon, F_ABS(reconstructed[i]));
            max_error = F_MAX(max_error, F_ABS(reconstructed[i] - original[i]));
        }
        
        ZITA_LOG("  Original max: %.6f", max_original);
        ZITA_LOG("  Reconstructed max: %.6f", max_recon);
        ZITA_LOG("  Max error: %.6f", max_error);
        ZITA_LOG("  Relative error: %.6f", max_error / max_original);
        
        if (max_error < 0.01f * max_original) {
            ZITA_LOG("  ✅ PASSED!");
        } else {
            ZITA_LOG("  ❌ FAILED!");
            ZITA_LOG("  Scaling off by: %.6fx", max_recon / max_original);
        }
        
        ZITA_LOG("\nTest 2: Sine wave");
            
        for (int i = 0; i < 2 * parsize; i++) {
            original[i] = sinf(2.0f * M_PI * 10.0f * i / (2 * parsize));
        }
        
        mclt.analyze(original, freq);
        mclt.synthesize(freq);
        
        T1* reconstructed_ptr = mclt.get_current_frame();
        memcpy(reconstructed, reconstructed_ptr, 2 * parsize * sizeof(T1));
        
        max_error = 0.0f;
        for (int i = 0; i < 2 * parsize; i++) {
            max_error = F_MAX(max_error, F_ABS(reconstructed[i] - original[i]));
        }
        
        ZITA_LOG("  Sine max error: %.6f", max_error);
        
        if (max_error < 0.01f) {
            ZITA_LOG("  ✅ SINE TEST PASSED!");
        } else {
            ZITA_LOG("  ❌ SINE TEST FAILED!");
        }
    }
    
    void run_all() {
        ZITA_LOG("╔════════════════════════════════════════════════════════╗");
        ZITA_LOG("║         MCLT TEST SUITE - WITH CORRECTED SCALING       ║");
        ZITA_LOG("╚════════════════════════════════════════════════════════╝");
        ZITA_LOG("");
        
        test_mclt_perfect_reconstruction();
        test_mclt_with_tdac();
        test_fft_gain();
        test_fft_roundtrip();
        
        ZITA_LOG("\n╔════════════════════════════════════════════════════════╗");
        ZITA_LOG("║                     TESTS COMPLETE                     ║");
        ZITA_LOG("╚════════════════════════════════════════════════════════╝");
    }
};

// ============================================================================
// MAIN
// ============================================================================

int main() {
    MCLTTestSuite suite;
    suite.run_all();
    return 0;
}
