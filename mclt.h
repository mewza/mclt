/**
 ** MCLTReal v1.35 - FIXED SCALING
 **
 ** Forward / Backward MCLT (Modulated complex lapped transform) C++ class templatable with a SIMD vector
 **  © 2025 Dmitry Boldyrev. All rights reserved.
 **
 ** @brief FFTReal-style interface for MCLT transforms
 **
 **     This class provides an API similar to FFTReal but using MCLT. The key difference
 **     is that all buffers (temp, window) are managed internally, so the API is simpler.
 **
 ** @contact  E-mail: subband@gmail.com
 **
 ** NOTE: this uses FFTReal class to accelerate FFT part and it can be obtained from:
 ** https://github.com/mewza/realfft/
 **
 **/

#pragma once

#include <memory>
#include <complex>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include "fftreal.h"

template<typename T, bool useFFTReal = true>
class MCLTReal {
public:
    using T1 = SimdBase<T>;
    using cmplxTT = cmplxT<T>;
    
private:
    static constexpr size_t ALIGNMENT = 128;
    
    int _length;
    int _M;
    int _hop;
    
    struct AlignedDeleterT { void operator()(T* ptr) const { free(ptr); } };
    struct AlignedDeleterT1 { void operator()(T1* ptr) const { free(ptr); } };
    struct AlignedDeleterCmplxTT { void operator()(cmplxTT* ptr) const { free(ptr); } };
    
    std::unique_ptr<T[], AlignedDeleterT> _prev_prev;
    std::unique_ptr<T[], AlignedDeleterT> _prev;
    std::unique_ptr<T[], AlignedDeleterT> _current;
    std::unique_ptr<cmplxTT[], AlignedDeleterCmplxTT> _temp_complex;
    std::unique_ptr<T[], AlignedDeleterT> _temp_real;
    std::unique_ptr<T1[], AlignedDeleterT1> _window;
    
    WindowType _window_type;
    
    std::conditional_t<useFFTReal, FFTReal<T>, char> _fft;
    
    T1 _synthesis_scale;
    T1 _coeff;
    
    static T* aligned_alloc_T(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T)) != 0)
            return nullptr;
        return static_cast<T*>(ptr);
    }
    
    static cmplxTT* aligned_alloc_cmplxTT(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(cmplxTT)) != 0)
            return nullptr;
        return static_cast<cmplxTT*>(ptr);
    }
    
    static T1* aligned_alloc_T1(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T1)) != 0)
            return nullptr;
        return static_cast<T1*>(ptr);
    }
    
    // Modified Bessel function I0 (for Kaiser window)
    static T1 bessel_i0(T1 x) {
        T1 sum = 1.0;
        T1 term = 1.0;
        T1 m = 1.0;
        
        while (term > 1e-12 * sum) {
            T1 y = x / (2.0 * m);
            term *= y * y;
            sum += term;
            m += 1.0;
        }
        
        return sum;
    }
    
    // Generate Kaiser-Bessel Derived (KBD) window
    void generate_kbd_window(T1* window, int length, T1 alpha = 4.0) {
        const int N = length;
        const int N2 = N / 2;
        
        // Step 1: Generate Kaiser window of length N/2 + 1
        T1* kaiser = new T1[N2 + 1];
        T1 i0_alpha = bessel_i0(M_PI * alpha);
        
        for (int i = 0; i <= N2; i++) {
            T1 x = 2.0 * i / (T1)N - 1.0;
            T1 arg = M_PI * alpha * F_SQRT(1.0 - x * x);
            kaiser[i] = bessel_i0(arg) / i0_alpha;
        }
        
        // Step 2: Compute cumulative sum for first half
        T1* cum_sum = new T1[N2 + 1];
        cum_sum[0] = kaiser[0];
        for (int i = 1; i <= N2; i++) {
            cum_sum[i] = cum_sum[i-1] + kaiser[i];
        }
        
        // Step 3: Normalize and create KBD window
        T1 total = cum_sum[N2];
        
        // First half (sqrt of normalized cumsum)
        for (int i = 0; i < N2; i++) {
            window[i] = F_SQRT(cum_sum[i] / total);
        }
        
        // Second half (mirror of first half)
        for (int i = N2; i < N; i++) {
            window[i] = F_SQRT(cum_sum[N - i] / total);
        }
        
        delete[] kaiser;
        delete[] cum_sum;
    }
    
    // Initialize window based on type
    void init_window() {
        const int N = _length;
        
        switch (_window_type) {
            case WINTYPE_SINE: {
                // Sine window (default, TDAC-compliant)
                const T1 K = M_PI / T1(N);
                for (int i = 0; i < N; i++) {
                    _window[i] = F_SIN(T1(i + 0.5) * K);
                }
                break;
            }
            
            case WINTYPE_KAISER: {
                // Kaiser-Bessel Derived window (TDAC-compliant, AAC standard)
                generate_kbd_window(_window.get(), N, 4.0);  // alpha=4.0 is AAC standard
                break;
            }
            
            case WINTYPE_HANNING: {
                // Hann window (NOT TDAC-compliant! Use only for manual OLA)
                for (int i = 0; i < N; i++) {
                    _window[i] = 0.5 * (1.0 - F_COS(2.0 * M_PI * i / (T1)(N - 1)));
                }
                break;
            }
            
            default:
                // Fallback to sine
                const T1 K = M_PI / T1(N);
                for (int i = 0; i < N; i++) {
                    _window[i] = F_SIN(T1(i + 0.5) * K);
                }
                break;
        }
    }
    
public:
    explicit MCLTReal(int length, int hop = -1, WindowType window_type = WINTYPE_SINE)
        : _length(length)
        , _M(length / 2)
        , _hop(hop < 0 ? _M : hop)
        , _window_type(window_type)
        , _coeff(F_SQRT(2.0 / T1(_M)) / 2)
        , _fft([length]() {
            if constexpr(useFFTReal) {
                return FFTReal<T>(length);
            } else {
                return char{};
            }
        }())
        , _prev_prev(aligned_alloc_T(length))
        , _prev(aligned_alloc_T(length))
        , _current(aligned_alloc_T(length))
        , _temp_complex(aligned_alloc_cmplxTT(2 * _M))
        , _temp_real(aligned_alloc_T(2 * length))
        , _window(aligned_alloc_T1(length))
    {
        if (_hop <= 0 || _hop > length) {
            throw std::invalid_argument("Hop size must be between 1 and 2M");
        }
        
        // Compute overlap factor
        T1 overlap_factor = (T1)_length / (T1)_hop;
        _synthesis_scale = overlap_factor / F_SQRT((T1)_length * 2.0);
        
        // Initialize window based on type
        init_window();
        
        reset();
        
        const char* win_name = (_window_type == WINTYPE_KAISER) ? "KBD" :
                               (_window_type == WINTYPE_HANNING) ? "Hann" : "Sine";
        printf("✅ MCLTRealO: N=%d, M=%d, hop=%d, window=%s\n",
               _length, _M, _hop, win_name);
    }
    
    void reset() {
        memset(_prev_prev.get(), 0, _length * sizeof(T));
        memset(_prev.get(), 0, _length * sizeof(T));
        memset(_current.get(), 0, _length * sizeof(T));
        memset(_temp_real.get(), 0, (2 * _length) * sizeof(T));
    }
    
    T* get_current_frame() { return _current.get(); }
    
    inline int get_length() const { return _length; }
    inline int get_half_length() const { return _M; }
    inline int get_hop() const { return _hop; }
    inline void set_hop(int hop) {
        if (hop <= 0 || hop > _length) {
            throw std::invalid_argument("Hop size must be between 1 and 2M");
        }
        _hop = hop;
    }
    inline T1* window() const { return _window.get(); }
    
    // ========== API ALIASES ==========
    
    inline void real_mclt(const T* in, cmplxTT* out, bool do_scale = false) {
        analyze(in, out);
    }
    
    inline void mclt(const T* in, cmplxTT* out, bool do_scale = false) {
        analyze(in, out);
    }
    
    inline void real_imclt(const cmplxTT* in, T* out, bool do_scale = false) {
        synthesize(in);
        overlap_add_to_buffer(out);
    }
    
    inline void imclt(const cmplxTT* in, T* out, bool do_scale = false) {
        real_imclt(in, out, do_scale);
    }
    
    // ========== FORWARD MCLT ==========
    
    void analyze(const T* __restrict__ x, cmplxTT* __restrict__ X) {
        const int M = _M;
        const int N = 2 * M;
        
        T* fft_buf = _temp_real.get();
        
        // Copy input WITHOUT windowing
        memcpy(fft_buf, x, N * sizeof(T));
        
        // FFT of 2M samples
        _fft.do_fft(fft_buf, fft_buf);
        
        // Unpack complex FFT result
        cmplxTT* fft_out = _temp_complex.get();
        for (int i = 0; i < M; i++) {
            fft_out[i].re = fft_buf[2*i];
            fft_out[i].im = fft_buf[2*i+1];
        }
        
        // Apply FFT-to-MCLT mapping
        const T uL = 0.5 / M;
        const T g_angle = M_PI * (0.5 + uL);
        const T cstep = F_COS(g_angle);
        const T sstep = F_SIN(g_angle);
        const T g = F_SQRT(uL);
        
        T ca = F_COS(M_PI / 4.0);
        T sa = -ca;
        
        T r0 = fft_out[0].re * ca;
        T i0 = fft_out[0].re * sa;
        
        for (int k = 0; k < M - 1; k++) {
            T r1 = fft_out[k+1].re;
            T i1 = fft_out[k+1].im;
            
            T tm = ca * cstep + sa * sstep;
            sa = sa * cstep - ca * sstep;
            ca = tm;
            
            T tp = ca * r1 - sa * i1;
            i1 = sa * r1 + ca * i1;
            r1 = tp;
            
            X[k].re = g * (r1 - i0);
            X[k].im = g * (i1 + r0);
            
            r0 = r1;
            i0 = i1;
        }
        
        T tm = ca * cstep + sa * sstep;
        sa = sa * cstep - ca * sstep;
        ca = tm;
        
        X[M-1].re = g * (ca * fft_out[M].re - i0);
        X[M-1].im = g * (sa * fft_out[M].re + r0);
        
        // Apply coefficient scaling
        for (int k = 0; k < M; k++) {
            X[k] *= _coeff;
        }
    }
    
    // ========== INVERSE MCLT ==========
    
    void synthesize(const cmplxTT* __restrict__ X) {
        const int M = _M;
        const int N = 2 * M;
        
        // Extract and scale coefficients
        cmplxTT* y = _temp_complex.get();
        
        const T1 coeff = 1.;
        for (int k = 0; k < M; k++) {
            y[k].re = X[k].re * coeff;
            y[k].im = X[k].im * coeff;
        }
        
        // Apply IMCLT-to-IDFT mapping
        T* t = _temp_real.get();
        
        const T uL = 0.5 / M;
        const T g_angle = M_PI * (0.5 + uL);
        const T cstep = F_COS(g_angle);
        const T sstep = F_SIN(-g_angle);
        
        T ca = F_COS(M_PI / 4.0);
        T sa = ca;
        
        for (int k = 1; k < M; k++) {
            T r1 = y[k].im + y[k-1].re;
            T i1 = y[k-1].im - y[k].re;
            
            T tm = ca * cstep + sa * sstep;
            sa = sa * cstep - ca * sstep;
            ca = tm;
            
            t[k * 2] = ca * r1 - sa * i1;
            t[k * 2 + 1] = sa * r1 + ca * i1;
        }
        
        t[0] = F_SQRT(2.0) * (y[0].re + y[0].im);
        t[1] = 0.0;
        t[M * 2] = -F_SQRT(2.0) * (y[M-1].re + y[M-1].im);
        t[M * 2 + 1] = 0.0;
        
        int k = M - 1;
        for (int j = M + 1; j < 2 * M; j++) {
            t[j * 2] = t[k * 2];
            t[j * 2 + 1] = -t[k * 2 + 1];
            k--;
        }
        
        // IFFT
        _fft.do_ifft(t, _temp_real.get());
        
        // Apply scaling and window
        T* current = _current.get();
        const T1* h = _window.get();
        const T1 fltScale = _synthesis_scale;
        
        for (int j = 0; j < N; j++) {
            current[j] = _temp_real[j] * fltScale * h[j];
        }
    }
    
    // ========== TDAC OVERLAP-ADD (50% only) ==========
    
    void overlap_add_to_buffer(T* __restrict__ output) {
        const int M = _M;
        const int M_twice = 2 * M;
        
        // First half
        for (int i = 0; i < M; i++) {
            output[i] = _prev_prev[i + M] + _prev[i];
        }
        
        // Second half
        for (int i = M; i < M_twice; i++) {
            output[i] = _current[i - M] + _prev[i];
        }
        
        // Shift frame buffers
        memcpy(_prev_prev.get(), _prev.get(), M_twice * sizeof(T));
        memcpy(_prev.get(), _current.get(), M_twice * sizeof(T));
        memset(_current.get(), 0, M_twice * sizeof(T));
    }
};
