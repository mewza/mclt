/**
 ***    MCLTReal v1.50
 ***    mclt_neon.h -- DO NOT include this file, instead only include mclt.h
 ***
 ***    Features include:
 ***        • Hybrid TDAC/COLA-based scaling with explicit overlap compensation
 ***        • Supports arbitrary HOP sizes with proper WOLA (Weighted Overlap-Add)
 ***        • Uses FFT-to-MCLT mapping for efficiency (uses FFTReal highly optimized and accurate FFT)
 ***        • 3-buffer architecture for flexible overlap
 ***        • Optimized overlap-add for common hop sizes
 ***        • NEON optimizations
 ***
 ***    @contact  E-mail: subband@gmail.com
 ***    @home https://github.com/mewza/MCLT/
 ***
 ***    NOTE: This software uses FFTReal class to accelerate FFT and can be obtained from:
 ***    https://github.com/mewza/realfft/
**/

#pragma once

#ifndef MCLT_DO_NOT_WARN

#pragma error("DO NOT include this file directly, include mclt.h instead!")

#else

#if USE_NEON && !TARGET_OS_MACCATALYST && TARGET_CPU_ARM64 && defined(__ARM_NEON)
#include <arm_neon.h>
#define MCLT_HAS_NEON 1

#pragma warning("Compiling with NEON optimizations")

#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)

template<typename T, bool useFFTReal = true>
class MCLTReal {
public:
    using T1 = SimdBase<T>;
    using cmplxTT = cmplxT<T>;
    
private:
    static constexpr size_t ALIGNMENT = 128;
    static constexpr int PREFETCH_DISTANCE = 16;
    
    int _length;
    int _M;
    int _hop;
    
    struct AlignedDeleterT { void operator()(T* ptr) const { free(ptr); } };
    struct AlignedDeleterT1 { void operator()(T1* ptr) const { free(ptr); } };
    struct AlignedDeleterCmplxTT { void operator()(cmplxTT* ptr) const { free(ptr); } };
    
    // Beam-style 3-buffer architecture for flexible overlap
    std::unique_ptr<T[], AlignedDeleterT> _prev_prev;   // Frame at t-2
    std::unique_ptr<T[], AlignedDeleterT> _prev;        // Frame at t-1
    std::unique_ptr<T[], AlignedDeleterT> _current;     // Frame at t
    
    std::unique_ptr<cmplxTT[], AlignedDeleterCmplxTT> _temp_complex;
    std::unique_ptr<T[], AlignedDeleterT> _temp_real;
    std::unique_ptr<T1[], AlignedDeleterT1> _window;
    
    WindowType _window_type;
    
    std::conditional_t<useFFTReal, FFTReal<T>, char> _fft;
    
    // Scaling factors
    T1 _analysis_scale;
    T1 _synthesis_scale;
    T1 _overlap_compensation;  // Additional factor for non-50% overlap
    
public:
    explicit MCLTReal(int length, int hop = -1, WindowType window_type = WINTYPE_SINE)
        : _length(length)
        , _M(length / 2)
        , _hop(hop < 0 ? (_length / 2) : hop)
        , _window_type(window_type)
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
        
        // ✅ HYBRID SCALING APPROACH:
        // Base MCLT normalization (independent of hop)
        _analysis_scale = F_SQRT(2.0 / (T1)_length);
        
        // For synthesis, we use TWO factors:
        // 1. Base MCLT inverse: sqrt(length/2)
        // 2. TDAC compensation: 0.5 for 50% overlap
        // Combined: sqrt(length/2) * 0.5 = sqrt(length/8)
        _synthesis_scale = F_SQRT((T1)_length / 8.0);
        
        // 3. Overlap compensation for non-50% hop sizes
        T1 overlap_factor = (T1)_length / (T1)_hop;
        if (overlap_factor != 2.0) {
            _overlap_compensation = F_SQRT(2.0 / overlap_factor);
        } else {
            _overlap_compensation = 1.0;
        }
        
        init_window();
        reset();
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
    inline T1* window() const { return _window.get(); }
    inline int get_M() const { return _M; }
    
    void set_hop(int hop) {
        if (hop <= 0 || hop > _length) {
            throw std::invalid_argument("Hop size must be between 1 and 2M");
        }
        _hop = hop;
        
        // Recalculate overlap compensation
        T1 overlap_factor = (T1)_length / (T1)_hop;
        _overlap_compensation = (overlap_factor != 2.0)
            ? F_SQRT(2.0 / overlap_factor)
            : 1.0;
        
        reset();
    }
   
    // ========== FORWARD MCLT (Analysis) ==========
    
    void analyze(const T* __restrict__ x, cmplxTT* __restrict__ X, bool apply_window = true) {
        const int M = _M;
        const int N = 2 * M;
        
        T* fft_buf = _temp_real.get();
        const T1* h = _window.get();
        
        if (apply_window) {
            // Windowing with NEON prefetch optimization
#pragma clang loop unroll_count(4)
            for (int i = 0; i < N; i++) {
                if (i + PREFETCH_DISTANCE < N) {
                    __builtin_prefetch(&x[i + PREFETCH_DISTANCE], 0, 1);
                    __builtin_prefetch(&h[i + PREFETCH_DISTANCE], 0, 1);
                    __builtin_prefetch(&fft_buf[i + PREFETCH_DISTANCE], 1, 1);
                }
                fft_buf[i] = x[i] * h[i];
            }
        } else {
            memcpy(fft_buf, x, N * sizeof(T));
        }
        
        // FFT
        _fft.do_fft(fft_buf, fft_buf);
        
        // Unpack complex with prefetch
        cmplxTT* fft_out = _temp_complex.get();
#pragma clang loop unroll_count(4)
        for (int i = 0; i < M; i++) {
            if (i + PREFETCH_DISTANCE < M) {
                __builtin_prefetch(&fft_buf[2*(i + PREFETCH_DISTANCE)], 0, 1);
                __builtin_prefetch(&fft_out[i + PREFETCH_DISTANCE], 1, 1);
            }
            fft_out[i].re = fft_buf[2*i];
            fft_out[i].im = fft_buf[2*i+1];
        }
        
        // FFT-to-MCLT mapping
        const T uL = 0.5 / M;
        const T g_angle = M_PI * (0.5 + uL);
        const T cstep = F_COS(g_angle);
        const T sstep = F_SIN(g_angle);
        const T g = F_SQRT(uL);
        
        T ca = F_COS(M_PI / 4.0);
        T sa = -ca;
        
        T r0 = fft_out[0].re * ca;
        T i0 = fft_out[0].re * sa;
        
        // Rotation loop with prefetch
#pragma clang loop unroll_count(4)
        for (int k = 0; k < M - 1; k++) {
            if (k + PREFETCH_DISTANCE < M - 1) {
                __builtin_prefetch(&fft_out[k + 1 + PREFETCH_DISTANCE], 0, 1);
                __builtin_prefetch(&X[k + PREFETCH_DISTANCE], 1, 1);
            }
            
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
        
        // Apply analysis scale with prefetch (independent of hop)
#pragma clang loop unroll_count(8)
        for (int k = 0; k < M; k++) {
            if (k + PREFETCH_DISTANCE < M) {
                __builtin_prefetch(&X[k + PREFETCH_DISTANCE], 1, 1);
            }
            X[k] *= _analysis_scale;
        }
    }
    
    // ========== INVERSE MCLT (Synthesis) ==========
    
    void synthesize(const cmplxTT* __restrict__ X, bool apply_window = true) {
        const int M = _M;
        const int N = 2 * M;
        
        cmplxTT* y = _temp_complex.get();
        
        // Extract coefficients with prefetch
#pragma clang loop unroll_count(4)
        for (int k = 0; k < M; k++) {
            if (k + PREFETCH_DISTANCE < M) {
                __builtin_prefetch(&X[k + PREFETCH_DISTANCE], 0, 1);
                __builtin_prefetch(&y[k + PREFETCH_DISTANCE], 1, 1);
            }
            y[k] = X[k];
        }
        
        // IMCLT-to-IDFT mapping
        T* t = _temp_real.get();
        
        const T uL = 0.5 / M;
        const T g_angle = M_PI * (0.5 + uL);
        const T cstep = F_COS(g_angle);
        const T sstep = F_SIN(-g_angle);
        
        T ca = F_COS(M_PI / 4.0);
        T sa = ca;
        
        // Rotation loop with prefetch
#pragma clang loop unroll_count(4)
        for (int k = 1; k < M; k++) {
            if (k + PREFETCH_DISTANCE < M) {
                __builtin_prefetch(&y[k + PREFETCH_DISTANCE], 0, 1);
                __builtin_prefetch(&t[(k + PREFETCH_DISTANCE) * 2], 1, 1);
            }
            
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
        
        // Mirror second half with prefetch
        int k = M - 1;
#pragma clang loop unroll_count(4)
        for (int j = M + 1; j < 2 * M; j++) {
            if (j + PREFETCH_DISTANCE < 2 * M) {
                __builtin_prefetch(&t[(k - PREFETCH_DISTANCE) * 2], 0, 1);
                __builtin_prefetch(&t[(j + PREFETCH_DISTANCE) * 2], 1, 1);
            }
            t[j * 2] = t[k * 2];
            t[j * 2 + 1] = -t[k * 2 + 1];
            k--;
        }
        
        // IFFT
        _fft.do_ifft(t, _temp_real.get());
        
        // Apply synthesis scale with hop compensation
        T* current = _current.get();
        const T1 scale = _synthesis_scale * _overlap_compensation;
        
        if (apply_window) {
            const T1* h = _window.get();
#pragma clang loop unroll_count(4)
            for (int j = 0; j < N; j++) {
                if (j + PREFETCH_DISTANCE < N) {
                    __builtin_prefetch(&_temp_real[j + PREFETCH_DISTANCE], 0, 1);
                    __builtin_prefetch(&h[j + PREFETCH_DISTANCE], 0, 1);
                    __builtin_prefetch(&current[j + PREFETCH_DISTANCE], 1, 1);
                }
                current[j] = _temp_real[j] * scale * h[j];
            }
        } else {
#pragma clang loop unroll_count(4)
            for (int j = 0; j < N; j++) {
                if (j + PREFETCH_DISTANCE < N) {
                    __builtin_prefetch(&_temp_real[j + PREFETCH_DISTANCE], 0, 1);
                    __builtin_prefetch(&current[j + PREFETCH_DISTANCE], 1, 1);
                }
                current[j] = _temp_real[j] * scale;
            }
        }
    }
    
    void synthesize_no_tdac(const cmplxTT* __restrict__ X, T* __restrict__ output) {
        synthesize(X);
        memcpy(output, _current.get(), _length * sizeof(T));
    }
    
    // ========== OPTIMIZED OVERLAP-ADD (Beam-style with NEON) ==========
    
    void overlap_add_to_buffer(T* __restrict__ output) {
        const int M = _M;
        const int L = _length;
        const int hop = _hop;
        
        T* pp = _prev_prev.get();
        T* p = _prev.get();
        T* c = _current.get();
        
        // Optimized paths for common hop sizes with prefetch
        if (hop == M) {
            // ✅ Standard 50% overlap (most common, fastest)
#pragma clang loop unroll_count(8)
            for (int i = 0; i < M; i++) {
                if (i + PREFETCH_DISTANCE < M) {
                    __builtin_prefetch(&pp[i + M + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&p[i + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&output[i + PREFETCH_DISTANCE], 1, 0);
                }
                output[i] = pp[i + M] + p[i];
            }
#pragma clang loop unroll_count(8)
            for (int i = M; i < L; i++) {
                if (i + PREFETCH_DISTANCE < L) {
                    __builtin_prefetch(&c[i - M + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&p[i + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&output[i + PREFETCH_DISTANCE], 1, 0);
                }
                output[i] = c[i - M] + p[i];
            }
        }
        else if (hop == M / 2) {
            // ✅ 75% overlap (hop = M/2)
#pragma clang loop unroll_count(8)
            for (int i = 0; i < hop; i++) {
                if (i + PREFETCH_DISTANCE < hop) {
                    __builtin_prefetch(&pp[i + M + hop + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&p[i + hop + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&c[i + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&output[i + PREFETCH_DISTANCE], 1, 0);
                }
                output[i] = pp[i + M + hop] + p[i + hop] + c[i];
            }
        }
        else if (hop == M / 4) {
            // ✅ 87.5% overlap (hop = M/4)
#pragma clang loop unroll_count(8)
            for (int i = 0; i < hop; i++) {
                if (i + PREFETCH_DISTANCE < hop) {
                    __builtin_prefetch(&c[i + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&p[i + hop + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&output[i + PREFETCH_DISTANCE], 1, 0);
                }
                // Sum contributions from 4 overlapping frames
                T sum = 0;
                if (i < L) sum += c[i];
                if (i + hop < L) sum += p[i + hop];
                if (i + 2*hop < L) sum += p[i + 2*hop];
                if (i + 3*hop < L) sum += pp[i + 3*hop];
                output[i] = sum;
            }
        }
        else if (hop == 3 * M / 4) {
            // ✅ 25% overlap (hop = 3M/4)
            const int M_half = M / 2;
            
#pragma clang loop unroll_count(8)
            for (int i = 0; i < M_half; i++) {
                if (i + PREFETCH_DISTANCE < M_half) {
                    __builtin_prefetch(&pp[i + M_half + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&p[i + M + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&output[i + PREFETCH_DISTANCE], 1, 0);
                }
                output[i] = pp[i + M_half] + p[i + M];
            }
            
#pragma clang loop unroll_count(8)
            for (int i = M_half; i < M; i++) {
                if (i + PREFETCH_DISTANCE < M) {
                    __builtin_prefetch(&p[i + M + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&output[i + PREFETCH_DISTANCE], 1, 0);
                }
                output[i] = p[i + M];
            }
            
#pragma clang loop unroll_count(8)
            for (int i = M; i < hop; i++) {
                if (i + PREFETCH_DISTANCE < hop) {
                    __builtin_prefetch(&c[i - M + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&p[i + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&output[i + PREFETCH_DISTANCE], 1, 0);
                }
                output[i] = c[i - M] + p[i];
            }
        }
        else {
            // ⚠️ Generic arbitrary hop with prefetch
            const int max_overlap_frames = (L + hop - 1) / hop;
            
#pragma clang loop unroll_count(4)
            for (int i = 0; i < hop && i < L; i++) {
                if (i + PREFETCH_DISTANCE < hop && i + PREFETCH_DISTANCE < L) {
                    __builtin_prefetch(&c[i + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&p[i + hop + PREFETCH_DISTANCE], 0, 0);
                    __builtin_prefetch(&output[i + PREFETCH_DISTANCE], 1, 0);
                }
                
                T sum = 0;
                
                // Current frame contribution
                if (i < L) {
                    sum += c[i];
                }
                
                // Previous frame contribution
                int p_idx = i + hop;
                if (p_idx < L) {
                    sum += p[p_idx];
                }
                
                // Previous-previous frame contribution
                int pp_idx = i + 2 * hop;
                if (pp_idx < L) {
                    sum += pp[pp_idx];
                }
                
                output[i] = sum;
            }
        }
        
        // Shift frame buffers
        memcpy(pp, p, L * sizeof(T));
        memcpy(p, c, L * sizeof(T));
        memset(c, 0, L * sizeof(T));
    }
    
    // ========== API ALIASES ==========
    
    void real_mclt(const T* in, cmplxTT* out) {
        analyze(in, out);
    }
    
    void mclt(const T* in, cmplxTT* out) {
        analyze(in, out);
    }
    
    void real_imclt(const cmplxTT* in, T* out) {
        synthesize(in);
        overlap_add_to_buffer(out);
    }
    
    void imclt(const cmplxTT* in, T* out) {
        synthesize(in);
        overlap_add_to_buffer(out);
    }
    
    void skip_frame() {
        memset(_current.get(), 0, _length * sizeof(T));
        T* pp = _prev_prev.get();
        T* p = _prev.get();
        T* c = _current.get();
        memcpy(pp, p, _length * sizeof(T));
        memcpy(p, c, _length * sizeof(T));
    }
    
private:
    
    void init_window() {
        const int N = _length;
        T1* w = _window.get();
            
        switch (_window_type) {
            case WINTYPE_SINE: {
                // Sine window (TDAC-compliant for MCLT) with unrolling
                const T1 K = M_PI / T1(N);
#pragma clang loop unroll_count(4)
                for (int i = 0; i < N; i++) {
                    w[i] = F_SIN((i + 0.5) * K);
                }
                break;
            }
            
            case WINTYPE_KAISER: {
                // Kaiser-Bessel Derived (AAC standard, TDAC-compliant)
                generate_kbd_window(w, N, 4.0);
                break;
            }
            
            case WINTYPE_HANNING: {
                // Hann window with unrolling
#pragma clang loop unroll_count(4)
                for (int i = 0; i < N; i++) {
                    w[i] = 0.5 * (1.0 - F_COS(2.0 * M_PI * i / (T1)(N - 1)));
                }
                break;
            }
            
            default:
                // Fallback to sine with unrolling
                const T1 K = M_PI / T1(N);
#pragma clang loop unroll_count(4)
                for (int i = 0; i < N; i++) {
                    w[i] = F_SIN(T1(i + 0.5) * K);
                }
                break;
        }
    }
    
    static inline void generate_kbd_window(T1* window, int length, T1 alpha = 4.0) {
        const int N = length;
        const int N2 = N / 2;
        
        T1* kaiser = new T1[N2 + 1];
        T1 i0_alpha = bessel_i0(M_PI * alpha);
        
        for (int i = 0; i <= N2; i++) {
            T1 x = 2.0 * i / (T1)N - 1.0;
            T1 arg = M_PI * alpha * F_SQRT(1.0 - x * x);
            kaiser[i] = bessel_i0(arg) / i0_alpha;
        }
        
        T1* cum_sum = new T1[N2 + 1];
        cum_sum[0] = kaiser[0];
        for (int i = 1; i <= N2; i++) {
            cum_sum[i] = cum_sum[i-1] + kaiser[i];
        }
        
        T1 total = cum_sum[N2];
        
        for (int i = 0; i < N2; i++) {
            window[i] = F_SQRT(cum_sum[i] / total);
        }
        
        for (int i = N2; i < N; i++) {
            window[i] = F_SQRT(cum_sum[N - i] / total);
        }
        
        delete[] kaiser;
        delete[] cum_sum;
    }
    
    static inline T1 bessel_i0(T1 x) {
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
    
    static inline T* aligned_alloc_T(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T)) != 0)
            return nullptr;
        return static_cast<T*>(ptr);
    }
    
    static inline cmplxTT* aligned_alloc_cmplxTT(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(cmplxTT)) != 0)
            return nullptr;
        return static_cast<cmplxTT*>(ptr);
    }
    
    static inline T1* aligned_alloc_T1(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, count * sizeof(T1)) != 0)
            return nullptr;
        return static_cast<T1*>(ptr);
    }
};

#endif // USE_NEON

#endif // MCLT_DO_NOT_WARN
