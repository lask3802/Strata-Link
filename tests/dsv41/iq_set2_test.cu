// DS4_IQ_SET2 (ds4_iq.cu set2_tile_kernel): the chunked prompt's expert sets (iq::set_gate_up / set_down) with set2_tile_kernel at
// row blocks of 64 and 128 against set_tc_kernel (set2_kernel() 0), bit for bit, for each type of the packs (IQ1_M,
// IQ2_XXS, IQ2_XS) x gate / up and down x row mixes: <= 16 (the light kernels' alone: set2 not launched), 17-40, 64, 65-128,
// > 128 (several row blocks), a 4K and a 30K prompt's mixes (with experts of a few rows: skip_le), 20 experts x 499 rows (past
// set2's own 64 MB scratch: set_tc_kernel's without the engine's room), the rows at a row0 not 0; gate / up both with the
// engine's room for the gathered input (Set2Room: xg, nrows) and without it (set2's scratch); set2_gathers (whether the
// engine allocates that room): none off, the three types on.
// --bench [--runs N]: us an expert, current / BM 64 / BM 128 (CUDA events, the median of N after 3 warm-ups).  Needs a GPU, no
// model.  ds4_iq_set2_test_k8: Turing's m16n8k8 path on any GPU; ds4_iq_set2_test_none: DS4_IQ_SET2_FORCE_NONE (an sm_60 /
// sm_70 build's stand-in: never launched, set_tc_kernel's bits).  Exit 0 PASS, 1 FAIL, 2 bad arguments.
#include "../../src/dsv41/ds4_iq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace iq = strata::ds4::gguf::iq;

namespace {

#define CU(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

constexpr int D = 5120, F = 2304, NE = 16, NB = 20, kRow0 = 37;   // (NB: the blobs, the largest mix's experts)

uint16_t f16(float v) {
    const __half h = __float2half_rn(v);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
// a matrix of n rows of k values of the type: random codes, scales and signs, the F16 block scale +-0.002 .. 0.02 (a block
// in 64: +-0, the decode's sign by XOR there)
void fill(int type, uint8_t* m, int n, int k, std::mt19937& g) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> dd(0.002f, 0.02f);
    const int bb = iq::block_bytes(type), nb = k / 256;
    for (size_t i = 0; i < (size_t) n * nb; ++i) {
        uint8_t* p = m + i * bb;
        for (int j = 0; j < bb; ++j) p[j] = (uint8_t) byte(g);
        const uint32_t r = (uint32_t) g();
        const uint16_t d = (uint16_t) ((r % 64 == 0 ? 0u : f16(dd(g))) | (r & 64u ? 0x8000u : 0u));
        if (type == iq::kIQ1_M) {   // (d: the top nibbles of the four u16 scales at 48)
            for (int q = 0; q < 4; ++q) {
                uint16_t s = (uint16_t) (p[48 + 2 * q] | p[49 + 2 * q] << 8);
                s = (uint16_t) ((s & 0x0fffu) | (((d >> (4 * q)) & 15u) << 12));
                p[48 + 2 * q] = (uint8_t) (s & 255u);
                p[49 + 2 * q] = (uint8_t) (s >> 8);
            }
        } else {   // IQ2_XXS / IQ2_XS: d first
            p[0] = (uint8_t) (d & 255u);
            p[1] = (uint8_t) (d >> 8);
        }
    }
}

template <class X>
X* up(const std::vector<X>& v) {
    X* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(X)));
    CU(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice));
    return p;
}
template <class X>
std::vector<X> get(const X* p, size_t n) {
    std::vector<X> v(n);
    CU(cudaMemcpy(v.data(), p, n * sizeof(X), cudaMemcpyDeviceToHost));
    return v;
}
float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Mix {
    const char* name;
    std::vector<int> rows;   // an expert's
    bool tiles;              // whether an expert has more than 16 rows (set2 launched)
};
std::vector<Mix> mixes(std::mt19937& g) {
    auto uni = [&](int a, int b) {
        std::uniform_int_distribution<int> u(a, b);
        std::vector<int> r(NE);
        for (int& x : r) x = u(g);
        return r;
    };
    std::vector<Mix> m;
    m.push_back({"<= 16 (light only)", uni(1, 16), false});
    m.push_back({"17-40", uni(17, 40), true});
    m.push_back({"64", std::vector<int>(NE, 64), true});
    m.push_back({"65-128", uni(65, 128), true});
    m.push_back({"> 128 (129-300)", uni(129, 300), true});
    std::vector<int> k4 = uni(33, 99), k30 = uni(60, 182);   // (means about 66 and 121, as r10's 4K and 30K prompts')
    k4[1] = 5, k4[2] = 16, k4[3] = 17;                        // (the light kernels' experts among them: skip_le)
    k30[1] = 3, k30[2] = 200, k30[3] = 129;
    m.push_back({"4K-like (mean 66)", k4, true});
    m.push_back({"30K-like (mean 121)", k30, true});
    m.push_back({"20 x 499 (past 64 MB)", std::vector<int>(NB, 499), true});
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int runs = 25;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else {
            std::fprintf(stderr, "usage: %s [--bench [--runs N]]\n", argv[0]);
            return 2;
        }
    }
    cudaDeviceProp prop{};
    CU(cudaGetDeviceProperties(&prop, 0));
    std::printf("device %s, sm_%d%d, %d SMs%s\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount,
#if defined(DS4_IQ_SET2_K8)
                " (m16n8k8 forced)"
#else
                ""
#endif
    );
    if (prop.major * 10 + prop.minor < 75) {
        std::printf("SKIP: set2_tile_kernel needs sm_75\n");
        return 0;
    }
    cudaStream_t s;
    CU(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    int fails = 0, cases = 0;
    std::mt19937 g(20261028);
    {   // set2_gathers (the engine's allocation of the room, alloc_chunks): off - none; on - the pack's types alone
        int bad = 0;
        iq::set2_kernel() = 0;
        for (int t : {iq::kIQ1_M, iq::kIQ2_XXS, iq::kIQ2_XS, 39}) bad += iq::set2_gathers(t) ? 1 : 0;
        iq::set2_kernel() = 64;
#if defined(DS4_IQ_SET2_FORCE_NONE)
        const bool want = false;   // (not taken: no room)
#else
        const bool want = true;
#endif
        for (int t : {iq::kIQ1_M, iq::kIQ2_XXS, iq::kIQ2_XS}) bad += iq::set2_gathers(t) != want ? 1 : 0;
        bad += iq::set2_gathers(39) ? 1 : 0;   // (MXFP4: not set2's)
        iq::set2_kernel() = 0;
        std::printf("set2_gathers: off none, on the three types%s  %s\n", want ? "" : " (none: not taken)", bad ? "FAIL" : "PASS");
        fails += bad ? 1 : 0;
        ++cases;
    }
    const std::vector<Mix> mx = mixes(g);
    const int types[3] = {iq::kIQ1_M, iq::kIQ2_XXS, iq::kIQ2_XS};
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int type : types) {
        // the experts' blobs: gate [F][D], up [F][D], down [D][F]
        const size_t rbg = iq::row_bytes(type, D), rbd = iq::row_bytes(type, F);
        const size_t tr1 = (size_t) F * rbg, tr2 = 2 * tr1, stride = tr2 + (size_t) D * rbd;
        std::vector<uint8_t> wb(stride * NB);
        for (int e = 0; e < NB; ++e) {
            fill(type, wb.data() + e * stride, F, D, g);
            fill(type, wb.data() + e * stride + tr1, F, D, g);
            fill(type, wb.data() + e * stride + tr2, D, F, g);
        }
        uint8_t* dw = up(wb);
        for (const Mix& m : mx) {
            std::vector<int> bounds(1, kRow0);
            int max_rows = 0;
            for (int r : m.rows) {
                bounds.push_back(bounds.back() + r);
                max_rows = std::max(max_rows, r);
            }
            const int rows = bounds.back() - kRow0;
            std::vector<int> idx(rows);   // (local row -> x row: a permutation of x's rows)
            for (int i = 0; i < rows; ++i) idx[i] = i;
            std::shuffle(idx.begin(), idx.end(), g);
            std::vector<float> x((size_t) rows * D);
            for (float& v : x) v = nd(g) * (g() % 64 == 0 ? 8.0f : 1.0f);
            std::vector<uint16_t> h((size_t) rows * F);
            for (uint16_t& v : h) v = f16(nd(g) * 0.5f);
            float* dx = up(x);
            uint16_t* dh = up(h);
            int* db = up(bounds);
            int* di = up(idx);
            uint16_t *dho, *dxg;
            float* dyo;
            CU(cudaMalloc(&dxg, (size_t) rows * D * 2));   // (the engine's room: the set's rows)
            CU(cudaMalloc(&dho, h.size() * 2));
            CU(cudaMalloc(&dyo, (size_t) rows * D * 4));
            iq::SetArgs a;
            a.wbase = dw;
            a.stride = stride;
            a.tr[0] = 0, a.tr[1] = tr1, a.tr[2] = tr2;
            a.bounds = db;
            a.row0 = kRow0;
            a.max_rows = max_rows;
            a.n_exp = (int) m.rows.size();
            a.n_embd = D;
            a.n_ff = F;
            a.limit = 10.0f;
            a.tg = a.td = type;
            iq::SetArgs gu = a, dn = a;
            gu.xf = dx, gu.idx = di, gu.ldx = D, gu.h = dho, gu.ldo = F;
            dn.xh = dh, dn.ldx = F, dn.y = dyo, dn.ldo = D;
            const iq::Set2Room room{dxg, rows};   // (as the engine's run_set: its room and the set's rows)
            // set2's own scratch takes a set of n_exp x max_rows rows up to 64 MB (set_tc_kernel's past it)
            const bool fits = (size_t) a.n_exp * max_rows * D * 2 <= ((size_t) 64 << 20);
            auto run = [&](int mode, bool gate_up, bool xg) {
                iq::set2_kernel() = mode;
                const bool ok = gate_up ? iq::set_gate_up(gu, s, xg ? &room : nullptr) : iq::set_down(dn, s);
                if (!ok) {
                    std::fprintf(stderr, "set_%s: not covered\n", gate_up ? "gate_up" : "down");
                    std::exit(1);
                }
            };
            for (int pass = 0; pass < 3; ++pass) {   // gate / up with the engine's room, without it; down
                const bool gate_up = pass < 2, xg = pass == 0;
                const char* role = pass == 0 ? "gate+up" : pass == 1 ? "gu (own)" : "down";
                const size_t nb = gate_up ? h.size() * 2 : (size_t) rows * D * 4;
                void* out = gate_up ? (void*) dho : (void*) dyo;
                CU(cudaMemset(out, 0xa5, nb));
                const unsigned long long z64 = iq::set2_runs(64), z128 = iq::set2_runs(128);
                CU(cudaMemset(dxg, 0xff, (size_t) rows * D * 2));   // (NaNs: a row the gather did not write shows)
                run(0, gate_up, xg);   // (off: set_tc_kernel's - set2 not launched)
                CU(cudaStreamSynchronize(s));
                CU(cudaGetLastError());
                const std::vector<uint8_t> ref = get((const uint8_t*) out, nb);
                const bool off_ok = iq::set2_runs(64) == z64 && iq::set2_runs(128) == z128;
                for (int mode : {64, 128}) {
                    CU(cudaMemset(out, 0x5a, nb));   // (a pattern unlike the reference's: a row not written shows)
                    const unsigned long long n0 = iq::set2_runs(mode), o0 = iq::set2_runs(mode == 64 ? 128 : 64);
                    run(mode, gate_up, xg);
                    CU(cudaStreamSynchronize(s));
                    CU(cudaGetLastError());
                    const unsigned long long ran = iq::set2_runs(mode) - n0, other = iq::set2_runs(mode == 64 ? 128 : 64) - o0;
                    const std::vector<uint8_t> got = get((const uint8_t*) out, nb);
                    const size_t es = gate_up ? 2 : 4;
                    size_t diff = 0, first = (size_t) -1;
                    for (size_t i = 0; i < nb / es; ++i)
                        if (std::memcmp(&got[i * es], &ref[i * es], es) != 0) {
                            if (diff++ == 0) first = i;
                        }
#if defined(DS4_IQ_SET2_FORCE_NONE)
                    const bool launched_ok = ran == 0 && other == 0;   // (not taken: set_tc_kernel)
#else
                    const bool launched_ok = (m.tiles && (xg || !gate_up || fits) ? ran == 1 : ran == 0) && other == 0;
#endif
                    const bool ok = diff == 0 && launched_ok && off_ok;
                    ++cases;
                    fails += ok ? 0 : 1;
                    std::printf("%-8s %-7s %-22s BM %3d: %8zu of %9zu differ%s, set2 launched %llu%s  %s\n", iq::name(type), role, m.name,
                                mode, diff, nb / es,
                                diff ? (" (first: local row " + std::to_string(first / (gate_up ? F : D)) + ")").c_str() : "",
                                ran, off_ok && other == 0 ? "" : " (and with it off / at the other row block)", ok ? "PASS" : "FAIL");
                }
                if (bench && m.tiles && pass != 1) {
                    auto timed = [&](int mode) {
                        for (int i = 0; i < 3; ++i) run(mode, gate_up, xg);
                        std::vector<float> t;
                        for (int i = 0; i < runs; ++i) {
                            CU(cudaEventRecord(e0, s));
                            run(mode, gate_up, xg);
                            CU(cudaEventRecord(e1, s));
                            CU(cudaEventSynchronize(e1));
                            float ms = 0;
                            CU(cudaEventElapsedTime(&ms, e0, e1));
                            t.push_back(ms * 1e3f / a.n_exp);
                        }
                        return median(t);
                    };
                    const float c = timed(0), b64 = timed(64), b128 = timed(128);
                    std::printf("bench %-8s %-7s %-22s us/expert: current %7.1f  BM64 %7.1f (x%.2f)  BM128 %7.1f (x%.2f)\n",
                                iq::name(type), role, m.name, c, b64, c / b64, b128, c / b128);
                }
            }
            CU(cudaFree(dx));
            CU(cudaFree(dh));
            CU(cudaFree(db));
            CU(cudaFree(di));
            CU(cudaFree(dho));
            CU(cudaFree(dxg));
            CU(cudaFree(dyo));
        }
        CU(cudaFree(dw));
    }
    iq::set2_kernel() = 0;
    std::printf("%s (%d of %d cases failed)\n", fails ? "FAIL" : "PASS", fails, cases);
    return fails ? 1 : 0;
}
