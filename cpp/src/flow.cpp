#include "framehands/flow.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace framehands {
namespace {

constexpr int kBits = 14;
constexpr float kScale = 1.0f / (1 << 20);
constexpr int kMax = kFlowMaxWin;

inline int descale(int v, int n) { return (v + (1 << (n - 1))) >> n; }

inline int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
}

struct Weights {
    int w00, w01, w10, w11;
    Weights(float a, float b) {
        w00 = int(std::lrint((1.f - a) * (1.f - b) * (1 << kBits)));
        w01 = int(std::lrint(a * (1.f - b) * (1 << kBits)));
        w10 = int(std::lrint((1.f - a) * b * (1 << kBits)));
        w11 = (1 << kBits) - w00 - w01 - w10;
    }
};

// The (n x m) pixels from (x0, y0): the image itself, or a copy extended by reflection where it ends.
const std::uint8_t* pixels(const FlowLevel& im, int x0, int y0, int n, int m, std::uint8_t* buf,
                           std::ptrdiff_t& stride) {
    if (x0 >= 0 && y0 >= 0 && x0 + n <= im.w && y0 + m <= im.h) {
        stride = im.stride;
        return im.data + y0 * im.stride + x0;
    }
    for (int y = 0; y < m; ++y) {
        const std::uint8_t* r = im.data + reflect101(y0 + y, im.h) * im.stride;
        for (int x = 0; x < n; ++x) buf[y * n + x] = r[reflect101(x0 + x, im.w)];
    }
    stride = n;
    return buf;
}

// Bilinear window of the image at 5 fractional bits.
void window(const std::uint8_t* s, std::ptrdiff_t st, int win, const Weights& k, std::int16_t* out) {
    const int w00 = k.w00, w01 = k.w01, w10 = k.w10, w11 = k.w11;
    for (int y = 0; y < win; ++y, s += st, out += win) {
        const std::uint8_t* t = s + st;
        for (int x = 0; x < win; ++x)
            out[x] = std::int16_t(descale(s[x] * w00 + s[x + 1] * w01 + t[x] * w10 + t[x + 1] * w11, kBits - 5));
    }
}

void window16(const std::int16_t* s, int st, int win, const Weights& k, std::int16_t* out) {
    const int w00 = k.w00, w01 = k.w01, w10 = k.w10, w11 = k.w11;
    for (int y = 0; y < win; ++y, s += st, out += win) {
        const std::int16_t* t = s + st;
        for (int x = 0; x < win; ++x)
            out[x] = std::int16_t(descale(s[x] * w00 + s[x + 1] * w01 + t[x] * w10 + t[x + 1] * w11, kBits));
    }
}

// Scharr gradients of the (n x n) pixels from (x0, y0); zero outside the image.
void gradients(const FlowLevel& im, int x0, int y0, int n, std::int16_t* dx, std::int16_t* dy) {
    std::uint8_t buf[(kMax + 3) * (kMax + 3)];
    std::ptrdiff_t st;
    const std::uint8_t* s = pixels(im, x0 - 1, y0 - 1, n + 2, n + 2, buf, st);
    std::int16_t t0[kMax + 3], t1[kMax + 3];
    for (int y = 0; y < n; ++y, s += st) {
        const std::uint8_t *a = s, *b = s + st, *c = s + 2 * st;
        for (int x = 0; x < n + 2; ++x) {
            t0[x] = std::int16_t((a[x] + c[x]) * 3 + b[x] * 10);
            t1[x] = std::int16_t(c[x] - a[x]);
        }
        std::int16_t *gx = dx + y * n, *gy = dy + y * n;
        for (int x = 0; x < n; ++x) {
            gx[x] = std::int16_t(t0[x + 2] - t0[x]);
            gy[x] = std::int16_t((t1[x + 2] + t1[x]) * 3 + t1[x + 1] * 10);
        }
    }
    if (x0 >= 0 && y0 >= 0 && x0 + n <= im.w && y0 + n <= im.h) return;
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            if (x0 + x < 0 || x0 + x >= im.w || y0 + y < 0 || y0 + y >= im.h) dx[y * n + x] = dy[y * n + x] = 0;
}

}  // namespace

void flow_lk(const FlowPyramid& prev, const FlowPyramid& cur, const float* pts, int n, float* next,
             std::uint8_t* status, const FlowParams& p) {
    const int win = p.win, area = win * win, top = int(prev.size()) - 1;
    const float half = float(win - 1) * 0.5f, eps2 = p.epsilon * p.epsilon;
    alignas(64) std::int16_t I[kMax * kMax], Ix[kMax * kMax], Iy[kMax * kMax], J[kMax * kMax];
    std::int16_t dx[(kMax + 1) * (kMax + 1)], dy[(kMax + 1) * (kMax + 1)];
    std::uint8_t buf[(kMax + 1) * (kMax + 1)];
    std::ptrdiff_t st;
    for (int i = 0; i < n; ++i) {
        status[i] = 1;
        float nx = 0, ny = 0;
        for (int level = top; level >= 0; --level) {
            const FlowLevel &A = prev[level], &B = cur[level];
            const float sc = float(1.0 / (1 << level));
            float px = pts[2 * i] * sc, py = pts[2 * i + 1] * sc;
            if (level == top) {
                nx = px;
                ny = py;
            } else {
                nx *= 2.f;
                ny *= 2.f;
            }
            px -= half;
            py -= half;
            if (!(std::fabs(px) < 1e6f && std::fabs(py) < 1e6f && std::fabs(nx) < 1e6f && std::fabs(ny) < 1e6f)) {
                status[i] = 0;
                break;
            }
            const int ipx = int(std::floor(px)), ipy = int(std::floor(py));
            if (ipx < -win || ipx >= A.w || ipy < -win || ipy >= A.h) {
                if (level == 0) status[i] = 0;
                continue;
            }
            const Weights k(px - float(ipx), py - float(ipy));
            const std::uint8_t* a = pixels(A, ipx, ipy, win + 1, win + 1, buf, st);
            window(a, st, win, k, I);
            gradients(A, ipx, ipy, win + 1, dx, dy);
            window16(dx, win + 1, win, k, Ix);
            window16(dy, win + 1, win, k, Iy);
            std::int64_t s11 = 0, s12 = 0, s22 = 0;
            for (int y = 0; y < win; ++y) {  // a row's sums fit 32 bits
                std::int32_t r11 = 0, r12 = 0, r22 = 0;
                const std::int16_t *gx = Ix + y * win, *gy = Iy + y * win;
                for (int x = 0; x < win; ++x) {
                    r11 += gx[x] * gx[x];
                    r12 += gx[x] * gy[x];
                    r22 += gy[x] * gy[x];
                }
                s11 += r11;
                s12 += r12;
                s22 += r22;
            }
            const float A11 = float(s11) * kScale, A12 = float(s12) * kScale, A22 = float(s22) * kScale;
            float D = A11 * A22 - A12 * A12;
            const float eig = (A22 + A11 - std::sqrt((A11 - A22) * (A11 - A22) + 4.f * A12 * A12)) / float(2 * area);
            if (eig < p.min_eig || D < FLT_EPSILON) {
                if (level == 0) status[i] = 0;
                continue;
            }
            D = 1.f / D;
            float qx = nx - half, qy = ny - half, ox = 0, oy = 0;
            for (int j = 0; j < p.iterations; ++j) {
                const int iqx = int(std::floor(qx)), iqy = int(std::floor(qy));
                if (iqx < -win || iqx >= B.w || iqy < -win || iqy >= B.h) {
                    if (level == 0) status[i] = 0;
                    break;
                }
                const std::uint8_t* b = pixels(B, iqx, iqy, win + 1, win + 1, buf, st);
                window(b, st, win, Weights(qx - float(iqx), qy - float(iqy)), J);
                std::int64_t s1 = 0, s2 = 0;
                for (int y = 0; y < win; ++y) {
                    std::int32_t r1 = 0, r2 = 0;
                    const std::int16_t *u = I + y * win, *v = J + y * win, *gx = Ix + y * win, *gy = Iy + y * win;
                    for (int x = 0; x < win; ++x) {
                        const std::int32_t d = std::int16_t(v[x] - u[x]);
                        r1 += d * gx[x];
                        r2 += d * gy[x];
                    }
                    s1 += r1;
                    s2 += r2;
                }
                const float b1 = float(s1) * kScale, b2 = float(s2) * kScale;
                const float ex = (A12 * b2 - A22 * b1) * D, ey = (A12 * b1 - A11 * b2) * D;
                if (!(std::fabs(qx + ex) < 1e6f && std::fabs(qy + ey) < 1e6f)) {
                    if (level == 0) status[i] = 0;
                    break;
                }
                qx += ex;
                qy += ey;
                nx = qx + half;
                ny = qy + half;
                if (ex * ex + ey * ey <= eps2) break;
                if (j > 0 && std::fabs(ex + ox) < 0.01f && std::fabs(ey + oy) < 0.01f) {
                    nx -= ex * 0.5f;
                    ny -= ey * 0.5f;
                    break;
                }
                ox = ex;
                oy = ey;
            }
        }
        if (status[i] && !(std::fabs(nx) < 1e6f && std::fabs(ny) < 1e6f)) status[i] = 0;
        if (status[i]) {
            const int fx = int(std::floor(nx - half)), fy = int(std::floor(ny - half));
            if (fx < -win || fx >= cur[0].w || fy < -win || fy >= cur[0].h) status[i] = 0;
        }
        next[2 * i] = nx;
        next[2 * i + 1] = ny;
    }
}

}  // namespace framehands
