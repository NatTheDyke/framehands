// framehands geometry core; see include/framehands/core.hpp. Each function follows the Python
// named in its comment, in the same evaluation order (numpy reductions of 3 or 4 values are
// sequential; sums of 8 or more follow numpy's pairwise summation).
#include "framehands/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace framehands {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;  // math.radians / np.radians

inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline double norm(const Vec3& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
inline Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 add(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline Vec3 mean4(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    Vec3 out;
    for (int i = 0; i < 3; ++i) out[i] = (((a[i] + b[i]) + c[i]) + d[i]) / 4;
    return out;
}

// numpy's pairwise sum for n <= 128 (eight accumulators, then the remainder)
template <typename T>
T pairwise_sum(const T* a, int n) {
    if (n < 8) {
        T res = a[0];
        for (int i = 1; i < n; ++i) res += a[i];
        return res;
    }
    T r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    int i = 8;
    for (; i < n - (n % 8); i += 8)
        for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
}

}  // namespace

// ---------------------------------------------------------------- calib.py
double KBCamera::distort(double th) const {
    double t2 = th * th;
    return th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
}

void KBCamera::project(const double* P, int n, double* uv) const {
    for (int i = 0; i < n; ++i) {
        double x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
        double r = std::hypot(x, y);
        double d = distort(std::atan2(r, z));
        double s = r > 1e-12 ? d / std::max(r, 1e-12) : 0.0;
        uv[2 * i] = fx * x * s + cx;
        uv[2 * i + 1] = fy * y * s + cy;
    }
}

void KBCamera::unproject(const double* uv, int n, double* rays) const {
    for (int i = 0; i < n; ++i) {
        double mx = (uv[2 * i] - cx) / fx, my = (uv[2 * i + 1] - cy) / fy;
        double d = std::hypot(mx, my);
        double th = d;
        for (int it = 0; it < 10; ++it) {  // Newton on d(th) = d
            double t2 = th * th;
            double f = distort(th) - d;
            double df = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
            th -= f / df;
        }
        double s = d > 1e-12 ? std::sin(th) / std::max(d, 1e-12) : 1.0;
        Vec3 ray{mx * s, my * s, std::cos(th)};
        double nr = norm(ray);
        for (int j = 0; j < 3; ++j) rays[3 * i + j] = ray[j] / nr;
    }
}

void triangulate(const double* o1, const double* d1, const double* o2, const double* d2, int n, double* P,
                 double* gap, double* s, double* t) {
    for (int i = 0; i < n; ++i) {
        Vec3 a1{o1[3 * i], o1[3 * i + 1], o1[3 * i + 2]}, v1{d1[3 * i], d1[3 * i + 1], d1[3 * i + 2]};
        Vec3 a2{o2[3 * i], o2[3 * i + 1], o2[3 * i + 2]}, v2{d2[3 * i], d2[3 * i + 1], d2[3 * i + 2]};
        Vec3 w = sub(a1, a2);
        double a = dot(v1, v1), b = dot(v1, v2), c = dot(v2, v2), d = dot(v1, w), e = dot(v2, w);
        double den = std::max(a * c - b * b, 1e-12);
        double si = (b * e - c * d) / den, ti = (a * e - b * d) / den;
        Vec3 p1 = add(a1, scale(v1, si)), p2 = add(a2, scale(v2, ti));
        if (P)
            for (int j = 0; j < 3; ++j) P[3 * i + j] = (p1[j] + p2[j]) / 2;
        if (gap) gap[i] = norm(sub(p1, p2));
        if (s) s[i] = si;
        if (t) t[i] = ti;
    }
}

void DetMap::raw_to_det(const double* raw, int count, double* det) const {
    double k = n / S, c = (S - 1) / 2.0;
    for (int i = 0; i < count; ++i) {
        double sx = (raw[2 * i] - x0 + 0.5) / k - 0.5, sy = (raw[2 * i + 1] - y0 + 0.5) / k - 0.5;
        double u = sx - c, v = sy - c;
        det[2 * i] = M[0] * u + M[1] * v + c;
        det[2 * i + 1] = M[2] * u + M[3] * v + c;
    }
}

void DetMap::det_to_raw(const double* det, int count, double* raw) const {
    double c = (S - 1) / 2.0, k = n / S;
    for (int i = 0; i < count; ++i) {
        double u = det[2 * i] - c, v = det[2 * i + 1] - c;
        double sx = M[0] * u + M[2] * v + c, sy = M[1] * u + M[3] * v + c;
        raw[2 * i] = (sx + 0.5) * k - 0.5 + x0;
        raw[2 * i + 1] = (sy + 0.5) * k - 0.5 + y0;
    }
}

// ---------------------------------------------------------------- mercury_kp.py
Mat3 q_matrix(const Quat& q) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    return {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
}

Quat q_mul(const Quat& a, const Quat& b) {
    double w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3], w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];
    return {w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2};
}

Vec3 q_rot(const Quat& q, const Vec3& v) {
    Vec3 u{q[1], q[2], q[3]};
    Vec3 t = scale(cross(u, v), 2);
    Vec3 ut = cross(u, t);
    Vec3 out;
    for (int i = 0; i < 3; ++i) out[i] = (v[i] + q[0] * t[i]) + ut[i];
    return out;
}

Quat q_from_two_vectors(Vec3 a, Vec3 b) {
    double na = norm(a), nb = norm(b);
    for (int i = 0; i < 3; ++i) a[i] /= na, b[i] /= nb;
    double c = dot(a, b);
    if (c < -1 + 1e-6) {  // opposite: rotate pi about any orthogonal axis
        Vec3 ax = cross(a, {1, 0, 0});
        if (norm(ax) < 1e-6) ax = cross(a, {0, 1, 0});
        double n = norm(ax);
        return {0.0, ax[0] / n, ax[1] / n, ax[2] / n};
    }
    Vec3 ax = cross(a, b);
    double s = std::sqrt((1 + c) * 2);
    return {s / 2, ax[0] / s, ax[1] / s, ax[2] / s};
}

static Quat direction_general(const Vec3& v, double twist) {  // mercury_kp.direction
    Quat one = q_from_two_vectors({0.0, 0, -1}, v);
    double h = twist / 2;
    return q_mul(one, {std::cos(h), 0, 0, -std::sin(h)});
}

Quat direction_q(const Vec3& v, double twist) {
    double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    double bx = v[0] / n, by = v[1] / n, bz = v[2] / n;
    double c = -bz;
    if (c < -1 + 1e-6) return direction_general(v, twist);
    double s = std::sqrt((1 + c) * 2);
    double w1 = s / 2, x1 = by / s, y1 = -bx / s;
    double h = twist / 2;
    double w2 = std::cos(h), z2 = -std::sin(h);
    return {w1 * w2, x1 * w2 + y1 * z2, -x1 * z2 + y1 * w2, w1 * z2};
}

namespace {
// rows R^T d of each unit joint direction, then the stereographic projection
void project_sg(const std::array<Vec3, kJoints>& dn, const Quat& q, double (*sg)[2]) {
    Mat3 Q = q_matrix(q);
    for (int i = 0; i < kJoints; ++i) {
        double d[3];
        for (int j = 0; j < 3; ++j) d[j] = dn[i][0] * Q[j] + dn[i][1] * Q[3 + j] + dn[i][2] * Q[6 + j];
        double den = 1 - d[2];
        sg[i][0] = d[0] / den;
        sg[i][1] = d[1] / den;
    }
}
}  // namespace

Instructions make_instructions(const Joints& jm, bool flip, double expand, double twist) {
    std::array<Vec3, kJoints> dn;
    std::array<double, kJoints> norms;
    for (int i = 0; i < kJoints; ++i) {
        norms[i] = norm(jm[i]);
        for (int j = 0; j < 3; ++j) dn[i][j] = jm[i][j] / norms[i];
    }
    double n9 = norm(jm[9]);
    Vec3 dir3{jm[9][0] / n9, jm[9][1] / n9, jm[9][2] / n9};
    Quat q = direction_q(dir3, twist);
    Vec3 old = dir3;
    double sg[kJoints][2];
    double r = 0;
    for (int it = 0; it < 8; ++it) {
        project_sg(dn, q, sg);
        double mn0 = sg[0][0], mn1 = sg[0][1], mx0 = sg[0][0], mx1 = sg[0][1];
        for (int i = 1; i < kJoints; ++i) {
            mn0 = std::min(mn0, sg[i][0]), mx0 = std::max(mx0, sg[i][0]);
            mn1 = std::min(mn1, sg[i][1]), mx1 = std::max(mx1, sg[i][1]);
        }
        double c0 = (mn0 + mx0) / 2, c1 = (mn1 + mx1) / 2;
        r = std::max(c0 - mn0, c1 - mn1);
        double dd = 1 + c0 * c0 + c1 * c1;
        Vec3 v{2 * c0 / dd, 2 * c1 / dd, (-1 + c0 * c0 + c1 * c1) / dd};
        Mat3 Q = q_matrix(q);
        Vec3 nw;
        for (int j = 0; j < 3; ++j) nw[j] = Q[3 * j] * v[0] + Q[3 * j + 1] * v[1] + Q[3 * j + 2] * v[2];
        q = direction_q(nw, twist);
        if (norm(sub(old, nw)) < 1e-4) break;
        old = nw;
    }
    project_sg(dn, q, sg);
    auto norm2 = [](double x, double y) { return std::sqrt(x * x + y * y); };  // np.linalg.norm
    double fwd = norm2(sg[0][0] - sg[9][0], sg[0][1] - sg[9][1]);
    double side = norm2(sg[5][0] - sg[17][0], sg[5][1] - sg[17][1]);
    double palm_l = std::max(fwd, side);
    r = std::max(r * expand, palm_l * 0.5 * (2.2 / 1.65) * expand);
    Instructions out;
    out.q = q;
    out.r = r;
    for (int i = 0; i < kJoints; ++i) {
        out.kp[2 * i] = static_cast<float>(sg[i][0] / r * (flip ? -1 : 1));
        out.kp[2 * i + 1] = static_cast<float>(-sg[i][1] / r);
    }
    double hand_size = norm(sub(jm[0], jm[9]));
    for (int i = 0; i < kJoints; ++i) out.rel[i] = (norms[i] - norms[9]) / std::max(hand_size, 1e-6);
    return out;
}

void crop_pixels(const KBCamera& cam, const Quat& q, double r, bool flip, double* out) {
    double a[kWSize], b[kWSize], a2[kWSize], b2[kWSize];
    for (int i = 0; i < kWSize; ++i) {
        double axis = 1 - static_cast<double>(2 * i) / kWSize;
        a[i] = axis * (flip ? r : -r);
        b[i] = axis * r;
        a2[i] = a[i] * a[i];
        b2[i] = b[i] * b[i];
    }
    Mat3 R = q_matrix(q);
    double xr[kWSize], yr[kWSize], zr[kWSize], xc[kWSize], yc[kWSize], zc[kWSize];
    for (int i = 0; i < kWSize; ++i) {
        xr[i] = 2 * R[1] * b[i] + R[2] * (b2[i] - 1);
        yr[i] = -(2 * R[4] * b[i] + R[5] * (b2[i] - 1));
        zr[i] = -(2 * R[7] * b[i] + R[8] * (b2[i] - 1));
        xc[i] = 2 * R[0] * a[i] + R[2] * a2[i];
        yc[i] = -(2 * R[3] * a[i] + R[5] * a2[i]);
        zc[i] = -(2 * R[6] * a[i] + R[8] * a2[i]);
    }
    for (int row = 0; row < kWSize; ++row)
        for (int col = 0; col < kWSize; ++col) {
            double x = xr[row] + xc[col], y = yr[row] + yc[col], z = zr[row] + zc[col];
            double rr = std::hypot(x, y);
            double d = cam.distort(std::atan2(rr, z));
            double s = rr > 1e-12 ? d / std::max(rr, 1e-12) : 0.0;
            double* o = out + 2 * (row * kWSize + col);
            o[0] = cam.fx * x * s + cam.cx;
            o[1] = cam.fy * y * s + cam.cy;
        }
}

void refine_all(const float* hm, double* out, int kmax) {
    for (int j = 0; j < kJoints; ++j) {
        const float* h = hm + j * kGrid * kGrid;
        int best = 0;
        for (int i = 1; i < kGrid * kGrid; ++i)
            if (h[i] > h[best]) best = i;
        int cy = best / kGrid, cx = best % kGrid;
        int kx = std::max(0, std::min(cx, std::min(kmax, std::abs(cx - kGrid) - 1)));
        int ky = std::max(0, std::min(cy, std::min(kmax, std::abs(cy - kGrid) - 1)));
        double s = 0, sx = 0, sy = 0;
        for (int y = cy - ky; y <= cy + ky; ++y) {
            double row = 0, rowx = 0;
            for (int x = cx - kx; x <= cx + kx; ++x) {
                double v = h[y * kGrid + x];
                row += v;
                rowx += v * (x + 0.5);
            }
            s += row;
            sx += rowx;
            sy += row * (y + 0.5);
        }
        if (s != 0) {
            out[2 * j] = sx / s;
            out[2 * j + 1] = sy / s;
        } else {
            out[2 * j] = cx;
            out[2 * j + 1] = cy;
        }
    }
}

Decoded decode(const float* hm, const float* hd, float extras0, const float* curls, const Quat& q, double r,
               bool flip, double cell_offset, const KBCamera& cam) {
    Decoded out;
    double locs[kJoints * 2];
    refine_all(hm, locs);
    double raw_rays[kJoints * 3];
    for (int j = 0; j < kJoints; ++j) {
        const float* h = hm + j * kGrid * kGrid;
        out.conf[j] = *std::max_element(h, h + kGrid * kGrid);
        double lx = locs[2 * j] - (0.5 - cell_offset), ly = locs[2 * j + 1] - (0.5 - cell_offset);
        double px = flip ? (1 - lx / 11.0) : (lx / 11.0 - 1);
        double py = 1 - ly / 11.0;
        double x = px * r, y = py * r;
        double d = 1 + x * x + y * y;
        Vec3 dl{2 * x / d, 2 * y / d, (-1 + x * x + y * y) / d};
        Vec3 dm = q_rot(q, dl);
        Vec3 ray{dm[0], -dm[1], -dm[2]};
        double n = norm(ray);
        for (int i = 0; i < 3; ++i) out.rays[j][i] = ray[i] / n, raw_rays[3 * j + i] = out.rays[j][i];
        // depth: centre of mass of the clipped depth heatmap row
        float dep[kGrid];
        double wdep[kGrid];
        for (int i = 0; i < kGrid; ++i) {
            dep[i] = std::min(std::max(hd[j * kGrid + i], 0.0f), 1.0f);
            wdep[i] = static_cast<double>(dep[i]) * i;
        }
        float s = std::max(pairwise_sum(dep, kGrid), 1e-5f);
        double com = pairwise_sum(wdep, kGrid) / static_cast<double>(s);
        out.rel_depth[j] = ((com + 0.5) / kGrid - 0.5) * 3.0;
    }
    cam.project(raw_rays, kJoints, out.raw.data());
    out.is_hand = 1 / (1 + std::exp(-static_cast<double>(extras0)));
    for (int i = 0; i < 5; ++i) {
        out.curls[i] = curls[i];
        out.curl_var[i] = std::fabs(curls[5 + i]) + 0.01f;
    }
    return out;
}

// ---------------------------------------------------------------- palm.py
void OneEuro::set(std::optional<std::pair<double, double>> params) {
    if (!params) {
        bypass = true;
    } else {
        bypass = false;
        min_cutoff = params->first;
        beta = params->second;
    }
}

static double alpha(double cutoff, double dt) {
    double r = 2 * kPi * cutoff * dt;
    return r / (r + 1);
}

std::vector<double> OneEuro::operator()(const std::vector<double>& xin, double tin) {
    if (has && tin <= t) return x;
    if (bypass || !has || tin - t > 0.3) {
        x = xin;
        dx.assign(xin.size(), 0.0);
        t = tin;
        has = true;
        return x;
    }
    double dt = tin - t;
    double ad = alpha(d_cutoff, dt);
    double ss = 0;
    for (size_t i = 0; i < xin.size(); ++i) {
        double d = (xin[i] - x[i]) / dt;
        dx[i] = ad * d + (1 - ad) * dx[i];
    }
    for (size_t i = 0; i < dx.size(); ++i) ss += dx[i] * dx[i];
    double cutoff = min_cutoff + beta * std::sqrt(ss);
    double a = alpha(cutoff, dt);
    for (size_t i = 0; i < xin.size(); ++i) x[i] = a * xin[i] + (1 - a) * x[i];
    t = tin;
    return x;
}

Quat quat_from_R(const Mat3& R) {
    auto at = [&](int i, int j) { return R[3 * i + j]; };
    double tr = at(0, 0) + at(1, 1) + at(2, 2);
    if (tr > 0) {
        double s = std::sqrt(tr + 1) * 2;
        return {0.25 * s, (at(2, 1) - at(1, 2)) / s, (at(0, 2) - at(2, 0)) / s, (at(1, 0) - at(0, 1)) / s};
    }
    int i = 0;
    if (at(1, 1) > at(i, i)) i = 1;
    if (at(2, 2) > at(i, i)) i = 2;
    int j = (i + 1) % 3, k = (i + 2) % 3;
    double s = std::sqrt(1 + at(i, i) - at(j, j) - at(k, k)) * 2;
    Quat q{};
    q[0] = (at(k, j) - at(j, k)) / s;
    q[1 + i] = 0.25 * s;
    q[1 + j] = (at(j, i) + at(i, j)) / s;
    q[1 + k] = (at(k, i) + at(i, k)) / s;
    return q;
}

static double ang(const Vec3& a, const Vec3& b) {
    double c = dot(a, b) / std::max(norm(a) * norm(b), 1e-9);
    return std::acos(std::max(-1.0, std::min(1.0, c)));
}

std::array<double, 5> finger_curls(const Joints& W) {
    static const int fingers[4][5] = {{0, 5, 6, 7, 8}, {0, 9, 10, 11, 12}, {0, 13, 14, 15, 16}, {0, 17, 18, 19, 20}};
    std::array<double, 5> out;
    Vec3 t0 = sub(W[2], W[1]), t1 = sub(W[3], W[2]), t2 = sub(W[4], W[3]);
    out[0] = std::min(1.0, std::max(0.0, (ang(t0, t1) + ang(t1, t2)) / (140 * kDegToRad)));
    for (int f = 0; f < 4; ++f) {
        Vec3 b[4];
        for (int i = 0; i < 4; ++i) b[i] = sub(W[fingers[f][i + 1]], W[fingers[f][i]]);
        double tot = ang(b[0], b[1]) + ang(b[1], b[2]) + ang(b[2], b[3]);
        out[1 + f] = std::min(1.0, std::max(0.0, tot / (250 * kDegToRad)));
    }
    return out;
}

std::array<double, 5> finger_splays(const Joints& P) {
    static const int bones[4][2] = {{5, 6}, {9, 10}, {13, 14}, {17, 18}};
    static const double rest[4] = {0.0 * kDegToRad, -6.0 * kDegToRad, -13.0 * kDegToRad, -15.0 * kDegToRad};
    const double unit = 15.0 * kDegToRad;
    std::array<double, 5> out{};
    Vec3 fwd = sub(P[9], P[0]), lat = sub(P[5], P[17]);
    if (norm(fwd) < 1e-6 || norm(lat) < 1e-6) return out;
    double nf = norm(fwd);
    for (auto& v : fwd) v /= nf;
    double lf = dot(lat, fwd);
    for (int i = 0; i < 3; ++i) lat[i] = lat[i] - lf * fwd[i];
    double nl = std::max(norm(lat), 1e-9);
    for (auto& v : lat) v /= nl;
    Vec3 n = cross(fwd, lat);
    for (int f = 0; f < 4; ++f) {
        Vec3 b = sub(P[bones[f][1]], P[bones[f][0]]);
        double bn = dot(b, n);
        Vec3 bp{b[0] - bn * n[0], b[1] - bn * n[1], b[2] - bn * n[2]};
        double inplane = norm(bp) / std::max(norm(b), 1e-9);
        double a = std::atan2(dot(bp, lat), dot(bp, fwd));
        double fade = std::min(1.0, std::max(0.0, (inplane - 0.5) / 0.3));
        out[1 + f] = std::max(-1.0, std::min(1.0, fade * (a - rest[f]) / unit));
    }
    return out;
}

PalmAxes palm_axes(bool right, const Joints& P, bool roll_flip) {
    const Vec3 &wrist = P[0], &idx = P[5], &mid = P[9], &ring = P[13], &pinky = P[17];
    PalmAxes a;
    for (int i = 0; i < 3; ++i) a.centre[i] = (((wrist[i] + idx[i]) + mid[i]) + pinky[i]) / 4;
    Vec3 fwd;
    for (int i = 0; i < 3; ++i) fwd[i] = (((idx[i] + mid[i]) + ring[i]) + pinky[i]) / 4 - wrist[i];
    Vec3 across = sub(idx, pinky);
    double nf = norm(fwd);
    for (int i = 0; i < 3; ++i) a.z[i] = -fwd[i] / nf;
    Vec3 x = right ? scale(across, -1) : across;
    if (roll_flip) x = scale(x, -1);
    double xz = dot(x, a.z);
    for (int i = 0; i < 3; ++i) x[i] = x[i] - a.z[i] * xz;
    double nx = norm(x);
    for (int i = 0; i < 3; ++i) a.x[i] = x[i] / nx;
    return a;
}

PalmPose PalmFilter::operator()(bool right, const Joints& P, double t, bool roll_flip) {
    PalmAxes a = palm_axes(right, P, roll_flip);
    std::vector<double> c = pos({a.centre[0], a.centre[1], a.centre[2]}, t);
    std::vector<double> zv = z({a.z[0], a.z[1], a.z[2]}, t);
    // Python divides the filter's own state in place (z /= norm(z)): keep that
    double nz = std::sqrt(zv[0] * zv[0] + zv[1] * zv[1] + zv[2] * zv[2]);
    for (int i = 0; i < 3; ++i) z.x[i] = zv[i] / nz;
    Vec3 zz{z.x[0], z.x[1], z.x[2]};
    std::vector<double> xv = x({a.x[0], a.x[1], a.x[2]}, t);
    Vec3 xx{xv[0], xv[1], xv[2]};
    double xz = dot(xx, zz);
    for (int i = 0; i < 3; ++i) xx[i] = xx[i] - zz[i] * xz;
    double nx = norm(xx);
    for (auto& v : xx) v /= nx;
    Vec3 y = cross(zz, xx);
    Mat3 R{xx[0], y[0], zz[0], xx[1], y[1], zz[1], xx[2], y[2], zz[2]};
    return {{c[0], c[1], c[2]}, quat_from_R(R)};
}

// ---------------------------------------------------------------- merctrack.py
HandState::HandState(const Joints& P0, double t0) : P(P0), t(t0), hand_size(norm(sub(P0[0], P0[9]))) {}

Vec3 palm_centre(const Joints& P) { return mean4(P[0], P[5], P[9], P[17]); }

Placement place(const HandState& hs, const std::optional<Vec3>& fp, bool predict_roi, double t_cap) {
    Placement out;
    out.P = hs.P;
    if (fp) {  // move the ROI with the 60 Hz palm position
        Vec3 d = sub(*fp, palm_centre(hs.P));
        for (auto& p : out.P) p = add(p, d);
    } else if (predict_roi) {
        double dt = std::min(std::max(t_cap - hs.t, 0.0), 0.15);
        Vec3 d = scale(hs.vel, dt);
        for (auto& p : out.P) p = add(p, d);
    }
    if (predict_roi) {
        double spd = norm(hs.vel);
        if (!fp || spd > 1.0) out.expand = 1.7 * ((!fp || spd > 2.5) ? 1.35 : 1.15);
    }
    return out;
}

bool camera_joints(const Joints& P, const Pose& pose, Joints& out) {
    bool ok = true;
    const Mat3& R = pose.R;
    for (int i = 0; i < kJoints; ++i) {
        Vec3 d = sub(P[i], pose.t);
        for (int j = 0; j < 3; ++j) out[i][j] = d[0] * R[j] + d[1] * R[3 + j] + d[2] * R[6 + j];
        if (out[i][2] < 0.08) ok = false;
    }
    return ok;
}

Observed observe(HandState& hs, int slot, double t, const Vec3& o, const std::array<Vec3, kJoints>& rays,
                 const std::array<double, kJoints>& rel_depth, const std::optional<Vec3>& fp, double stereo_ms,
                 bool stereo_comp) {
    Observed res;
    bool have = false;
    auto other = hs.last_rays.find(1 - slot);
    if (other != hs.last_rays.end() && std::fabs(t - other->second.t) * 1e3 <= stereo_ms &&
        (stereo_ms > 0 || t == other->second.t)) {
        const ViewRays& ov = other->second;
        Vec3 oo = ov.origin;
        if (stereo_comp && t != ov.t) {  // the hand moved since the other view: move that view with it
            Vec3 move = (fp && ov.palm) ? sub(*fp, *ov.palm) : scale(hs.vel, t - ov.t);
            if (norm(move) < 0.1) oo = add(oo, move);
            res.comp_mm = norm(move) * 1e3;
        }
        double O1[kJoints * 3], D1[kJoints * 3], O2[kJoints * 3], D2[kJoints * 3];
        for (int i = 0; i < kJoints; ++i)
            for (int j = 0; j < 3; ++j) {
                O1[3 * i + j] = o[j], D1[3 * i + j] = rays[i][j];
                O2[3 * i + j] = oo[j], D2[3 * i + j] = ov.rays[i][j];
            }
        double Pm[kJoints * 3], gap[kJoints], s1[kJoints], s2[kJoints];
        triangulate(O1, D1, O2, D2, kJoints, Pm, gap, s1, s2);
        std::array<double, kJoints> g;
        std::copy(gap, gap + kJoints, g.begin());
        std::nth_element(g.begin(), g.begin() + kJoints / 2, g.end());
        bool ahead = true;
        for (int i = 0; i < kJoints; ++i) ahead = ahead && s1[i] > 0.05 && s2[i] > 0.05;
        if (g[kJoints / 2] < 0.02 && ahead) {
            for (int i = 0; i < kJoints; ++i) res.P[i] = {Pm[3 * i], Pm[3 * i + 1], Pm[3 * i + 2]};
            hs.hand_size = 0.9 * hs.hand_size + 0.1 * norm(sub(res.P[0], res.P[9]));
            res.outcome = Outcome::Stereo;
            have = true;
        }
    }
    if (!have) {  // single view: rays at Mercury's relative depths, anchored on the palm position
        Vec3 prevc = palm_centre(hs.P);
        Vec3 anchor = fp ? *fp : prevc;
        double d9 = norm(sub(add(hs.P[9], sub(anchor, prevc)), o));
        for (int i = 0; i < kJoints; ++i) res.P[i] = add(o, scale(rays[i], d9 + rel_depth[i] * hs.hand_size));
        if (fp) {
            Vec3 d = sub(anchor, palm_centre(res.P));
            for (auto& p : res.P) p = add(p, d);
        }
        res.outcome = Outcome::Mono;
    }
    double jump = norm(sub(palm_centre(res.P), palm_centre(hs.P)));
    if (jump > 0.25 && t - hs.t < 0.2) res.outcome = Outcome::Jump;
    return res;
}

void accept(HandState& hs, const Joints& P, double t) {
    double dtv = t - hs.t;
    if (0.005 < dtv && dtv < 0.2) {
        Vec3 v = sub(palm_centre(P), palm_centre(hs.P));
        for (auto& c : v) c /= dtv;
        double sp = norm(v);
        hs.vel = sp > 0 ? scale(v, std::min(1.0, 6.0 / sp)) : v;
    } else {
        hs.vel = {0, 0, 0};
    }
    hs.P = P;
    hs.t = t;
    hs.misses = 0;
}

}  // namespace framehands
