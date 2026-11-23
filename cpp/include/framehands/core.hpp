// framehands geometry core: camera model, Mercury crop placement / decode, palm pose and the
// per-hand Mercury tracking state, in plain C++17 (no Python). The Python modules
// (framehands/calib.py, mercury_kp.py, palm.py, merctrack.py) are the reference; every function
// here names the Python it follows and keeps its formulas and evaluation order.
//
// Arrays are row-major double unless noted: points (n, 3), pixels (n, 2), rotations 3x3.
// Quaternions are (w, x, y, z). Build with -ffp-contract=off: fused multiply-adds change results.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace framehands {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;  // row-major
using Quat = std::array<double, 4>;  // w, x, y, z
constexpr int kJoints = 21;
using Joints = std::array<Vec3, kJoints>;

// ---------------------------------------------------------------- calib.py
struct KBCamera {  // Kannala-Brandt, 4 coefficients, on raw buffer pixels
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::array<double, 4> k{};
    double distort(double th) const;                                // KBCamera._d
    void project(const double* P, int n, double* uv) const;         // camera points -> pixels
    void unproject(const double* uv, int n, double* rays) const;    // pixels -> unit rays
};

struct Pose {  // head_from_cam: p_head = R p_cam + t
    Mat3 R{};
    Vec3 t{};
};

// calib.triangulate: midpoint of closest approach of rays o + s d. Each output may be null.
void triangulate(const double* o1, const double* d1, const double* o2, const double* d2, int n,
                 double* P, double* gap, double* s, double* t);

// source.Source.raw_to_det / det_to_raw: raw camera pixels <-> the rotated detector image
struct DetMap {
    double n = 0, S = 0, x0 = 0, y0 = 0;
    std::array<double, 4> M{};  // 2x2 rotation, row-major
    void raw_to_det(const double* raw, int count, double* det) const;
    void det_to_raw(const double* det, int count, double* raw) const;
};

// ---------------------------------------------------------------- mercury_kp.py
constexpr int kWSize = 128;
constexpr int kGrid = 22;

Mat3 q_matrix(const Quat& q);
Quat q_mul(const Quat& a, const Quat& b);
Vec3 q_rot(const Quat& q, const Vec3& v);
Quat q_from_two_vectors(Vec3 a, Vec3 b);
Quat direction_q(const Vec3& v, double twist);  // mercury_kp._direction

struct Instructions {
    Quat q{};
    double r = 0;
    std::array<float, kJoints * 2> kp{};  // the model's hint input
    std::array<double, kJoints> rel{};
};
// make_instructions: joints in Mercury's camera frame (x right, y up, -z forward)
Instructions make_instructions(const Joints& joints_m, bool flip, double expand, double twist = 0.0);

// MercuryKeypoint.crop_pixels: raw pixel (kWSize * kWSize * 2) sampled by each crop pixel
void crop_pixels(const KBCamera& cam, const Quat& q, double r, bool flip, double* out);

struct Decoded {
    std::array<Vec3, kJoints> rays{};  // camera frame (OpenCV), unit
    std::array<double, kJoints * 2> raw{};
    std::array<float, kJoints> conf{};
    std::array<double, kJoints> rel_depth{};
    double is_hand = 0;
    std::array<float, 5> curls{}, curl_var{};
};
// MercuryKeypoint._decode without the image fields: heatmap_xy (21 * 22 * 22 float32),
// heatmap_depth (21 * 22), scalar_extras[0], curls (10)
Decoded decode(const float* heatmap_xy, const float* heatmap_depth, float extras0, const float* curls,
               const Quat& q, double r, bool flip, double cell_offset, const KBCamera& cam);
// MercuryKeypoint._refine_all: (21, 2) refined peak (x, y) of each joint's heatmap
void refine_all(const float* hm, double* out, int kmax = 10);

// ---------------------------------------------------------------- palm.py
class OneEuro {  // palm.OneEuro for vectors of any length
  public:
    explicit OneEuro(double min_cutoff = 1.5, double beta = 0.6, double d_cutoff = 1.0)
        : min_cutoff(min_cutoff), beta(beta), d_cutoff(d_cutoff) {}
    void set(std::optional<std::pair<double, double>> params);
    std::vector<double> operator()(const std::vector<double>& x, double t);
    double min_cutoff, beta, d_cutoff;
    bool bypass = false;
    bool has = false;
    std::vector<double> x, dx;
    double t = 0;
};

Quat quat_from_R(const Mat3& R);
std::array<double, 5> finger_curls(const Joints& W);
std::array<double, 5> finger_splays(const Joints& P);

struct PalmAxes {
    Vec3 centre{}, x{}, z{};
};
PalmAxes palm_axes(bool right, const Joints& P, bool roll_flip = false);  // PalmTracker._axes

struct PalmPose {
    Vec3 centre{};
    Quat q{};
};
class PalmFilter {  // the three filters of PalmTracker._pose for one hand
  public:
    PalmFilter() : pos(), z(1.0, 0.3), x(1.0, 0.3) {}
    PalmPose operator()(bool right, const Joints& P, double t, bool roll_flip = false);
    OneEuro pos, z, x;
};

// ---------------------------------------------------------------- merctrack.py
constexpr double kStereoMs = 40.0;
constexpr double kLostAfterS = 0.35;

struct ViewRays {  // HandState.last_rays[slot]: t, origin, head-frame rays, palm position at t
    double t = 0;
    Vec3 origin{};
    std::array<Vec3, kJoints> rays{};
    std::optional<Vec3> palm;
};

struct HandState {
    Joints P{};
    double t = 0, hand_size = 0;
    Vec3 vel{};
    int misses = 0;
    std::map<int, ViewRays> last_rays;
    explicit HandState(const Joints& P0 = {}, double t0 = 0);
};

Vec3 palm_centre(const Joints& P);  // MercuryTracker.palm_c

struct Placement {
    Joints P{};             // head frame
    std::optional<double> expand;
};
// MercuryTracker.submit's crop placement for one hand
Placement place(const HandState& hs, const std::optional<Vec3>& fast_pos, bool predict_roi, double t_cap);
// the hand's joints in one camera (pose) frame, and whether every joint is >= 8 cm in front
bool camera_joints(const Joints& P, const Pose& pose, Joints& out);

enum class Outcome : int { Stereo = 0, Mono = 1, Jump = 2 };
struct Observed {
    Outcome outcome = Outcome::Mono;
    Joints P{};
    std::optional<double> comp_mm;
};
// MercuryTracker.collect for one valid, unpaired result of view `slot` (rays in the head frame):
// triangulate against the other view when fresh, else single-view joints; then the jump check.
// On Stereo / Mono the caller accepts (accept); on Jump it counts a miss.
Observed observe(HandState& hs, int slot, double t, const Vec3& origin, const std::array<Vec3, kJoints>& rays,
                 const std::array<double, kJoints>& rel_depth, const std::optional<Vec3>& fast_pos,
                 double stereo_ms, bool stereo_comp);
void accept(HandState& hs, const Joints& P, double t);  // velocity, state, misses = 0

}  // namespace framehands
