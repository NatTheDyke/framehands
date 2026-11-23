// framehands._core: pybind11 bindings of the geometry core (include/framehands/core.hpp).
// Arrays cross as float64 (float32 for model outputs), C-contiguous; the wrappers in
// framehands/_native.py do the broadcasting and shape checks the Python callers rely on.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "framehands/core.hpp"

namespace py = pybind11;
namespace fh = framehands;

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

void need(const py::array& a, py::ssize_t n, const char* what) {
    if (a.size() != n) throw std::invalid_argument(std::string(what) + ": wrong size");
}
template <typename T>
fh::Joints joints(const T& a) {
    need(a, fh::kJoints * 3, "joints (21, 3)");
    fh::Joints J;
    auto p = a.data();
    for (int i = 0; i < fh::kJoints; ++i) J[i] = {p[3 * i], p[3 * i + 1], p[3 * i + 2]};
    return J;
}
F64 joints_array(const fh::Joints& J) {
    F64 out({fh::kJoints, 3});
    auto p = out.mutable_data();
    for (int i = 0; i < fh::kJoints; ++i)
        for (int j = 0; j < 3; ++j) p[3 * i + j] = J[i][j];
    return out;
}
fh::Vec3 vec3(const F64& a) {
    need(a, 3, "vector (3,)");
    return {a.data()[0], a.data()[1], a.data()[2]};
}
fh::Quat quat(const F64& a) {
    need(a, 4, "quaternion (4,)");
    auto p = a.data();
    return {p[0], p[1], p[2], p[3]};
}
F64 arr(const double* p, std::vector<py::ssize_t> shape) {
    F64 out(shape);
    std::copy(p, p + out.size(), out.mutable_data());
    return out;
}
std::optional<fh::Vec3> opt_vec3(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return vec3(o.cast<F64>());
}
py::object opt_array(const std::optional<fh::Vec3>& v) {
    if (!v) return py::none();
    return arr(v->data(), {3});
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "framehands geometry core (C++)";
    m.attr("ABI") = 1;

    py::class_<fh::KBCamera>(m, "KBCamera")
        .def(py::init([](double fx, double fy, double cx, double cy, F64 k) {
            need(k, 4, "k");
            fh::KBCamera c;
            c.fx = fx, c.fy = fy, c.cx = cx, c.cy = cy;
            for (int i = 0; i < 4; ++i) c.k[i] = k.data()[i];
            return c;
        }))
        .def("distort", [](const fh::KBCamera& c, F64 th) {
            F64 out(std::vector<py::ssize_t>(th.shape(), th.shape() + th.ndim()));
            for (py::ssize_t i = 0; i < th.size(); ++i) out.mutable_data()[i] = c.distort(th.data()[i]);
            return out;
        })
        .def("project", [](const fh::KBCamera& c, F64 P) {
            if (P.size() % 3) throw std::invalid_argument("points (n, 3)");
            int n = static_cast<int>(P.size() / 3);
            F64 out({n, 2});
            c.project(P.data(), n, out.mutable_data());
            return out;
        })
        .def("unproject", [](const fh::KBCamera& c, F64 uv) {
            if (uv.size() % 2) throw std::invalid_argument("pixels (n, 2)");
            int n = static_cast<int>(uv.size() / 2);
            F64 out({n, 3});
            c.unproject(uv.data(), n, out.mutable_data());
            return out;
        });

    m.def("triangulate", [](F64 o1, F64 d1, F64 o2, F64 d2) {
        py::ssize_t n3 = o1.size();
        if (n3 % 3 || d1.size() != n3 || o2.size() != n3 || d2.size() != n3)
            throw std::invalid_argument("triangulate: four (n, 3) arrays");
        int n = static_cast<int>(n3 / 3);
        F64 P({n, 3}), gap({n}), s({n}), t({n});
        fh::triangulate(o1.data(), d1.data(), o2.data(), d2.data(), n, P.mutable_data(), gap.mutable_data(),
                        s.mutable_data(), t.mutable_data());
        return py::make_tuple(P, gap, s, t);
    });

    py::class_<fh::DetMap>(m, "DetMap")
        .def(py::init([](double n, double S, double x0, double y0, F64 M) {
            need(M, 4, "M (2, 2)");
            fh::DetMap d;
            d.n = n, d.S = S, d.x0 = x0, d.y0 = y0;
            for (int i = 0; i < 4; ++i) d.M[i] = M.data()[i];
            return d;
        }))
        .def("raw_to_det", [](const fh::DetMap& d, F64 raw) {
            F64 out(std::vector<py::ssize_t>(raw.shape(), raw.shape() + raw.ndim()));
            d.raw_to_det(raw.data(), static_cast<int>(raw.size() / 2), out.mutable_data());
            return out;
        })
        .def("det_to_raw", [](const fh::DetMap& d, F64 det) {
            F64 out(std::vector<py::ssize_t>(det.shape(), det.shape() + det.ndim()));
            d.det_to_raw(det.data(), static_cast<int>(det.size() / 2), out.mutable_data());
            return out;
        });

    m.def("make_instructions", [](F64 jm, bool flip, double expand, double twist) {
        fh::Instructions in = fh::make_instructions(joints(jm), flip, expand, twist);
        F32 kp({fh::kJoints, 2});
        std::copy(in.kp.begin(), in.kp.end(), kp.mutable_data());
        return py::make_tuple(arr(in.q.data(), {4}), in.r, kp, arr(in.rel.data(), {fh::kJoints}));
    }, py::arg("joints_m"), py::arg("flip"), py::arg("expand"), py::arg("twist") = 0.0);

    m.def("crop_pixels", [](const fh::KBCamera& cam, F64 q, double r, bool flip) {
        F64 out({fh::kWSize, fh::kWSize, 2});
        fh::crop_pixels(cam, quat(q), r, flip, out.mutable_data());
        return out;
    });

    m.def("refine_all", [](F32 hm) {
        need(hm, fh::kJoints * fh::kGrid * fh::kGrid, "heatmap (21, 484)");
        F64 out({fh::kJoints, 2});
        fh::refine_all(hm.data(), out.mutable_data());
        return out;
    });

    m.def("decode", [](F32 hm, F32 hd, float extras0, F32 curls, F64 q, double r, bool flip, double cell_offset,
                       const fh::KBCamera& cam) {
        need(hm, fh::kJoints * fh::kGrid * fh::kGrid, "heatmap_xy");
        need(hd, fh::kJoints * fh::kGrid, "heatmap_depth");
        need(curls, 10, "curls");
        fh::Decoded d = fh::decode(hm.data(), hd.data(), extras0, curls.data(), quat(q), r, flip, cell_offset, cam);
        F32 conf({fh::kJoints}), c({5}), v({5});
        std::copy(d.conf.begin(), d.conf.end(), conf.mutable_data());
        std::copy(d.curls.begin(), d.curls.end(), c.mutable_data());
        std::copy(d.curl_var.begin(), d.curl_var.end(), v.mutable_data());
        py::dict out;
        out["rays"] = arr(d.rays[0].data(), {fh::kJoints, 3});
        out["raw"] = arr(d.raw.data(), {fh::kJoints, 2});
        out["conf"] = conf;
        out["rel_depth"] = arr(d.rel_depth.data(), {fh::kJoints});
        out["is_hand"] = d.is_hand;
        out["curls"] = c;
        out["curl_var"] = v;
        return out;
    });

    py::class_<fh::OneEuro>(m, "OneEuro")
        .def(py::init<double, double, double>(), py::arg("min_cutoff") = 1.5, py::arg("beta") = 0.6,
             py::arg("d_cutoff") = 1.0)
        .def("set", [](fh::OneEuro& f, py::object params) {
            if (params.is_none()) f.set(std::nullopt);
            else f.set(params.cast<std::pair<double, double>>());
        })
        .def("__call__", [](fh::OneEuro& f, F64 x, double t) {
            std::vector<double> v(x.data(), x.data() + x.size());
            std::vector<double> r = f(v, t);
            return arr(r.data(), {static_cast<py::ssize_t>(r.size())});
        })
        .def_readwrite("min_cutoff", &fh::OneEuro::min_cutoff)
        .def_readwrite("beta", &fh::OneEuro::beta)
        .def_readwrite("d_cutoff", &fh::OneEuro::d_cutoff)
        .def_readwrite("bypass", &fh::OneEuro::bypass)
        .def_property_readonly("t", [](const fh::OneEuro& f) -> py::object {
            return f.has ? py::object(py::float_(f.t)) : py::none();
        })
        .def_property_readonly("x", [](const fh::OneEuro& f) -> py::object {
            return f.has ? py::object(arr(f.x.data(), {static_cast<py::ssize_t>(f.x.size())})) : py::none();
        });

    m.def("quat_from_R", [](F64 R) {
        need(R, 9, "R (3, 3)");
        fh::Mat3 M;
        std::copy(R.data(), R.data() + 9, M.begin());
        fh::Quat q = fh::quat_from_R(M);
        return arr(q.data(), {4});
    });
    m.def("finger_curls", [](F64 W) {
        auto c = fh::finger_curls(joints(W));
        return arr(c.data(), {5});
    });
    m.def("finger_splays", [](F64 P) {
        auto c = fh::finger_splays(joints(P));
        return arr(c.data(), {5});
    });
    m.def("palm_axes", [](bool right, F64 P, bool roll_flip) {
        fh::PalmAxes a = fh::palm_axes(right, joints(P), roll_flip);
        return py::make_tuple(arr(a.centre.data(), {3}), arr(a.x.data(), {3}), arr(a.z.data(), {3}));
    }, py::arg("right"), py::arg("P"), py::arg("roll_flip") = false);

    py::class_<fh::PalmFilter>(m, "PalmFilter")
        .def(py::init<>())
        .def("__call__", [](fh::PalmFilter& f, bool right, F64 P, double t, bool roll_flip) {
            fh::PalmPose p = f(right, joints(P), t, roll_flip);
            return py::make_tuple(arr(p.centre.data(), {3}), arr(p.q.data(), {4}));
        }, py::arg("right"), py::arg("P"), py::arg("t"), py::arg("roll_flip") = false)
        .def_readonly("pos", &fh::PalmFilter::pos, py::return_value_policy::reference_internal)
        .def_readonly("z", &fh::PalmFilter::z, py::return_value_policy::reference_internal)
        .def_readonly("x", &fh::PalmFilter::x, py::return_value_policy::reference_internal);

    py::class_<fh::HandState>(m, "HandState")
        .def(py::init([](F64 P, double t) { return fh::HandState(joints(P), t); }))
        .def_property("P", [](const fh::HandState& h) { return joints_array(h.P); },
                      [](fh::HandState& h, F64 P) { h.P = joints(P); })
        .def_readwrite("t", &fh::HandState::t)
        .def_readwrite("hand_size", &fh::HandState::hand_size)
        .def_readwrite("misses", &fh::HandState::misses)
        .def_property("vel", [](const fh::HandState& h) { return arr(h.vel.data(), {3}); },
                      [](fh::HandState& h, F64 v) { h.vel = vec3(v); })
        .def("set_view", [](fh::HandState& h, int slot, double t, F64 origin, F64 rays, py::object palm) {
            fh::ViewRays v;
            v.t = t;
            v.origin = vec3(origin);
            fh::Joints r = joints(rays);
            std::copy(r.begin(), r.end(), v.rays.begin());
            v.palm = opt_vec3(palm);
            h.last_rays[slot] = v;
        })
        .def("view", [](const fh::HandState& h, int slot) -> py::object {
            auto it = h.last_rays.find(slot);
            if (it == h.last_rays.end()) return py::none();
            const fh::ViewRays& v = it->second;
            return py::make_tuple(v.t, arr(v.origin.data(), {3}), arr(v.rays[0].data(), {fh::kJoints, 3}),
                                  opt_array(v.palm));
        });

    m.def("palm_centre", [](F64 P) {
        fh::Vec3 c = fh::palm_centre(joints(P));
        return arr(c.data(), {3});
    });
    m.def("place", [](const fh::HandState& h, py::object fp, bool predict_roi, double t_cap) {
        fh::Placement p = fh::place(h, opt_vec3(fp), predict_roi, t_cap);
        return py::make_tuple(joints_array(p.P), p.expand ? py::object(py::float_(*p.expand)) : py::none());
    });
    m.def("camera_joints", [](F64 P, F64 R, F64 t) {
        need(R, 9, "R (3, 3)");
        fh::Pose pose;
        std::copy(R.data(), R.data() + 9, pose.R.begin());
        pose.t = vec3(t);
        fh::Joints out;
        bool ok = fh::camera_joints(joints(P), pose, out);
        return py::make_tuple(joints_array(out), ok);
    });
    m.def("observe", [](fh::HandState& h, int slot, double t, F64 origin, F64 rays, F64 rel, py::object fp,
                        double stereo_ms, bool stereo_comp) {
        need(rel, fh::kJoints, "rel_depth (21,)");
        fh::Joints r = joints(rays);
        std::array<fh::Vec3, fh::kJoints> rr;
        std::copy(r.begin(), r.end(), rr.begin());
        std::array<double, fh::kJoints> rd;
        std::copy(rel.data(), rel.data() + fh::kJoints, rd.begin());
        fh::Observed o = fh::observe(h, slot, t, vec3(origin), rr, rd, opt_vec3(fp), stereo_ms, stereo_comp);
        static const char* names[] = {"stereo", "mono", "jump"};
        return py::make_tuple(names[static_cast<int>(o.outcome)], joints_array(o.P),
                              o.comp_mm ? py::object(py::float_(*o.comp_mm)) : py::none());
    });
    m.def("accept", [](fh::HandState& h, F64 P, double t) { fh::accept(h, joints(P), t); });
}
