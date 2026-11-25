// Pyramidal Lucas-Kanade on 8-bit images: OpenCV's calcOpticalFlowPyrLK (fixed-point bilinear windows,
// Scharr gradients, the same start, stop and rejection rules), for a few points on images whose pyramids
// the caller keeps. Gradients are computed in the windows that are read, not for whole images.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace framehands {

struct FlowLevel {
    const std::uint8_t* data;
    int w, h;
    std::ptrdiff_t stride;
};
using FlowPyramid = std::vector<FlowLevel>;  // [0] = the image, each next level half the size (pyrDown)

struct FlowParams {
    int win = 15;          // odd, <= kFlowMaxWin
    int iterations = 15;
    float epsilon = 0.05f;
    float min_eig = 1e-4f;
};
constexpr int kFlowMaxWin = 31;

// pts, next: n x (x, y). status: 1 = found. Both pyramids have the same levels.
void flow_lk(const FlowPyramid& prev, const FlowPyramid& cur, const float* pts, int n, float* next,
             std::uint8_t* status, const FlowParams& p);

}  // namespace framehands
