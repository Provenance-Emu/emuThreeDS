// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/vector_math_neon.h"

namespace Common {

template <typename T>
class Quaternion {
public:
    Vec3<T> xyz;
    T w{};

    [[nodiscard]] Quaternion<decltype(-T{})> Inverse() const {
        Vec3<decltype(-T{})> neg_xyz = {
            -xyz.x,
            -xyz.y,
            -xyz.z
        };
        return {neg_xyz, w};
    }

    [[nodiscard]] Quaternion<decltype(T{} + T{})> operator+(const Quaternion& other) const {
        Vec3<decltype(T{} + T{})> result_xyz = {
            xyz.x + other.xyz.x,
            xyz.y + other.xyz.y,
            xyz.z + other.xyz.z
        };
        return {result_xyz, w + other.w};
    }

    [[nodiscard]] Quaternion<decltype(T{} - T{})> operator-(const Quaternion& other) const {
        Vec3<decltype(T{} - T{})> result_xyz = {
            xyz.x - other.xyz.x,
            xyz.y - other.xyz.y,
            xyz.z - other.xyz.z
        };
        return {result_xyz, w - other.w};
    }

    [[nodiscard]] Quaternion<decltype(T{} * T{} - T{} * T{})> operator*(
        const Quaternion& other) const {
        // Component-wise multiplication and addition to avoid ambiguity
        Vec3<decltype(T{} * T{})> term1 = {
            xyz.x * other.w,
            xyz.y * other.w,
            xyz.z * other.w
        };
        
        Vec3<decltype(T{} * T{})> term2 = {
            other.xyz.x * w,
            other.xyz.y * w,
            other.xyz.z * w
        };
        
        Vec3<decltype(T{} * T{})> cross_result = Cross(xyz, other.xyz);
        
        Vec3<decltype(T{} * T{} + T{} * T{} + T{} * T{})> result_xyz = {
            term1.x + term2.x + cross_result.x,
            term1.y + term2.y + cross_result.y,
            term1.z + term2.z + cross_result.z
        };
        
        // Calculate dot product manually to avoid ambiguity
        T dot_result = xyz.x * other.xyz.x + xyz.y * other.xyz.y + xyz.z * other.xyz.z;
        
        return {result_xyz, w * other.w - dot_result};
    }

    [[nodiscard]] Quaternion<T> Normalized() const {
        T length = std::sqrt(xyz.Length2() + w * w);
        Vec3<T> normalized_xyz = {
            xyz.x / length,
            xyz.y / length,
            xyz.z / length
        };
        return {normalized_xyz, w / length};
    }
};

template <typename T>
[[nodiscard]] auto QuaternionRotate(const Quaternion<T>& q, const Vec3<T>& v) {
    // Use component-wise operations to avoid ambiguity
    Vec3<T> cross1 = Cross(q.xyz, v);
    Vec3<T> scaled_v = v;
    scaled_v.x *= q.w;
    scaled_v.y *= q.w;
    scaled_v.z *= q.w;
    
    Vec3<T> term = cross1;
    term.x += scaled_v.x;
    term.y += scaled_v.y;
    term.z += scaled_v.z;
    
    Vec3<T> cross2 = Cross(q.xyz, term);
    cross2.x *= 2;
    cross2.y *= 2;
    cross2.z *= 2;
    
    Vec3<T> result;
    result.x = v.x + cross2.x;
    result.y = v.y + cross2.y;
    result.z = v.z + cross2.z;
    
    return result;
}

[[nodiscard]] inline Quaternion<float> MakeQuaternion(const Vec3<float>& axis, float angle) {
    return {axis * std::sin(angle / 2), std::cos(angle / 2)};
}

} // namespace Common
