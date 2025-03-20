// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Copyright 2014 Tony Wasserka
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the owner nor the names of its contributors may
//       be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cmath>
#include <cstring>
#include <type_traits>
#include <boost/serialization/access.hpp>

// Check for ARM NEON support
#if defined(__ARM_NEON) || defined(__aarch64__)
#define CITRA_USE_NEON 1
#include <arm_neon.h>
#endif

namespace Common {


template <typename T>
class Vec2;
template <typename T>
class Vec3;
template <typename T>
class Vec4;

// Forward declare NEON namespace for ARM NEON optimized functions
#ifdef CITRA_USE_NEON
namespace NEON {
// Forward declarations for NEON optimized functions
inline float Dot3(const Vec3<float>& vec1, const Vec3<float>& vec2);
inline float Dot4(const Vec4<float>& vec1, const Vec4<float>& vec2);
inline float Dot2(const Vec2<float>& vec1, const Vec2<float>& vec2);
inline Vec3<float> Cross3(const Vec3<float>& vec1, const Vec3<float>& vec2);
inline Vec3<float> Multiply3(const Vec3<float>& vec, float scalar);
inline Vec4<float> Multiply4(const Vec4<float>& vec, float scalar);
inline float Length3(const Vec3<float>& vec);
inline float Length2_3(const Vec3<float>& vec);
inline Vec3<float> Add3(const Vec3<float>& vec1, const Vec3<float>& vec2);
inline Vec3<float> Subtract3(const Vec3<float>& vec1, const Vec3<float>& vec2);
inline Vec3<float> Multiply3Scalar(const Vec3<float>& vec, float scalar);
inline Vec3<float> Multiply3(const Vec3<float>& vec1, const Vec3<float>& vec2);
inline Vec3<float> Divide3(const Vec3<float>& vec, float scalar);
inline bool Equals3(const Vec3<float>& vec1, const Vec3<float>& vec2);
inline bool NotEquals3(const Vec3<float>& vec1, const Vec3<float>& vec2);
inline Vec3<float> Lerp3(const Vec3<float>& start, const Vec3<float>& end, float t);
inline Vec3<float> BilinearInterp3(const Vec3<float>& x00, const Vec3<float>& x01, const Vec3<float>& x10, const Vec3<float>& x11, float s, float t);
inline Vec3<float> BilinearInterp3Fast(const Vec3<float>& x00, const Vec3<float>& x01, const Vec3<float>& x10, const Vec3<float>& x11, float s, float t);
inline Vec4<float> Add4(const Vec4<float>& vec1, const Vec4<float>& vec2);
inline Vec4<float> Subtract4(const Vec4<float>& vec1, const Vec4<float>& vec2);
inline Vec4<float> Multiply4Scalar(const Vec4<float>& vec, float scalar);
inline Vec4<float> Multiply4(const Vec4<float>& vec1, const Vec4<float>& vec2);
inline Vec4<float> Divide4(const Vec4<float>& vec, float scalar);
}
#endif

template <typename T>
class Vec2 {
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar& x;
        ar& y;
    }

public:
    T x;
    T y;

    T* AsArray() {
        return &x;
    }

    const T* AsArray() const {
        return &x;
    }

    constexpr Vec2() = default;
    constexpr Vec2(const T& x_, const T& y_) : x(x_), y(y_) {}

    template <typename T2>
    [[nodiscard]] constexpr Vec2<T2> Cast() const {
        return Vec2<T2>(static_cast<T2>(x), static_cast<T2>(y));
    }

    [[nodiscard]] static constexpr Vec2 AssignToAll(const T& f) {
        return Vec2{f, f};
    }

    [[nodiscard]] constexpr Vec2<decltype(T{} + T{})> operator+(const Vec2& other) const {
        return {x + other.x, y + other.y};
    }
    constexpr Vec2& operator+=(const Vec2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    [[nodiscard]] constexpr Vec2<decltype(T{} - T{})> operator-(const Vec2& other) const {
        return {x - other.x, y - other.y};
    }
    constexpr Vec2& operator-=(const Vec2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    template <typename U = T>
    [[nodiscard]] constexpr Vec2<std::enable_if_t<std::is_signed_v<U>, U>> operator-() const {
        return {-x, -y};
    }
    [[nodiscard]] constexpr Vec2<decltype(T{} * T{})> operator*(const Vec2& other) const {
        return {x * other.x, y * other.y};
    }

    template <typename V>
    [[nodiscard]] constexpr Vec2<decltype(T{} * V{})> operator*(const V& f) const {
        return {x * f, y * f};
    }

    template <typename V>
    constexpr Vec2& operator*=(const V& f) {
        *this = *this * f;
        return *this;
    }

    template <typename V>
    [[nodiscard]] constexpr Vec2<decltype(T{} / V{})> operator/(const V& f) const {
        return {x / f, y / f};
    }

    template <typename V>
    constexpr Vec2& operator/=(const V& f) {
        *this = *this / f;
        return *this;
    }

    [[nodiscard]] constexpr T Length2() const {
        return x * x + y * y;
    }

    [[nodiscard]] constexpr bool operator!=(const Vec2& other) const {
        return std::memcmp(AsArray(), other.AsArray(), sizeof(Vec2)) != 0;
    }

    [[nodiscard]] constexpr bool operator==(const Vec2& other) const {
        return std::memcmp(AsArray(), other.AsArray(), sizeof(Vec2)) == 0;
    }

    // Only implemented for T=float
    [[nodiscard]] float Length() const;
    float Normalize(); // returns the previous length, which is often useful

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }
    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    constexpr void SetZero() {
        x = 0;
        y = 0;
    }

    // Common aliases: UV (texel coordinates), ST (texture coordinates)
    [[nodiscard]] constexpr T& u() {
        return x;
    }
    [[nodiscard]] constexpr T& v() {
        return y;
    }
    [[nodiscard]] constexpr T& s() {
        return x;
    }
    [[nodiscard]] constexpr T& t() {
        return y;
    }

    [[nodiscard]] constexpr const T& u() const {
        return x;
    }
    [[nodiscard]] constexpr const T& v() const {
        return y;
    }
    [[nodiscard]] constexpr const T& s() const {
        return x;
    }
    [[nodiscard]] constexpr const T& t() const {
        return y;
    }

    // swizzlers - create a subvector of specific components
    [[nodiscard]] constexpr Vec2 yx() const {
        return Vec2(y, x);
    }
    [[nodiscard]] constexpr Vec2 vu() const {
        return Vec2(y, x);
    }
    [[nodiscard]] constexpr Vec2 ts() const {
        return Vec2(y, x);
    }
};

template <typename T, typename V>
[[nodiscard]] constexpr Vec2<T> operator*(const V& f, const Vec2<T>& vec) {
    return Vec2<T>(f * vec.x, f * vec.y);
}

using Vec2f = Vec2<float>;
using Vec2i = Vec2<int>;
using Vec2u = Vec2<unsigned int>;

template <>
inline float Vec2<float>::Length() const {
    return std::sqrt(x * x + y * y);
}

template <>
inline float Vec2<float>::Normalize() {
    float length = Length();
    *this /= length;
    return length;
}

template <typename T>
class Vec3 {
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar& x;
        ar& y;
        ar& z;
    }

public:
    T x;
    T y;
    T z;

    T* AsArray() {
        return &x;
    }

    const T* AsArray() const {
        return &x;
    }

    constexpr Vec3() = default;
    constexpr Vec3(const T& x_, const T& y_, const T& z_) : x(x_), y(y_), z(z_) {}

    template <typename T2>
    [[nodiscard]] constexpr Vec3<T2> Cast() const {
        return Vec3<T2>(static_cast<T2>(x), static_cast<T2>(y), static_cast<T2>(z));
    }

    [[nodiscard]] static constexpr Vec3 AssignToAll(const T& f) {
        return Vec3(f, f, f);
    }

    [[nodiscard]] Vec3<decltype(T{} + T{})> operator+(const Vec3& other) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float>) {
            return NEON::Add3(*this, other);
        } else {
            return {x + other.x, y + other.y, z + other.z};
        }
#else
        return {x + other.x, y + other.y, z + other.z};
#endif
    }

    constexpr Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    [[nodiscard]] Vec3<decltype(T{} - T{})> operator-(const Vec3& other) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float>) {
            return NEON::Subtract3(*this, other);
        } else {
            return {x - other.x, y - other.y, z - other.z};
        }
#else
        return {x - other.x, y - other.y, z - other.z};
#endif
    }

    constexpr Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    template <typename U = T>
    [[nodiscard]] constexpr Vec3<std::enable_if_t<std::is_signed_v<U>, U>> operator-() const {
        return {-x, -y, -z};
    }

    [[nodiscard]] constexpr Vec3<decltype(T{} * T{})> operator*(const Vec3& other) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float>) {
            return NEON::Multiply3(*this, other);
        } else {
            return {x * other.x, y * other.y, z * other.z};
        }
#else
        return {x * other.x, y * other.y, z * other.z};
#endif
    }

    template <typename V>
    [[nodiscard]] constexpr Vec3<decltype(T{} * V{})> operator*(const V& f) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float> && std::is_same_v<V, float>) {
            return NEON::Multiply3Scalar(*this, f);
        } else {
            return {x * f, y * f, z * f};
        }
#else
        return {x * f, y * f, z * f};
#endif
    }

    template <typename V>
    constexpr Vec3& operator*=(const V& f) {
        *this = *this * f;
        return *this;
    }
    template <typename V>
    [[nodiscard]] constexpr Vec3<decltype(T{} / V{})> operator/(const V& f) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float> && std::is_same_v<V, float>) {
            return NEON::Divide3(*this, f);
        } else {
            return {x / f, y / f, z / f};
        }
#else
        return {x / f, y / f, z / f};
#endif
    }

    template <typename V>
    constexpr Vec3& operator/=(const V& f) {
        *this = *this / f;
        return *this;
    }

    [[nodiscard]] constexpr bool operator!=(const Vec3& other) const {
        return std::memcmp(AsArray(), other.AsArray(), sizeof(Vec3)) != 0;
    }

    [[nodiscard]] constexpr bool operator==(const Vec3& other) const {
        return std::memcmp(AsArray(), other.AsArray(), sizeof(Vec3)) == 0;
    }

    [[nodiscard]] constexpr T Length2() const {
        return x * x + y * y + z * z;
    }

    // Only implemented for T=float
    [[nodiscard]] float Length() const;
    [[nodiscard]] Vec3 Normalized() const;
    float Normalize(); // returns the previous length, which is often useful

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    constexpr void SetZero() {
        x = 0;
        y = 0;
        z = 0;
    }

    // Common aliases: UVW (texel coordinates), RGB (colors), STQ (texture coordinates)
    [[nodiscard]] constexpr T& u() {
        return x;
    }
    [[nodiscard]] constexpr T& v() {
        return y;
    }
    [[nodiscard]] constexpr T& w() {
        return z;
    }

    [[nodiscard]] constexpr T& r() {
        return x;
    }
    [[nodiscard]] constexpr T& g() {
        return y;
    }
    [[nodiscard]] constexpr T& b() {
        return z;
    }

    [[nodiscard]] constexpr T& s() {
        return x;
    }
    [[nodiscard]] constexpr T& t() {
        return y;
    }
    [[nodiscard]] constexpr T& q() {
        return z;
    }

    [[nodiscard]] constexpr const T& u() const {
        return x;
    }
    [[nodiscard]] constexpr const T& v() const {
        return y;
    }
    [[nodiscard]] constexpr const T& w() const {
        return z;
    }

    [[nodiscard]] constexpr const T& r() const {
        return x;
    }
    [[nodiscard]] constexpr const T& g() const {
        return y;
    }
    [[nodiscard]] constexpr const T& b() const {
        return z;
    }

    [[nodiscard]] constexpr const T& s() const {
        return x;
    }
    [[nodiscard]] constexpr const T& t() const {
        return y;
    }
    [[nodiscard]] constexpr const T& q() const {
        return z;
    }

// swizzlers - create a subvector of specific components
// e.g. Vec2 uv() { return Vec2(x,y); }
// _DEFINE_SWIZZLER2 defines a single such function, DEFINE_SWIZZLER2 defines all of them for all
// component names (x<->r) and permutations (xy<->yx)
#define _DEFINE_SWIZZLER2(a, b, name)                                                              \
    [[nodiscard]] constexpr Vec2<T> name() const {                                                 \
        return Vec2<T>(a, b);                                                                      \
    }
#define DEFINE_SWIZZLER2(a, b, a2, b2, a3, b3, a4, b4)                                             \
    _DEFINE_SWIZZLER2(a, b, a##b);                                                                 \
    _DEFINE_SWIZZLER2(a, b, a2##b2);                                                               \
    _DEFINE_SWIZZLER2(a, b, a3##b3);                                                               \
    _DEFINE_SWIZZLER2(a, b, a4##b4);                                                               \
    _DEFINE_SWIZZLER2(b, a, b##a);                                                                 \
    _DEFINE_SWIZZLER2(b, a, b2##a2);                                                               \
    _DEFINE_SWIZZLER2(b, a, b3##a3);                                                               \
    _DEFINE_SWIZZLER2(b, a, b4##a4)

    DEFINE_SWIZZLER2(x, y, r, g, u, v, s, t);
    DEFINE_SWIZZLER2(x, z, r, b, u, w, s, q);
    DEFINE_SWIZZLER2(y, z, g, b, v, w, t, q);
#undef DEFINE_SWIZZLER2
#undef _DEFINE_SWIZZLER2
};

template <typename T, typename V>
[[nodiscard]] constexpr Vec3<T> operator*(const V& f, const Vec3<T>& vec) {
    return Vec3<T>(f * vec.x, f * vec.y, f * vec.z);
}

template <>
inline float Vec3<float>::Length() const {
#ifdef CITRA_USE_NEON
    return NEON::Length3(*this);
#else
    return std::sqrt(x * x + y * y + z * z);
#endif
}

#ifdef CITRA_USE_NEON
template <>
inline Vec3<float> Vec3<float>::Normalized() const {
    float length = Length();
    if (length > 0.0f) {
        return *this / length;
    }
    return *this;
}
#else
template <>
inline Vec3<float> Vec3<float>::Normalized() const {
    return *this / Length();
}
#endif

#ifdef CITRA_USE_NEON
template <>
inline float Vec3<float>::Normalize() {
    float length = Length();
    if (length > 0.0f) {
        *this /= length;
    }
    return length;
}
#else
template <>
inline float Vec3<float>::Normalize() {
    float length = Length();
    *this /= length;
    return length;
}
#endif

using Vec3f = Vec3<float>;
using Vec3i = Vec3<int>;
using Vec3u = Vec3<unsigned int>;

template <typename T>
class Vec4 {
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar& x;
        ar& y;
        ar& z;
        ar& w;
    }

public:
    T x;
    T y;
    T z;
    T w;

    T* AsArray() {
        return &x;
    }

    const T* AsArray() const {
        return &x;
    }

    constexpr Vec4() = default;
    constexpr Vec4(const T& x_, const T& y_, const T& z_, const T& w_)
        : x(x_), y(y_), z(z_), w(w_) {}

    template <typename T2>
    [[nodiscard]] constexpr Vec4<T2> Cast() const {
        return Vec4<T2>(static_cast<T2>(x), static_cast<T2>(y), static_cast<T2>(z),
                        static_cast<T2>(w));
    }

    [[nodiscard]] static constexpr Vec4 AssignToAll(const T& f) {
        return Vec4(f, f, f, f);
    }

    [[nodiscard]] Vec4<decltype(T{} + T{})> operator+(const Vec4& other) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float>) {
            return NEON::Add4(*this, other);
        } else {
            return {x + other.x, y + other.y, z + other.z, w + other.w};
        }
#else
        return {x + other.x, y + other.y, z + other.z, w + other.w};
#endif
    }

    constexpr Vec4& operator+=(const Vec4& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }

    [[nodiscard]] Vec4<decltype(T{} - T{})> operator-(const Vec4& other) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float>) {
            return NEON::Subtract4(*this, other);
        } else {
            return {x - other.x, y - other.y, z - other.z, w - other.w};
        }
#else
        return {x - other.x, y - other.y, z - other.z, w - other.w};
#endif
    }

    constexpr Vec4& operator-=(const Vec4& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }

    template <typename U = T>
    [[nodiscard]] constexpr Vec4<std::enable_if_t<std::is_signed_v<U>, U>> operator-() const {
        return {-x, -y, -z, -w};
    }

    [[nodiscard]] Vec4<decltype(T{} * T{})> operator*(const Vec4& other) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float>) {
            return NEON::Multiply4(*this, other);
        } else {
            return {x * other.x, y * other.y, z * other.z, w * other.w};
        }
#else
        return {x * other.x, y * other.y, z * other.z, w * other.w};
#endif
    }

    template <typename V>
    [[nodiscard]] Vec4<decltype(T{} * V{})> operator*(const V& f) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float> && std::is_same_v<V, float>) {
            return NEON::Multiply4Scalar(*this, f);
        } else {
            return {x * f, y * f, z * f, w * f};
        }
#else
        return {x * f, y * f, z * f, w * f};
#endif
    }

    template <typename V>
    constexpr Vec4& operator*=(const V& f) {
        *this = *this * f;
        return *this;
    }

    template <typename V>
    [[nodiscard]] Vec4<decltype(T{} / V{})> operator/(const V& f) const {
#if defined(CITRA_USE_NEON)
        if constexpr (std::is_same_v<T, float> && std::is_same_v<V, float>) {
            return NEON::Divide4(*this, f);
        } else {
            return {x / f, y / f, z / f, w / f};
        }
#else
        return {x / f, y / f, z / f, w / f};
#endif
    }

    template <typename V>
    constexpr Vec4& operator/=(const V& f) {
        *this = *this / f;
        return *this;
    }

    [[nodiscard]] constexpr bool operator!=(const Vec4& other) const {
        return std::memcmp(AsArray(), other.AsArray(), sizeof(Vec4)) != 0;
    }

    [[nodiscard]] constexpr bool operator==(const Vec4& other) const {
        return std::memcmp(AsArray(), other.AsArray(), sizeof(Vec4)) == 0;
    }

    [[nodiscard]] constexpr T Length2() const {
        return x * x + y * y + z * z + w * w;
    }

    // Only implemented for T=float
    [[nodiscard]] float Length() const;
    [[nodiscard]] Vec4 Normalized() const;
    float Normalize(); // returns the previous length, which is often useful

    [[nodiscard]] constexpr T& operator[](std::size_t i) {
        return *((&x) + i);
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i) const {
        return *((&x) + i);
    }

    constexpr void SetZero() {
        x = 0;
        y = 0;
        z = 0;
        w = 0;
    }

    // Common alias: RGBA (colors)
    [[nodiscard]] constexpr T& r() {
        return x;
    }
    [[nodiscard]] constexpr T& g() {
        return y;
    }
    [[nodiscard]] constexpr T& b() {
        return z;
    }
    [[nodiscard]] constexpr T& a() {
        return w;
    }

    [[nodiscard]] constexpr const T& r() const {
        return x;
    }
    [[nodiscard]] constexpr const T& g() const {
        return y;
    }
    [[nodiscard]] constexpr const T& b() const {
        return z;
    }
    [[nodiscard]] constexpr const T& a() const {
        return w;
    }

// Swizzlers - Create a subvector of specific components
// e.g. Vec2 uv() { return Vec2(x,y); }

// _DEFINE_SWIZZLER2 defines a single such function
// DEFINE_SWIZZLER2_COMP1 defines one-component functions for all component names (x<->r)
// DEFINE_SWIZZLER2_COMP2 defines two component functions for all component names (x<->r) and
// permutations (xy<->yx)
#define _DEFINE_SWIZZLER2(a, b, name)                                                              \
    [[nodiscard]] constexpr Vec2<T> name() const {                                                 \
        return Vec2<T>(a, b);                                                                      \
    }
#define DEFINE_SWIZZLER2_COMP1(a, a2)                                                              \
    _DEFINE_SWIZZLER2(a, a, a##a);                                                                 \
    _DEFINE_SWIZZLER2(a, a, a2##a2)
#define DEFINE_SWIZZLER2_COMP2(a, b, a2, b2)                                                       \
    _DEFINE_SWIZZLER2(a, b, a##b);                                                                 \
    _DEFINE_SWIZZLER2(a, b, a2##b2);                                                               \
    _DEFINE_SWIZZLER2(b, a, b##a);                                                                 \
    _DEFINE_SWIZZLER2(b, a, b2##a2)

    DEFINE_SWIZZLER2_COMP2(x, y, r, g);
    DEFINE_SWIZZLER2_COMP2(x, z, r, b);
    DEFINE_SWIZZLER2_COMP2(x, w, r, a);
    DEFINE_SWIZZLER2_COMP2(y, z, g, b);
    DEFINE_SWIZZLER2_COMP2(y, w, g, a);
    DEFINE_SWIZZLER2_COMP2(z, w, b, a);
    DEFINE_SWIZZLER2_COMP1(x, r);
    DEFINE_SWIZZLER2_COMP1(y, g);
    DEFINE_SWIZZLER2_COMP1(z, b);
    DEFINE_SWIZZLER2_COMP1(w, a);
#undef DEFINE_SWIZZLER2_COMP1
#undef DEFINE_SWIZZLER2_COMP2
#undef _DEFINE_SWIZZLER2

#define _DEFINE_SWIZZLER3(a, b, c, name)                                                           \
    [[nodiscard]] constexpr Vec3<T> name() const {                                                 \
        return Vec3<T>(a, b, c);                                                                   \
    }
#define DEFINE_SWIZZLER3_COMP1(a, a2)                                                              \
    _DEFINE_SWIZZLER3(a, a, a, a##a##a);                                                           \
    _DEFINE_SWIZZLER3(a, a, a, a2##a2##a2)
#define DEFINE_SWIZZLER3_COMP3(a, b, c, a2, b2, c2)                                                \
    _DEFINE_SWIZZLER3(a, b, c, a##b##c);                                                           \
    _DEFINE_SWIZZLER3(a, c, b, a##c##b);                                                           \
    _DEFINE_SWIZZLER3(b, a, c, b##a##c);                                                           \
    _DEFINE_SWIZZLER3(b, c, a, b##c##a);                                                           \
    _DEFINE_SWIZZLER3(c, a, b, c##a##b);                                                           \
    _DEFINE_SWIZZLER3(c, b, a, c##b##a);                                                           \
    _DEFINE_SWIZZLER3(a, b, c, a2##b2##c2);                                                        \
    _DEFINE_SWIZZLER3(a, c, b, a2##c2##b2);                                                        \
    _DEFINE_SWIZZLER3(b, a, c, b2##a2##c2);                                                        \
    _DEFINE_SWIZZLER3(b, c, a, b2##c2##a2);                                                        \
    _DEFINE_SWIZZLER3(c, a, b, c2##a2##b2);                                                        \
    _DEFINE_SWIZZLER3(c, b, a, c2##b2##a2)

    DEFINE_SWIZZLER3_COMP3(x, y, z, r, g, b);
    DEFINE_SWIZZLER3_COMP3(x, y, w, r, g, a);
    DEFINE_SWIZZLER3_COMP3(x, z, w, r, b, a);
    DEFINE_SWIZZLER3_COMP3(y, z, w, g, b, a);
    DEFINE_SWIZZLER3_COMP1(x, r);
    DEFINE_SWIZZLER3_COMP1(y, g);
    DEFINE_SWIZZLER3_COMP1(z, b);
    DEFINE_SWIZZLER3_COMP1(w, a);
#undef DEFINE_SWIZZLER3_COMP1
#undef DEFINE_SWIZZLER3_COMP3
#undef _DEFINE_SWIZZLER3
};

template <typename T, typename V>
[[nodiscard]] constexpr Vec4<decltype(V{} * T{})> operator*(const V& f, const Vec4<T>& vec) {
    return {f * vec.x, f * vec.y, f * vec.z, f * vec.w};
}

using Vec4f = Vec4<float>;
using Vec4i = Vec4<int>;
using Vec4u = Vec4<unsigned int>;

template <typename T>
constexpr decltype(T{} * T{} + T{} * T{}) Dot(const Vec2<T>& a, const Vec2<T>& b) {
#if defined(CITRA_USE_NEON)
    return NEON::Dot2(a, b);
#else
    return a.x * b.x + a.y * b.y;
#endif
}

template <typename T>
[[nodiscard]] constexpr decltype(T{} * T{} + T{} * T{}) Dot(const Vec3<T>& a, const Vec3<T>& b) {
#if defined(CITRA_USE_NEON)
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Dot3(a, b);
    } else {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
#else
    return a.x * b.x + a.y * b.y + a.z * b.z;
#endif
}

template <typename T>
[[nodiscard]] constexpr decltype(T{} * T{} + T{} * T{}) Dot(const Vec4<T>& a, const Vec4<T>& b) {
#if defined(CITRA_USE_NEON)
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Dot4(a, b);
    } else {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
#else
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
#endif
}

template <typename T>
[[nodiscard]] constexpr Vec3<decltype(T{} * T{} - T{} * T{})> Cross(const Vec3<T>& a,
                                                                    const Vec3<T>& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// linear interpolation via float: 0.0=begin, 1.0=end
template <typename X>
[[nodiscard]] constexpr decltype(X{} * float{} + X{} * float{})
    Lerp(const X& begin, const X& end, const float t) {
#if defined(CITRA_USE_NEON)
    if constexpr (std::is_same_v<X, Vec3f>) {
        return NEON::Lerp3(begin, end, t);
    } else if constexpr (std::is_same_v<X, Vec4f>) {
        Vec4f result;
        result.x = begin.x * (1.0f - t) + end.x * t;
        result.y = begin.y * (1.0f - t) + end.y * t;
        result.z = begin.z * (1.0f - t) + end.z * t;
        result.w = begin.w * (1.0f - t) + end.w * t;
        return result;
    }
#else
    return begin * (1.f - t) + end * t;
#endif
}

// linear interpolation via int: 0=begin, base=end
template <typename X, int base>
[[nodiscard]] constexpr decltype((X{} * int{} + X{} * int{}) / base) LerpInt(const X& begin,
                                                                             const X& end,
                                                                             const int t) {
    return (begin * (base - t) + end * t) / base;
}

// bilinear interpolation. s is for interpolating x00-x01 and x10-x11, and t is for the second
// interpolation.
template <typename X>
[[nodiscard]] constexpr auto BilinearInterp(const X& x00, const X& x01, const X& x10, const X& x11,
                                             const float s, const float t) {
#if defined(CITRA_USE_NEON)
    if constexpr (std::is_same_v<X, Vec3f>) {
        return NEON::BilinearInterp3Fast(x00, x01, x10, x11, s, t);
    } else if constexpr (std::is_same_v<X, Vec4f>) {
        Vec4f y0;
        y0.x = x00.x * (1.0f - s) + x01.x * s;
        y0.y = x00.y * (1.0f - s) + x01.y * s;
        y0.z = x00.z * (1.0f - s) + x01.z * s;
        y0.w = x00.w * (1.0f - s) + x01.w * s;

        Vec4f y1;
        y1.x = x10.x * (1.0f - s) + x11.x * s;
        y1.y = x10.y * (1.0f - s) + x11.y * s;
        y1.z = x10.z * (1.0f - s) + x11.z * s;
        y1.w = x10.w * (1.0f - s) + x11.w * s;

        Vec4f result;
        result.x = y0.x * (1.0f - t) + y1.x * t;
        result.y = y0.y * (1.0f - t) + y1.y * t;
        result.z = y0.z * (1.0f - t) + y1.z * t;
        result.w = y0.w * (1.0f - t) + y1.w * t;

        return result;
    } else if constexpr (std::is_same_v<X, float>) {
        float y0 = x00 * (1.0f - s) + x01 * s;
        float y1 = x10 * (1.0f - s) + x11 * s;
        return y0 * (1.0f - t) + y1 * t;
    }
#else
    auto y0 = Lerp(x00, x01, s);
    auto y1 = Lerp(x10, x11, s);
    return Lerp(y0, y1, t);
#endif
}

// Utility vector factories
template <typename T>
[[nodiscard]] constexpr Vec2<T> MakeVec(const T& x, const T& y) {
    return Vec2<T>{x, y};
}

template <typename T>
[[nodiscard]] constexpr Vec3<T> MakeVec(const T& x, const T& y, const T& z) {
    return Vec3<T>{x, y, z};
}

template <typename T>
[[nodiscard]] constexpr Vec4<T> MakeVec(const T& x, const T& y, const Vec2<T>& zw) {
    return MakeVec(x, y, zw[0], zw[1]);
}

template <typename T>
[[nodiscard]] constexpr Vec3<T> MakeVec(const Vec2<T>& xy, const T& z) {
    return MakeVec(xy[0], xy[1], z);
}

template <typename T>
[[nodiscard]] constexpr Vec3<T> MakeVec(const T& x, const Vec2<T>& yz) {
    return MakeVec(x, yz[0], yz[1]);
}

template <typename T>
[[nodiscard]] constexpr Vec4<T> MakeVec(const T& x, const T& y, const T& z, const T& w) {
    return Vec4<T>{x, y, z, w};
}

template <typename T>
[[nodiscard]] constexpr Vec4<T> MakeVec(const Vec2<T>& xy, const T& z, const T& w) {
    return MakeVec(xy[0], xy[1], z, w);
}

template <typename T>
[[nodiscard]] constexpr Vec4<T> MakeVec(const T& x, const Vec2<T>& yz, const T& w) {
    return MakeVec(x, yz[0], yz[1], w);
}

// NOTE: This has priority over "Vec2<Vec2<T>> MakeVec(const Vec2<T>& x, const Vec2<T>& y)".
//       Even if someone wanted to use an odd object like Vec2<Vec2<T>>, the compiler would error
//       out soon enough due to misuse of the returned structure.
template <typename T>
[[nodiscard]] constexpr Vec4<T> MakeVec(const Vec2<T>& xy, const Vec2<T>& zw) {
    return MakeVec(xy[0], xy[1], zw[0], zw[1]);
}

template <typename T>
[[nodiscard]] constexpr Vec4<T> MakeVec(const Vec3<T>& xyz, const T& w) {
    return MakeVec(xyz[0], xyz[1], xyz[2], w);
}

template <typename T>
[[nodiscard]] constexpr Vec4<T> MakeVec(const T& x, const Vec3<T>& yzw) {
    return MakeVec(x, yzw[0], yzw[1], yzw[2]);
}

// Vec4<float> specializations
template <>
inline float Vec4<float>::Length() const {
#if defined(CITRA_USE_NEON)
    return std::sqrt(x * x + y * y + z * z + w * w);
#else
    return std::sqrt(x * x + y * y + z * z + w * w);
#endif
}

#ifdef CITRA_USE_NEON
template <>
inline Vec4<float> Vec4<float>::Normalized() const {
    float length = Length();
    if (length > 0.0f) {
        return *this / length;
    }
    return *this;
}
#else
template <>
inline Vec4<float> Vec4<float>::Normalized() const {
    return *this / Length();
}
#endif

#ifdef CITRA_USE_NEON
template <>
inline float Vec4<float>::Normalize() {
    float length = Length();
    if (length > 0.0f) {
        *this /= length;
    }
    return length;
}
#else
template <>
inline float Vec4<float>::Normalize() {
    float length = Length();
    *this /= length;
    return length;
}
#endif

// Specialized implementations for Vec3f and Vec4f using NEON optimizations

// Vec3f specialized implementations
// Removed ambiguous global operator+ and operator- for Vec3f
// Use the member operators from Vec3 class template instead

// Removed ambiguous global operator* for Vec3f
// Use the member operators from Vec3 class template instead

// Removed ambiguous global operator/ for Vec3f
// Use the member operators from Vec3 class template instead

// NEON-optimized operator== is now handled in the Vec3 class template

// NEON-optimized operator!= is now handled in the Vec3 class template

#ifdef CITRA_USE_NEON
inline float Dot(const Vec3f& a, const Vec3f& b) {
    return NEON::Dot3(a, b);
}
#endif

// Vec4f specialized implementations
// Removed ambiguous operator+ for Vec4f

// Removed ambiguous global operators for Vec4f
// Use the member operators from Vec4 class template instead

// NEON-optimized equality operator for Vec4f is handled in the class definition

// NEON-optimized inequality operator for Vec4f is handled in the class definition

#ifdef CITRA_USE_NEON
inline float Dot(const Vec4f& a, const Vec4f& b) {
    return NEON::Dot4(a, b);
}
#endif

// Specialized Lerp for Vec3f
inline Vec3f Lerp(const Vec3f& begin, const Vec3f& end, float t) {
#ifdef CITRA_USE_NEON
    return NEON::Lerp3(begin, end, t);
#else
    return begin * (1.0f - t) + end * t;
#endif
}

// NEON-optimized Lerp for Vec4f is handled in the template specialization

// Specialized BilinearInterp for Vec3f
inline Vec3f BilinearInterp(const Vec3f& x00, const Vec3f& x01, const Vec3f& x10, const Vec3f& x11,
                           float s, float t) {
#ifdef CITRA_USE_NEON
    return NEON::BilinearInterp3Fast(x00, x01, x10, x11, s, t);
#else
    return Lerp(Lerp(x00, x01, s), Lerp(x10, x11, s), t);
#endif
}

// Specialized BilinearInterp for Vec4f
inline Vec4f BilinearInterp(const Vec4f& x00, const Vec4f& x01, const Vec4f& x10, const Vec4f& x11,
                           float s, float t) {
#ifdef CITRA_USE_NEON
    Vec4f y0;
    y0.x = x00.x * (1.0f - s) + x01.x * s;
    y0.y = x00.y * (1.0f - s) + x01.y * s;
    y0.z = x00.z * (1.0f - s) + x01.z * s;
    y0.w = x00.w * (1.0f - s) + x01.w * s;

    Vec4f y1;
    y1.x = x10.x * (1.0f - s) + x11.x * s;
    y1.y = x10.y * (1.0f - s) + x11.y * s;
    y1.z = x10.z * (1.0f - s) + x11.z * s;
    y1.w = x10.w * (1.0f - s) + x11.w * s;

    Vec4f result;
    result.x = y0.x * (1.0f - t) + y1.x * t;
    result.y = y0.y * (1.0f - t) + y1.y * t;
    result.z = y0.z * (1.0f - t) + y1.z * t;
    result.w = y0.w * (1.0f - t) + y1.w * t;

    return result;
#else
    return Lerp(Lerp(x00, x01, s), Lerp(x10, x11, s), t);
#endif
}

#ifdef CITRA_USE_NEON
// Add the missing NEON functions

/**
 * Multiplies a 3D vector by a scalar using NEON intrinsics
 * @param vec Vector to scale
 * @param scalar Scalar value to multiply by
 * @return Scaled vector
 */
inline Vec3f NEON::Multiply3(const Vec3f& vec, float scalar) {
    // Load vector components into NEON register
    float32x4_t v = vdupq_n_f32(0.0f);
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);

    // Create scalar register with all lanes set to scalar value
    float32x4_t s = vdupq_n_f32(scalar);

    // Multiply vector by scalar
    float32x4_t result = vmulq_f32(v, s);

    // Extract results
    Vec3f scaled;
    scaled.x = vgetq_lane_f32(result, 0);
    scaled.y = vgetq_lane_f32(result, 1);
    scaled.z = vgetq_lane_f32(result, 2);

    return scaled;
}

/**
 * Performs a fast dot product between two 4D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Dot product result
 */
inline float NEON::Dot4(const Vec4f& vec1, const Vec4f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);

    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    v1 = vsetq_lane_f32(vec1.w, v1, 3);

    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    v2 = vsetq_lane_f32(vec2.w, v2, 3);

    // Multiply vectors component-wise
    float32x4_t mul = vmulq_f32(v1, v2);

    // Sum components horizontally
    float32x2_t sum = vpadd_f32(vget_low_f32(mul), vget_high_f32(mul));
    sum = vpadd_f32(sum, sum);

    // Extract result
    return vget_lane_f32(sum, 0);
}

#endif // CITRA_USE_NEON

// Helper functions to safely use optimized operations

/**
 * Safely adds two vectors with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline Vec3<T> SafeAdd(const Vec3<T>& a, const Vec3<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Add3(a, b);
    }
#endif
    return a + b;
}

/**
 * Safely adds a vector to another vector (in-place) with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline void SafeAddAssign(Vec3<T>& a, const Vec3<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        a = NEON::Add3(a, b);
        return;
    }
#endif
    a += b;
}

/**
 * Safely subtracts two vectors with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline Vec3<T> SafeSubtract(const Vec3<T>& a, const Vec3<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Subtract3(a, b);
    }
#endif
    return a - b;
}

/**
 * Safely multiplies a vector by a scalar with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline Vec3<T> SafeMultiply(const Vec3<T>& a, T scalar) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Multiply3Scalar(a, scalar);
    }
#endif
    return a * scalar;
}

/**
 * Safely multiplies two vectors component-wise with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline Vec3<T> SafeMultiply(const Vec3<T>& a, const Vec3<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Multiply3(a, b);
    }
#endif
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

/**
 * Safely calculates the dot product of two vectors with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline auto SafeDot(const Vec3<T>& a, const Vec3<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Dot3(a, b);
    }
#endif
    return Dot(a, b);
}

/**
 * Safely adds two 4D vectors with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline Vec4<T> SafeAdd(const Vec4<T>& a, const Vec4<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Add4(a, b);
    }
#endif
    return a + b;
}

/**
 * Safely adds a vector to another vector (in-place) with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline void SafeAddAssign(Vec4<T>& a, const Vec4<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        a = NEON::Add4(a, b);
        return;
    }
#endif
    a += b;
}

/**
 * Safely subtracts two 4D vectors with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline Vec4<T> SafeSubtract(const Vec4<T>& a, const Vec4<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Subtract4(a, b);
    }
#endif
    return a - b;
}

/**
 * Safely calculates the dot product of two 4D vectors with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline T SafeDot(const Vec4<T>& a, const Vec4<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Dot4(a, b);
    }
#endif
    return Dot(a, b);
}

/**
 * Safely multiplies a 4D vector by a scalar with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline Vec4<T> SafeMultiply(const Vec4<T>& a, T scalar) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Multiply4Scalar(a, scalar);
    }
#endif
    return a * scalar;
}

/**
 * Safely multiplies two 4D vectors component-wise with proper type checking
 * Uses NEON optimization when available for float vectors
 */
template <typename T>
inline Vec4<T> SafeMultiply(const Vec4<T>& a, const Vec4<T>& b) {
#ifdef CITRA_USE_NEON
    if constexpr (std::is_same_v<T, float>) {
        return NEON::Multiply4(a, b);
    }
#endif
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

} // namespace Common
