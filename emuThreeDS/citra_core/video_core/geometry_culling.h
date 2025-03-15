// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/vector_math.h"
#include "video_core/pica_types.h"

// Define our own matrix type since Common::Mat4x4 is not available
namespace Pica {
    struct Matrix4x4 {
        std::array<Common::Vec4<float>, 4> r;
    };
}

namespace Pica {

struct CullingBoundingBox {
    Common::Vec3<float> min;
    Common::Vec3<float> max;
};

struct CullingFrustumPlanes {
    std::array<Common::Vec4<float>, 6> planes; // Left, Right, Bottom, Top, Near, Far
};

class GeometryCulling {
public:
    /**
     * Tests if a bounding box is visible within the view frustum.
     * @param bbox The bounding box to test
     * @param frustum The view frustum planes
     * @return True if the bounding box is visible, false if it's completely outside the frustum
     */
    static bool IsBoundingBoxVisible(const CullingBoundingBox& bbox, const CullingFrustumPlanes& frustum);

    /**
     * Tests if a point is inside the view frustum.
     * @param point The point to test
     * @param frustum The view frustum planes
     * @return True if the point is inside the frustum, false otherwise
     */
    static bool IsPointVisible(const Common::Vec3<float>& point, const CullingFrustumPlanes& frustum);

    /**
     * Calculates the frustum planes from the view and projection matrices.
     * @param view_projection The combined view-projection matrix
     * @return The calculated frustum planes
     */
    static CullingFrustumPlanes CalculateFrustumPlanes(const Matrix4x4& view_projection);

    /**
     * Performs backface culling test.
     * @param v0, v1, v2 The three vertices of the triangle
     * @param view_position The camera position
     * @return True if the triangle is facing the camera, false if it's facing away
     */
    static bool IsTriangleFacingCamera(
        const Common::Vec3<float>& v0,
        const Common::Vec3<float>& v1,
        const Common::Vec3<float>& v2,
        const Common::Vec3<float>& view_position);
};

} // namespace Pica
