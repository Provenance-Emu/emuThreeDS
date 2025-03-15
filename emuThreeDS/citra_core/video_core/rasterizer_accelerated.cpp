// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <limits>
#include <cmath>
#include "common/alignment.h"
#include "core/memory.h"
#include "video_core/pica_state.h"
#include "video_core/rasterizer_accelerated.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace VideoCore {

static Common::Vec4f ColorRGBA8(const u32 color) {
    const auto rgba =
        Common::Vec4u{color >> 0 & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF, color >> 24 & 0xFF};
    return rgba / 255.0f;
}

static Common::Vec3f LightColor(const Pica::LightingRegs::LightColor& color) {
    return Common::Vec3u{color.r, color.g, color.b} / 255.0f;
}

RasterizerAccelerated::HardwareVertex::HardwareVertex(const Pica::Shader::OutputVertex& v,
                                                      bool flip_quaternion) {
    position[0] = v.pos.x.ToFloat32();
    position[1] = v.pos.y.ToFloat32();
    position[2] = v.pos.z.ToFloat32();
    position[3] = v.pos.w.ToFloat32();
    color[0] = v.color.x.ToFloat32();
    color[1] = v.color.y.ToFloat32();
    color[2] = v.color.z.ToFloat32();
    color[3] = v.color.w.ToFloat32();
    tex_coord0[0] = v.tc0.x.ToFloat32();
    tex_coord0[1] = v.tc0.y.ToFloat32();
    tex_coord1[0] = v.tc1.x.ToFloat32();
    tex_coord1[1] = v.tc1.y.ToFloat32();
    tex_coord2[0] = v.tc2.x.ToFloat32();
    tex_coord2[1] = v.tc2.y.ToFloat32();
    tex_coord0_w = v.tc0_w.ToFloat32();
    normquat[0] = v.quat.x.ToFloat32();
    normquat[1] = v.quat.y.ToFloat32();
    normquat[2] = v.quat.z.ToFloat32();
    normquat[3] = v.quat.w.ToFloat32();
    view[0] = v.view.x.ToFloat32();
    view[1] = v.view.y.ToFloat32();
    view[2] = v.view.z.ToFloat32();

    if (flip_quaternion) {
        normquat = -normquat;
    }
}

RasterizerAccelerated::RasterizerAccelerated(Memory::MemorySystem& memory_)
    : memory{memory_}, regs{Pica::g_state.regs} {
    uniform_block_data.lighting_lut_dirty.fill(true);
    
    // Initialize frustum planes for culling
    // In a real implementation, these would be calculated from the view-projection matrix
    // For now, we'll use a default frustum that's updated when needed
    frustum_dirty = true;
    camera_dirty = true;
    
    // Initialize culling statistics
    culling_stats.Reset();
}

/**
 * This is a helper function to resolve an issue when interpolating opposite quaternions. See below
 * for a detailed description of this issue (yuriks):
 *
 * For any rotation, there are two quaternions Q, and -Q, that represent the same rotation. If you
 * interpolate two quaternions that are opposite, instead of going from one rotation to another
 * using the shortest path, you'll go around the longest path. You can test if two quaternions are
 * opposite by checking if Dot(Q1, Q2) < 0. In that case, you can flip either of them, therefore
 * making Dot(Q1, -Q2) positive.
 *
 * This solution corrects this issue per-vertex before passing the quaternions to OpenGL. This is
 * correct for most cases but can still rotate around the long way sometimes. An implementation
 * which did `lerp(lerp(Q1, Q2), Q3)` (with proper weighting), applying the dot product check
 * between each step would work for those cases at the cost of being more complex to implement.
 *
 * Fortunately however, the 3DS hardware happens to also use this exact same logic to work around
 * these issues, making this basic implementation actually more accurate to the hardware.
 */
static bool AreQuaternionsOpposite(Common::Vec4<Pica::float24> qa, Common::Vec4<Pica::float24> qb) {
    Common::Vec4f a{qa.x.ToFloat32(), qa.y.ToFloat32(), qa.z.ToFloat32(), qa.w.ToFloat32()};
    Common::Vec4f b{qb.x.ToFloat32(), qb.y.ToFloat32(), qb.z.ToFloat32(), qb.w.ToFloat32()};

    return (Common::Dot(a, b) < 0.f);
}

void RasterizerAccelerated::AddTriangle(const Pica::Shader::OutputVertex& v0,
                                         const Pica::Shader::OutputVertex& v1,
                                         const Pica::Shader::OutputVertex& v2) {
    // Update culling statistics
    culling_state.stats.triangles_drawn++;
    
    // Skip culling if it's disabled in the rasterizer configuration
    if (culling_state.cull_mode != Pica::RasterizerRegs::CullMode::KeepAll) {
#if defined(__ARM_NEON) || defined(__aarch64__)
        // NEON-optimized vertex conversion
        // Create NEON registers for vertex positions
        float32x4_t v0_pos = vdupq_n_f32(0.0f);
        float32x4_t v1_pos = vdupq_n_f32(0.0f);
        float32x4_t v2_pos = vdupq_n_f32(0.0f);
        
        // Load vertex positions more efficiently
        v0_pos = vsetq_lane_f32(v0.pos.x.ToFloat32(), v0_pos, 0);
        v0_pos = vsetq_lane_f32(v0.pos.y.ToFloat32(), v0_pos, 1);
        v0_pos = vsetq_lane_f32(v0.pos.z.ToFloat32(), v0_pos, 2);
        v0_pos = vsetq_lane_f32(v0.pos.w.ToFloat32(), v0_pos, 3);
        
        v1_pos = vsetq_lane_f32(v1.pos.x.ToFloat32(), v1_pos, 0);
        v1_pos = vsetq_lane_f32(v1.pos.y.ToFloat32(), v1_pos, 1);
        v1_pos = vsetq_lane_f32(v1.pos.z.ToFloat32(), v1_pos, 2);
        v1_pos = vsetq_lane_f32(v1.pos.w.ToFloat32(), v1_pos, 3);
        
        v2_pos = vsetq_lane_f32(v2.pos.x.ToFloat32(), v2_pos, 0);
        v2_pos = vsetq_lane_f32(v2.pos.y.ToFloat32(), v2_pos, 1);
        v2_pos = vsetq_lane_f32(v2.pos.z.ToFloat32(), v2_pos, 2);
        v2_pos = vsetq_lane_f32(v2.pos.w.ToFloat32(), v2_pos, 3);
        
        // Create Vec3 objects for compatibility with culling functions
        Common::Vec3<float> pos0 = {vgetq_lane_f32(v0_pos, 0), vgetq_lane_f32(v0_pos, 1), vgetq_lane_f32(v0_pos, 2)};
        Common::Vec3<float> pos1 = {vgetq_lane_f32(v1_pos, 0), vgetq_lane_f32(v1_pos, 1), vgetq_lane_f32(v1_pos, 2)};
        Common::Vec3<float> pos2 = {vgetq_lane_f32(v2_pos, 0), vgetq_lane_f32(v2_pos, 1), vgetq_lane_f32(v2_pos, 2)};
#else
        // Standard vertex conversion
        Common::Vec3<float> pos0 = {v0.pos.x.ToFloat32(), v0.pos.y.ToFloat32(), v0.pos.z.ToFloat32()};
        Common::Vec3<float> pos1 = {v1.pos.x.ToFloat32(), v1.pos.y.ToFloat32(), v1.pos.z.ToFloat32()};
        Common::Vec3<float> pos2 = {v2.pos.x.ToFloat32(), v2.pos.y.ToFloat32(), v2.pos.z.ToFloat32()};
#endif
        
        // Recalculate frustum planes if needed
        if (culling_state.dirty) {
            // Force a sync of the entire state to update the frustum planes
            SyncEntireState();
        }
        
        // Perform fast frustum culling using bounding box
#if defined(__ARM_NEON) || defined(__aarch64__)
        // Create NEON vectors for each vertex position
        float32x4_t pos0_vec = vdupq_n_f32(0.0f);
        float32x4_t pos1_vec = vdupq_n_f32(0.0f);
        float32x4_t pos2_vec = vdupq_n_f32(0.0f);
        
        // Load XYZ components
        pos0_vec = vsetq_lane_f32(pos0.x, pos0_vec, 0);
        pos0_vec = vsetq_lane_f32(pos0.y, pos0_vec, 1);
        pos0_vec = vsetq_lane_f32(pos0.z, pos0_vec, 2);
        
        pos1_vec = vsetq_lane_f32(pos1.x, pos1_vec, 0);
        pos1_vec = vsetq_lane_f32(pos1.y, pos1_vec, 1);
        pos1_vec = vsetq_lane_f32(pos1.z, pos1_vec, 2);
        
        pos2_vec = vsetq_lane_f32(pos2.x, pos2_vec, 0);
        pos2_vec = vsetq_lane_f32(pos2.y, pos2_vec, 1);
        pos2_vec = vsetq_lane_f32(pos2.z, pos2_vec, 2);
        
        // Compute min/max for bounding box in a single operation
        float32x4_t min_vec = vminq_f32(vminq_f32(pos0_vec, pos1_vec), pos2_vec);
        float32x4_t max_vec = vmaxq_f32(vmaxq_f32(pos0_vec, pos1_vec), pos2_vec);
        
        // Create bounding box for frustum culling
        Pica::CullingBoundingBox bbox;
        bbox.min = {vgetq_lane_f32(min_vec, 0), vgetq_lane_f32(min_vec, 1), vgetq_lane_f32(min_vec, 2)};
        bbox.max = {vgetq_lane_f32(max_vec, 0), vgetq_lane_f32(max_vec, 1), vgetq_lane_f32(max_vec, 2)};
#else
        // Create bounding box for frustum culling (standard implementation)
        Pica::CullingBoundingBox bbox;
        bbox.min = {
            std::min({pos0.x, pos1.x, pos2.x}),
            std::min({pos0.y, pos1.y, pos2.y}),
            std::min({pos0.z, pos1.z, pos2.z})
        };
        bbox.max = {
            std::max({pos0.x, pos1.x, pos2.x}),
            std::max({pos0.y, pos1.y, pos2.y}),
            std::max({pos0.z, pos1.z, pos2.z})
        };
#endif
        
        // Check if bounding box is visible
        if (!Pica::GeometryCulling::IsBoundingBoxVisible(bbox, culling_state.frustum)) {
            culling_state.stats.triangles_culled++;
            culling_stats.triangles_rejected++;
            return; // Skip this triangle
        }
        
        // Perform backface culling based on the culling mode
        if (regs.rasterizer.cull_mode == Pica::RasterizerRegs::CullMode::KeepClockWise) {
            // Check if triangle is counter-clockwise (should be culled)
            if (!Pica::GeometryCulling::IsTriangleFacingCamera(pos0, pos1, pos2, camera_position)) {
                culling_state.stats.triangles_backface++;
                culling_stats.triangles_rejected++;
                return; // Skip this triangle
            }
        } else if (regs.rasterizer.cull_mode == Pica::RasterizerRegs::CullMode::KeepCounterClockWise) {
            // Check if triangle is clockwise (should be culled)
            if (Pica::GeometryCulling::IsTriangleFacingCamera(pos0, pos1, pos2, camera_position)) {
                culling_state.stats.triangles_backface++;
                culling_stats.triangles_rejected++;
                return; // Skip this triangle
            }
        }
    }
    
    // If we reach here, the triangle passed all culling tests
    vertex_batch.emplace_back(v0, false);
    vertex_batch.emplace_back(v1, AreQuaternionsOpposite(v0.quat, v1.quat));
    vertex_batch.emplace_back(v2, AreQuaternionsOpposite(v0.quat, v2.quat));
}

RasterizerAccelerated::VertexArrayInfo RasterizerAccelerated::AnalyzeVertexArray(
    bool is_indexed, u32 stride_alignment) {
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;

    u32 vertex_min;
    u32 vertex_max;
    if (is_indexed) {
        const auto& index_info = regs.pipeline.index_array;
        const PAddr address = vertex_attributes.GetPhysicalBaseAddress() + index_info.offset;
        const u8* index_address_8 = memory.GetPhysicalPointer(address);
        const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
        const bool index_u16 = index_info.format != 0;

        vertex_min = 0xFFFF;
        vertex_max = 0;
        const u32 size = regs.pipeline.num_vertices * (index_u16 ? 2 : 1);
        FlushRegion(address, size);
        for (u32 index = 0; index < regs.pipeline.num_vertices; ++index) {
            const u32 vertex = index_u16 ? index_address_16[index] : index_address_8[index];
            vertex_min = std::min(vertex_min, vertex);
            vertex_max = std::max(vertex_max, vertex);
        }
    } else {
        vertex_min = regs.pipeline.vertex_offset;
        vertex_max = regs.pipeline.vertex_offset + regs.pipeline.num_vertices - 1;
    }

    const u32 vertex_num = vertex_max - vertex_min + 1;
    u32 vs_input_size = 0;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count != 0) {
            const u32 aligned_stride =
                Common::AlignUp(static_cast<u32>(loader.byte_count), stride_alignment);
            vs_input_size += Common::AlignUp(aligned_stride * vertex_num, 4);
        }
    }

    return {vertex_min, vertex_max, vs_input_size};
}

void RasterizerAccelerated::SyncEntireState() {
    // Sync renderer-specific fixed-function state
    SyncFixedState();

    // Sync uniforms
    SyncClipCoef();
    SyncDepthScale();
    SyncDepthOffset();
    SyncAlphaTest();
    SyncCombinerColor();
    auto& tev_stages = regs.texturing.GetTevStages();
    for (std::size_t index = 0; index < tev_stages.size(); ++index) {
        SyncTevConstColor(index, tev_stages[index]);
    }

#if defined(__ARM_NEON) || defined(__aarch64__)
    // Since we can't directly access the matrices, we'll create identity matrices
    // and update the frustum planes directly
    Pica::Matrix4x4 view_matrix = {};
    Pica::Matrix4x4 projection_matrix = {};
    Pica::Matrix4x4 view_projection = {};
    
    // Set identity matrices
    view_matrix.r[0].x = 1.0f;
    view_matrix.r[1].y = 1.0f;
    view_matrix.r[2].z = 1.0f;
    view_matrix.r[3].w = 1.0f;
    
    projection_matrix.r[0].x = 1.0f;
    projection_matrix.r[1].y = 1.0f;
    projection_matrix.r[2].z = 1.0f;
    projection_matrix.r[3].w = 1.0f;
    
    // Optimized matrix multiplication using NEON
    for (int i = 0; i < 4; i++) {
        // Load the entire row of view matrix
        float32x4_t view_row = vld1q_f32(&view_matrix.r[i].x);
        
        // For each column in the result
        for (int j = 0; j < 4; j++) {
            // Initialize accumulator
            float32x4_t acc = vdupq_n_f32(0.0f);
            
            // For each element in the dot product
            for (int k = 0; k < 4; k++) {
                // Extract the scalar from view_row based on k
                float view_element;
                if (k == 0) view_element = vgetq_lane_f32(view_row, 0);
                else if (k == 1) view_element = vgetq_lane_f32(view_row, 1);
                else if (k == 2) view_element = vgetq_lane_f32(view_row, 2);
                else view_element = vgetq_lane_f32(view_row, 3);
                
                // Load the column element from projection matrix
                float proj_element = projection_matrix.r[k][j];
                
                // Multiply and accumulate
                acc = vmlaq_n_f32(acc, vdupq_n_f32(view_element), proj_element);
            }
            
            // Store the result - extract first element of accumulator
            view_projection.r[i][j] = vgetq_lane_f32(acc, 0);
        }
    }
    
    // Calculate frustum planes for culling
    culling_state.frustum = Pica::GeometryCulling::CalculateFrustumPlanes(view_projection);
    culling_state.dirty = false;
    
    // Update camera position from view matrix inverse translation
    // In a typical view matrix, the camera position is the negative of the translation part
    // transformed by the rotation part of the matrix
    float32x4_t cam_pos = vdupq_n_f32(0.0f);
    
    // Extract translation components (last row of view matrix)
    float tx = view_matrix.r[3][0];
    float ty = view_matrix.r[3][1];
    float tz = view_matrix.r[3][2];
    
    // Negate to get camera position in world space
    cam_pos = vsetq_lane_f32(-tx, cam_pos, 0);
    cam_pos = vsetq_lane_f32(-ty, cam_pos, 1);
    cam_pos = vsetq_lane_f32(-tz, cam_pos, 2);
    
    // Store camera position for backface culling
    camera_position = {vgetq_lane_f32(cam_pos, 0), vgetq_lane_f32(cam_pos, 1), vgetq_lane_f32(cam_pos, 2)};
    camera_dirty = false;
#else
    // Extract view matrix from the PICA registers
    Common::Mat4x4<float> view_matrix;
    for (int i = 0; i < 4; i++) {
        view_matrix.r[i] = regs.clip_coef.GetViewMatrix()[i];
    }
    
    // Extract projection matrix from the PICA registers
    Common::Mat4x4<float> proj_matrix;
    for (int i = 0; i < 4; i++) {
        proj_matrix.r[i] = regs.clip_coef.GetProjectionMatrix()[i];
    }
    
    // Standard matrix multiplication
    Common::Mat4x4<float> view_projection;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            view_projection.r[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                view_projection.r[i][j] += view_matrix.r[i][k] * proj_matrix.r[k][j];
            }
        }
    }
    
    // Calculate frustum planes from view-projection matrix
    current_frustum = Pica::GeometryCulling::CalculateFrustumPlanes(view_projection);
    frustum_dirty = false;
    
    // Update camera position from view matrix inverse translation
    camera_position = {-view_matrix.r[3][0], -view_matrix.r[3][1], -view_matrix.r[3][2]};
    camera_dirty = false;
#endif

    SyncGlobalAmbient();
    for (unsigned light_index = 0; light_index < 8; light_index++) {
        SyncLightSpecular0(light_index);
        SyncLightSpecular1(light_index);
        SyncLightDiffuse(light_index);
        SyncLightAmbient(light_index);
        SyncLightPosition(light_index);
        SyncLightDistanceAttenuationBias(light_index);
        SyncLightDistanceAttenuationScale(light_index);
    }

    SyncFogColor();
    SyncProcTexNoise();
    SyncProcTexBias();
    SyncShadowBias();
    SyncShadowTextureBias();

    for (unsigned tex_index = 0; tex_index < 3; tex_index++) {
        SyncTextureLodBias(tex_index);
    }
    
    // Reset culling statistics
    culling_stats.Reset();
    culling_state.stats.triangles_drawn = 0;
    culling_state.stats.triangles_culled = 0;
    culling_state.stats.triangles_backface = 0;
}

void RasterizerAccelerated::NotifyPicaRegisterChanged(u32 id) {
    switch (id) {
    // Depth modifiers
    case PICA_REG_INDEX(rasterizer.viewport_depth_range):
        SyncDepthScale();
        break;
    case PICA_REG_INDEX(rasterizer.viewport_depth_near_plane):
        SyncDepthOffset();
        break;

    // Depth buffering
    case PICA_REG_INDEX(rasterizer.depthmap_enable):
        shader_dirty = true;
        break;
        
    // Culling mode
    case PICA_REG_INDEX(rasterizer.cull_mode):
        // Mark frustum as dirty to recalculate culling state
        frustum_dirty = true;
        break;

    // Shadow texture
    case PICA_REG_INDEX(texturing.shadow):
        SyncShadowTextureBias();
        break;

    // Fog state
    case PICA_REG_INDEX(texturing.fog_color):
        SyncFogColor();
        break;
    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]):
        uniform_block_data.fog_lut_dirty = true;
        break;

    // ProcTex state
    case PICA_REG_INDEX(texturing.proctex):
    case PICA_REG_INDEX(texturing.proctex_lut):
    case PICA_REG_INDEX(texturing.proctex_lut_offset):
        SyncProcTexBias();
        shader_dirty = true;
        break;

    case PICA_REG_INDEX(texturing.proctex_noise_u):
    case PICA_REG_INDEX(texturing.proctex_noise_v):
    case PICA_REG_INDEX(texturing.proctex_noise_frequency):
        SyncProcTexNoise();
        break;

    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]):
        using Pica::TexturingRegs;
        switch (regs.texturing.proctex_lut_config.ref_table.Value()) {
        case TexturingRegs::ProcTexLutTable::Noise:
            uniform_block_data.proctex_noise_lut_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            uniform_block_data.proctex_color_map_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            uniform_block_data.proctex_alpha_map_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            uniform_block_data.proctex_lut_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            uniform_block_data.proctex_diff_lut_dirty = true;
            break;
        }
        break;

    // Alpha test
    case PICA_REG_INDEX(framebuffer.output_merger.alpha_test):
        SyncAlphaTest();
        shader_dirty = true;
        break;

    case PICA_REG_INDEX(framebuffer.shadow):
        SyncShadowBias();
        break;

    // Scissor test
    case PICA_REG_INDEX(rasterizer.scissor_test.mode):
        shader_dirty = true;
        break;

    case PICA_REG_INDEX(texturing.main_config):
        shader_dirty = true;
        break;

    // Texture 0 type
    case PICA_REG_INDEX(texturing.texture0.type):
        shader_dirty = true;
        break;

    // TEV stages
    // (This also syncs fog_mode and fog_flip which are part of tev_combiner_buffer_input)
    case PICA_REG_INDEX(texturing.tev_stage0.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage0.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage0.color_op):
    case PICA_REG_INDEX(texturing.tev_stage0.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage1.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage1.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage1.color_op):
    case PICA_REG_INDEX(texturing.tev_stage1.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage2.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage2.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage2.color_op):
    case PICA_REG_INDEX(texturing.tev_stage2.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage3.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage3.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage3.color_op):
    case PICA_REG_INDEX(texturing.tev_stage3.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage4.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage4.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage4.color_op):
    case PICA_REG_INDEX(texturing.tev_stage4.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage5.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage5.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage5.color_op):
    case PICA_REG_INDEX(texturing.tev_stage5.color_scale):
    case PICA_REG_INDEX(texturing.tev_combiner_buffer_input):
        shader_dirty = true;
        break;
    case PICA_REG_INDEX(texturing.tev_stage0.const_r):
        SyncTevConstColor(0, regs.texturing.tev_stage0);
        break;
    case PICA_REG_INDEX(texturing.tev_stage1.const_r):
        SyncTevConstColor(1, regs.texturing.tev_stage1);
        break;
    case PICA_REG_INDEX(texturing.tev_stage2.const_r):
        SyncTevConstColor(2, regs.texturing.tev_stage2);
        break;
    case PICA_REG_INDEX(texturing.tev_stage3.const_r):
        SyncTevConstColor(3, regs.texturing.tev_stage3);
        break;
    case PICA_REG_INDEX(texturing.tev_stage4.const_r):
        SyncTevConstColor(4, regs.texturing.tev_stage4);
        break;
    case PICA_REG_INDEX(texturing.tev_stage5.const_r):
        SyncTevConstColor(5, regs.texturing.tev_stage5);
        break;

    // TEV combiner buffer color
    case PICA_REG_INDEX(texturing.tev_combiner_buffer_color):
        SyncCombinerColor();
        break;

    // Fragment lighting switches
    case PICA_REG_INDEX(lighting.disable):
    case PICA_REG_INDEX(lighting.max_light_index):
    case PICA_REG_INDEX(lighting.config0):
    case PICA_REG_INDEX(lighting.config1):
    case PICA_REG_INDEX(lighting.abs_lut_input):
    case PICA_REG_INDEX(lighting.lut_input):
    case PICA_REG_INDEX(lighting.lut_scale):
    case PICA_REG_INDEX(lighting.light_enable):
        break;

    // Fragment lighting specular 0 color
    case PICA_REG_INDEX(lighting.light[0].specular_0):
        SyncLightSpecular0(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].specular_0):
        SyncLightSpecular0(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].specular_0):
        SyncLightSpecular0(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].specular_0):
        SyncLightSpecular0(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].specular_0):
        SyncLightSpecular0(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].specular_0):
        SyncLightSpecular0(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].specular_0):
        SyncLightSpecular0(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].specular_0):
        SyncLightSpecular0(7);
        break;

    // Fragment lighting specular 1 color
    case PICA_REG_INDEX(lighting.light[0].specular_1):
        SyncLightSpecular1(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].specular_1):
        SyncLightSpecular1(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].specular_1):
        SyncLightSpecular1(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].specular_1):
        SyncLightSpecular1(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].specular_1):
        SyncLightSpecular1(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].specular_1):
        SyncLightSpecular1(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].specular_1):
        SyncLightSpecular1(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].specular_1):
        SyncLightSpecular1(7);
        break;

    // Fragment lighting diffuse color
    case PICA_REG_INDEX(lighting.light[0].diffuse):
        SyncLightDiffuse(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].diffuse):
        SyncLightDiffuse(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].diffuse):
        SyncLightDiffuse(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].diffuse):
        SyncLightDiffuse(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].diffuse):
        SyncLightDiffuse(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].diffuse):
        SyncLightDiffuse(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].diffuse):
        SyncLightDiffuse(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].diffuse):
        SyncLightDiffuse(7);
        break;

    // Fragment lighting ambient color
    case PICA_REG_INDEX(lighting.light[0].ambient):
        SyncLightAmbient(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].ambient):
        SyncLightAmbient(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].ambient):
        SyncLightAmbient(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].ambient):
        SyncLightAmbient(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].ambient):
        SyncLightAmbient(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].ambient):
        SyncLightAmbient(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].ambient):
        SyncLightAmbient(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].ambient):
        SyncLightAmbient(7);
        break;

    // Fragment lighting position
    case PICA_REG_INDEX(lighting.light[0].x):
    case PICA_REG_INDEX(lighting.light[0].z):
        SyncLightPosition(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].x):
    case PICA_REG_INDEX(lighting.light[1].z):
        SyncLightPosition(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].x):
    case PICA_REG_INDEX(lighting.light[2].z):
        SyncLightPosition(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].x):
    case PICA_REG_INDEX(lighting.light[3].z):
        SyncLightPosition(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].x):
    case PICA_REG_INDEX(lighting.light[4].z):
        SyncLightPosition(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].x):
    case PICA_REG_INDEX(lighting.light[5].z):
        SyncLightPosition(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].x):
    case PICA_REG_INDEX(lighting.light[6].z):
        SyncLightPosition(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].x):
    case PICA_REG_INDEX(lighting.light[7].z):
        SyncLightPosition(7);
        break;

    // Fragment spot lighting direction
    case PICA_REG_INDEX(lighting.light[0].spot_x):
    case PICA_REG_INDEX(lighting.light[0].spot_z):
        SyncLightSpotDirection(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].spot_x):
    case PICA_REG_INDEX(lighting.light[1].spot_z):
        SyncLightSpotDirection(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].spot_x):
    case PICA_REG_INDEX(lighting.light[2].spot_z):
        SyncLightSpotDirection(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].spot_x):
    case PICA_REG_INDEX(lighting.light[3].spot_z):
        SyncLightSpotDirection(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].spot_x):
    case PICA_REG_INDEX(lighting.light[4].spot_z):
        SyncLightSpotDirection(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].spot_x):
    case PICA_REG_INDEX(lighting.light[5].spot_z):
        SyncLightSpotDirection(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].spot_x):
    case PICA_REG_INDEX(lighting.light[6].spot_z):
        SyncLightSpotDirection(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].spot_x):
    case PICA_REG_INDEX(lighting.light[7].spot_z):
        SyncLightSpotDirection(7);
        break;

    // Fragment lighting light source config
    case PICA_REG_INDEX(lighting.light[0].config):
    case PICA_REG_INDEX(lighting.light[1].config):
    case PICA_REG_INDEX(lighting.light[2].config):
    case PICA_REG_INDEX(lighting.light[3].config):
    case PICA_REG_INDEX(lighting.light[4].config):
    case PICA_REG_INDEX(lighting.light[5].config):
    case PICA_REG_INDEX(lighting.light[6].config):
    case PICA_REG_INDEX(lighting.light[7].config):
        shader_dirty = true;
        break;

    // Fragment lighting distance attenuation bias
    case PICA_REG_INDEX(lighting.light[0].dist_atten_bias):
        SyncLightDistanceAttenuationBias(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].dist_atten_bias):
        SyncLightDistanceAttenuationBias(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].dist_atten_bias):
        SyncLightDistanceAttenuationBias(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].dist_atten_bias):
        SyncLightDistanceAttenuationBias(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].dist_atten_bias):
        SyncLightDistanceAttenuationBias(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].dist_atten_bias):
        SyncLightDistanceAttenuationBias(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].dist_atten_bias):
        SyncLightDistanceAttenuationBias(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].dist_atten_bias):
        SyncLightDistanceAttenuationBias(7);
        break;

    // Fragment lighting distance attenuation scale
    case PICA_REG_INDEX(lighting.light[0].dist_atten_scale):
        SyncLightDistanceAttenuationScale(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].dist_atten_scale):
        SyncLightDistanceAttenuationScale(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].dist_atten_scale):
        SyncLightDistanceAttenuationScale(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].dist_atten_scale):
        SyncLightDistanceAttenuationScale(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].dist_atten_scale):
        SyncLightDistanceAttenuationScale(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].dist_atten_scale):
        SyncLightDistanceAttenuationScale(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].dist_atten_scale):
        SyncLightDistanceAttenuationScale(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].dist_atten_scale):
        SyncLightDistanceAttenuationScale(7);
        break;

    // Fragment lighting global ambient color (emission + ambient * ambient)
    case PICA_REG_INDEX(lighting.global_ambient):
        SyncGlobalAmbient();
        break;

    // Fragment lighting lookup tables
    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        const auto& lut_config = regs.lighting.lut_config;
        uniform_block_data.lighting_lut_dirty[lut_config.type] = true;
        uniform_block_data.lighting_lut_dirty_any = true;
        break;
    }

    // Texture LOD biases
    case PICA_REG_INDEX(texturing.texture0.lod.bias):
        SyncTextureLodBias(0);
        break;
    case PICA_REG_INDEX(texturing.texture1.lod.bias):
        SyncTextureLodBias(1);
        break;
    case PICA_REG_INDEX(texturing.texture2.lod.bias):
        SyncTextureLodBias(2);
        break;

    // Clipping plane
    case PICA_REG_INDEX(rasterizer.clip_coef[0]):
    case PICA_REG_INDEX(rasterizer.clip_coef[1]):
    case PICA_REG_INDEX(rasterizer.clip_coef[2]):
    case PICA_REG_INDEX(rasterizer.clip_coef[3]):
        SyncClipCoef();
        break;

    default:
        // Forward registers that map to fixed function API features to the video backend
        NotifyFixedFunctionPicaRegisterChanged(id);
    }
}

void RasterizerAccelerated::SyncDepthScale() {
    float depth_scale = Pica::float24::FromRaw(regs.rasterizer.viewport_depth_range).ToFloat32();

    if (depth_scale != uniform_block_data.data.depth_scale) {
        uniform_block_data.data.depth_scale = depth_scale;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncDepthOffset() {
    float depth_offset =
        Pica::float24::FromRaw(regs.rasterizer.viewport_depth_near_plane).ToFloat32();

    if (depth_offset != uniform_block_data.data.depth_offset) {
        uniform_block_data.data.depth_offset = depth_offset;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncFogColor() {
    const auto& fog_color_regs = regs.texturing.fog_color;
    const Common::Vec3f fog_color = {
        fog_color_regs.r.Value() / 255.0f,
        fog_color_regs.g.Value() / 255.0f,
        fog_color_regs.b.Value() / 255.0f,
    };

    if (fog_color != uniform_block_data.data.fog_color) {
        uniform_block_data.data.fog_color = fog_color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncProcTexNoise() {
    const Common::Vec2f proctex_noise_f = {
        Pica::float16::FromRaw(regs.texturing.proctex_noise_frequency.u).ToFloat32(),
        Pica::float16::FromRaw(regs.texturing.proctex_noise_frequency.v).ToFloat32(),
    };
    const Common::Vec2f proctex_noise_a = {
        regs.texturing.proctex_noise_u.amplitude / 4095.0f,
        regs.texturing.proctex_noise_v.amplitude / 4095.0f,
    };
    const Common::Vec2f proctex_noise_p = {
        Pica::float16::FromRaw(regs.texturing.proctex_noise_u.phase).ToFloat32(),
        Pica::float16::FromRaw(regs.texturing.proctex_noise_v.phase).ToFloat32(),
    };

    if (proctex_noise_f != uniform_block_data.data.proctex_noise_f ||
        proctex_noise_a != uniform_block_data.data.proctex_noise_a ||
        proctex_noise_p != uniform_block_data.data.proctex_noise_p) {
        uniform_block_data.data.proctex_noise_f = proctex_noise_f;
        uniform_block_data.data.proctex_noise_a = proctex_noise_a;
        uniform_block_data.data.proctex_noise_p = proctex_noise_p;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncProcTexBias() {
    const auto proctex_bias = Pica::float16::FromRaw(regs.texturing.proctex.bias_low |
                                                     (regs.texturing.proctex_lut.bias_high << 8))
                                  .ToFloat32();
    if (proctex_bias != uniform_block_data.data.proctex_bias) {
        uniform_block_data.data.proctex_bias = proctex_bias;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncAlphaTest() {
    if (regs.framebuffer.output_merger.alpha_test.ref !=
        static_cast<u32>(uniform_block_data.data.alphatest_ref)) {
        uniform_block_data.data.alphatest_ref = regs.framebuffer.output_merger.alpha_test.ref;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncCombinerColor() {
    auto combiner_color = ColorRGBA8(regs.texturing.tev_combiner_buffer_color.raw);
    if (combiner_color != uniform_block_data.data.tev_combiner_buffer_color) {
        uniform_block_data.data.tev_combiner_buffer_color = combiner_color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncTevConstColor(
    std::size_t stage_index, const Pica::TexturingRegs::TevStageConfig& tev_stage) {
    const auto const_color = ColorRGBA8(tev_stage.const_color);

    if (const_color == uniform_block_data.data.const_color[stage_index]) {
        return;
    }

    uniform_block_data.data.const_color[stage_index] = const_color;
    uniform_block_data.dirty = true;
}

void RasterizerAccelerated::SyncGlobalAmbient() {
    auto color = LightColor(regs.lighting.global_ambient);
    if (color != uniform_block_data.data.lighting_global_ambient) {
        uniform_block_data.data.lighting_global_ambient = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpecular0(int light_index) {
    auto color = LightColor(regs.lighting.light[light_index].specular_0);
    if (color != uniform_block_data.data.light_src[light_index].specular_0) {
        uniform_block_data.data.light_src[light_index].specular_0 = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpecular1(int light_index) {
    auto color = LightColor(regs.lighting.light[light_index].specular_1);
    if (color != uniform_block_data.data.light_src[light_index].specular_1) {
        uniform_block_data.data.light_src[light_index].specular_1 = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDiffuse(int light_index) {
    auto color = LightColor(regs.lighting.light[light_index].diffuse);
    if (color != uniform_block_data.data.light_src[light_index].diffuse) {
        uniform_block_data.data.light_src[light_index].diffuse = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightAmbient(int light_index) {
    auto color = LightColor(regs.lighting.light[light_index].ambient);
    if (color != uniform_block_data.data.light_src[light_index].ambient) {
        uniform_block_data.data.light_src[light_index].ambient = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightPosition(int light_index) {
    const Common::Vec3f position = {
        Pica::float16::FromRaw(regs.lighting.light[light_index].x).ToFloat32(),
        Pica::float16::FromRaw(regs.lighting.light[light_index].y).ToFloat32(),
        Pica::float16::FromRaw(regs.lighting.light[light_index].z).ToFloat32(),
    };

    if (position != uniform_block_data.data.light_src[light_index].position) {
        uniform_block_data.data.light_src[light_index].position = position;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpotDirection(int light_index) {
    const auto& light = regs.lighting.light[light_index];
    const auto spot_direction =
        Common::Vec3f{light.spot_x / 2047.0f, light.spot_y / 2047.0f, light.spot_z / 2047.0f};

    if (spot_direction != uniform_block_data.data.light_src[light_index].spot_direction) {
        uniform_block_data.data.light_src[light_index].spot_direction = spot_direction;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDistanceAttenuationBias(int light_index) {
    float dist_atten_bias =
        Pica::float20::FromRaw(regs.lighting.light[light_index].dist_atten_bias).ToFloat32();

    if (dist_atten_bias != uniform_block_data.data.light_src[light_index].dist_atten_bias) {
        uniform_block_data.data.light_src[light_index].dist_atten_bias = dist_atten_bias;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDistanceAttenuationScale(int light_index) {
    float dist_atten_scale =
        Pica::float20::FromRaw(regs.lighting.light[light_index].dist_atten_scale).ToFloat32();

    if (dist_atten_scale != uniform_block_data.data.light_src[light_index].dist_atten_scale) {
        uniform_block_data.data.light_src[light_index].dist_atten_scale = dist_atten_scale;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncShadowBias() {
    const auto& shadow = regs.framebuffer.shadow;
    float constant = Pica::float16::FromRaw(shadow.constant).ToFloat32();
    float linear = Pica::float16::FromRaw(shadow.linear).ToFloat32();

    if (constant != uniform_block_data.data.shadow_bias_constant ||
        linear != uniform_block_data.data.shadow_bias_linear) {
        uniform_block_data.data.shadow_bias_constant = constant;
        uniform_block_data.data.shadow_bias_linear = linear;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncShadowTextureBias() {
    int bias = regs.texturing.shadow.bias << 1;
    if (bias != uniform_block_data.data.shadow_texture_bias) {
        uniform_block_data.data.shadow_texture_bias = bias;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncTextureLodBias(int tex_index) {
    const auto pica_textures = regs.texturing.GetTextures();
    const float bias = pica_textures[tex_index].config.lod.bias / 256.0f;
    if (bias != uniform_block_data.data.tex_lod_bias[tex_index]) {
        uniform_block_data.data.tex_lod_bias[tex_index] = bias;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncClipCoef() {
    const auto raw_clip_coef = regs.rasterizer.GetClipCoef();
    const Common::Vec4f new_clip_coef = {raw_clip_coef.x.ToFloat32(), raw_clip_coef.y.ToFloat32(),
                                         raw_clip_coef.z.ToFloat32(), raw_clip_coef.w.ToFloat32()};
    if (new_clip_coef != uniform_block_data.data.clip_coef) {
        uniform_block_data.data.clip_coef = new_clip_coef;
        uniform_block_data.dirty = true;
    }
}

} // namespace VideoCore
