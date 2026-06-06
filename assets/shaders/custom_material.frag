#version 450

layout(location = 0) in vec2 v_Uv;
layout(location = 0) out vec4 o_Target;

layout(set = 3, binding = 0) uniform vec4 CustomMaterial_camera_world_pos;
layout(set = 3, binding = 1) uniform uvec4 CustomMaterial_config;
layout(set = 3, binding = 4) uniform vec4 CustomMaterial_plane_center_world;
layout(set = 3, binding = 5) uniform vec4 CustomMaterial_plane_normal_world;
layout(set = 3, binding = 6) uniform vec4 CustomMaterial_plane_right_world;
layout(set = 3, binding = 7) uniform vec4 CustomMaterial_plane_up_world;
layout(set = 3, binding = 9) uniform vec4 CustomMaterial_plane_size;
layout(set = 3, binding = 10) uniform uvec4 CustomMaterial_pixel_layout;
layout(set = 3, binding = 11) uniform float CustomMaterial_hogel_fov_degrees;

layout(std430, set = 3, binding = 2) readonly buffer CustomMaterial_image_meta {
    uvec4 image_meta[];
};

layout(set = 3, binding = 3, rgba8ui) readonly uniform uimage2DArray CustomMaterial_image_pixels;

layout(std430, set = 3, binding = 8) readonly buffer CustomMaterial_hogel_lookup {
    uint hogel_lookup[];
};

const uint PARALLAX_HALF = 0u;
const uint PARALLAX_FULL = 1u;
const uint INVALID_IMAGE_INDEX = 0xffffffffu;

float srgb_to_linear_channel(float c) {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

vec3 srgb_to_linear(vec3 c) {
    return vec3(
        srgb_to_linear_channel(c.r),
        srgb_to_linear_channel(c.g),
        srgb_to_linear_channel(c.b)
    );
}

uint read_rgb_byte(
    uint base_byte_in_layer,
    uint base_layer,
    uint local_byte_index,
    uint tex_words_per_row,
    uint tex_rows_per_layer,
    uint bytes_per_layer,
    uint layer_count
) {
    if (bytes_per_layer < 4u || tex_words_per_row == 0u || tex_rows_per_layer == 0u || layer_count == 0u) {
        return 0u;
    }

    uint words_per_layer = tex_words_per_row * tex_rows_per_layer;
    uint byte_in_layer = base_byte_in_layer + local_byte_index;
    uint layer = base_layer + (byte_in_layer / bytes_per_layer);
    if (layer >= layer_count) {
        return 0u;
    }

    uint local_byte = byte_in_layer % bytes_per_layer;
    uint word_in_layer = local_byte >> 2u;
    if (word_in_layer >= words_per_layer) {
        return 0u;
    }

    uint x = word_in_layer % tex_words_per_row;
    uint y = word_in_layer / tex_words_per_row;
    uvec4 packed = imageLoad(CustomMaterial_image_pixels, ivec3(int(x), int(y), int(layer)));

    uint lane = local_byte & 3u;
    if (lane == 0u) {
        return packed.r;
    } else if (lane == 1u) {
        return packed.g;
    } else if (lane == 2u) {
        return packed.b;
    }
    return packed.a;
}

vec4 unpack_rgb8(
    uint base_byte_in_layer,
    uint base_layer,
    uint local_byte_offset,
    uint image_byte_len,
    uint tex_words_per_row,
    uint tex_rows_per_layer,
    uint bytes_per_layer,
    uint layer_count
) {
    if (image_byte_len < 3u || local_byte_offset > image_byte_len - 3u) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    float r = float(read_rgb_byte(
        base_byte_in_layer,
        base_layer,
        local_byte_offset + 0u,
        tex_words_per_row,
        tex_rows_per_layer,
        bytes_per_layer,
        layer_count
    ));
    float g = float(read_rgb_byte(
        base_byte_in_layer,
        base_layer,
        local_byte_offset + 1u,
        tex_words_per_row,
        tex_rows_per_layer,
        bytes_per_layer,
        layer_count
    ));
    float b = float(read_rgb_byte(
        base_byte_in_layer,
        base_layer,
        local_byte_offset + 2u,
        tex_words_per_row,
        tex_rows_per_layer,
        bytes_per_layer,
        layer_count
    ));
    vec3 srgb = vec3(r, g, b) / 255.0;
    return vec4(srgb_to_linear(srgb), 1.0);
}

void main() {
    uint mode = CustomMaterial_config.x;
    uint grid_width = CustomMaterial_config.y;
    uint grid_height = CustomMaterial_config.z;
    bool flip_x = (CustomMaterial_config.w & 1u) != 0u;
    bool flip_y = (CustomMaterial_config.w >> 1) != 0u;
    uint tex_words_per_row = max(CustomMaterial_pixel_layout.x, 1u);
    uint tex_rows_per_layer = max(CustomMaterial_pixel_layout.y, 1u);
    uint bytes_per_layer = max(CustomMaterial_pixel_layout.z, 4u);
    uint layer_count = max(CustomMaterial_pixel_layout.w, 1u);

    if (grid_width == 0u || grid_height == 0u) {
        o_Target = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float u = clamp(v_Uv.x, 0.0, 1.0);
    float v = clamp(v_Uv.y, 0.0, 1.0);

    uint cell_x = min(uint(floor(u * float(grid_width))), grid_width - 1u);
    uint geom_cell_y = 0u;
    uint lookup_cell_y = 0u;
    if (mode == PARALLAX_FULL) {
        geom_cell_y = min(uint(floor(v * float(grid_height))), grid_height - 1u);
        lookup_cell_y = flip_y
            ? (grid_height - 1u - geom_cell_y)
            : geom_cell_y;
    } else if (mode != PARALLAX_HALF) {
        o_Target = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uint lookup_index = lookup_cell_y * grid_width + cell_x;
    uint image_index = hogel_lookup[lookup_index];
    if (image_index == INVALID_IMAGE_INDEX) {
        o_Target = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uvec4 meta = image_meta[image_index];
    uint base_byte_in_layer = meta.x;
    uint width = meta.y;
    uint height = meta.z;
    uint base_layer = meta.w;
    if (width == 0u || height == 0u) {
        o_Target = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 plane_normal = normalize(CustomMaterial_plane_normal_world.xyz);
    vec3 plane_right = normalize(CustomMaterial_plane_right_world.xyz);
    vec3 plane_up = normalize(CustomMaterial_plane_up_world.xyz);
    float plane_width = max(CustomMaterial_plane_size.x, 1e-6);
    float plane_height = max(CustomMaterial_plane_size.y, 1e-6);
    vec3 plane_center = CustomMaterial_plane_center_world.xyz;
    vec3 sample_origin = plane_center;

    float cell_u_center = (float(cell_x) + 0.5) / float(grid_width);
    float sample_v = (mode == PARALLAX_FULL)
        ? ((float(geom_cell_y) + 0.5) / float(grid_height))
        : v;

    float local_x = (cell_u_center - 0.5) * plane_width;
    float local_y = (sample_v - 0.5) * plane_height;
    sample_origin = plane_center + plane_right * local_x + plane_up * local_y;

    vec3 view_dir = normalize(CustomMaterial_camera_world_pos.xyz - sample_origin);

    float forward = dot(view_dir, plane_normal);
    if (forward <= 0.0) {
        o_Target = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float hogel_fov_degrees = clamp(CustomMaterial_hogel_fov_degrees, 1e-6, 179.999);
    float r_m_h = tan(radians(hogel_fov_degrees) * 0.5);
    float hogel_aspect = max(float(width) / float(height), 1e-6);
    float r_m_v = r_m_h / hogel_aspect;
    float slope_h = -dot(view_dir, plane_right) / max(forward, 1e-6);
    float slope_v = -dot(view_dir, plane_up) / max(forward, 1e-6);
    float horizontal_t = (slope_h + r_m_h) / (2.0 * r_m_h);
    float vertical_t = (slope_v + r_m_v) / (2.0 * r_m_v);

    float horizontal_index = floor(horizontal_t * float(width));
    uint column = uint(clamp(horizontal_index, 0.0, float(width - 1u)));

    uint row = 0u;
    if (mode == PARALLAX_HALF) {
        row = uint(v * float(height - 1u));
    } else {
        float vertical_index = floor(vertical_t * float(height));
        row = uint(clamp(vertical_index, 0.0, float(height - 1u)));
    }

    if (flip_x) column = (width - 1u) - column;
    if (flip_y) row = (height - 1u) - row;
    
    uint pixel_index = row * width + column;
    uint local_byte_offset = pixel_index * 3u;
    uint image_byte_len = width * height * 3u;
    o_Target = unpack_rgb8(
        base_byte_in_layer,
        base_layer,
        local_byte_offset,
        image_byte_len,
        tex_words_per_row,
        tex_rows_per_layer,
        bytes_per_layer,
        layer_count
    );
}
