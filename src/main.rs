use std::{
    fs,
    io::BufReader,
    path::{Path, PathBuf},
};

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::{
            AsBindGroup, Extent3d, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDescriptor, TextureViewDimension,
        },
        storage::ShaderStorageBuffer,
    },
    shader::ShaderRef,
};
use clap::{Parser, ValueEnum};

const VERTEX_SHADER_ASSET_PATH: &str = "shaders/custom_material.vert";
const FRAGMENT_SHADER_ASSET_PATH: &str = "shaders/custom_material.frag";
const INVALID_IMAGE_INDEX: u32 = u32::MAX;
const PIXEL_TEXTURE_WORDS_PER_ROW: u32 = 4096;
const PIXEL_TEXTURE_ROWS_PER_LAYER: u32 = 4096;

#[derive(Debug, Clone, Copy)]
struct ImageMetaCpu {
    offset_bytes: u64,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ParallaxMode {
    Half,
    Full,
}

impl ParallaxMode {
    fn as_u32(self) -> u32 {
        match self {
            Self::Half => 0,
            Self::Full => 1,
        }
    }
}

#[derive(Resource, Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    image_directory: PathBuf,
    #[arg(long, value_enum)]
    mode: ParallaxMode,
    #[arg(long, value_parser = parse_positive_f32, default_value_t = 2.0)]
    plane_width: f32,
    #[arg(long, value_parser = parse_positive_f32, default_value_t = 3.0)]
    plane_height: f32,
}

fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid float: '{value}'"))?;
    if parsed <= 0.0 {
        return Err("must be > 0".to_string());
    }
    Ok(parsed)
}

fn main() {
    App::new()
        .insert_resource(CliArgs::parse())
        .add_plugins((DefaultPlugins, MaterialPlugin::<CustomMaterial>::default()))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (orbit_camera_controls, sync_camera_uniform))
        .run();
}

#[derive(Component)]
struct OrbitCamera {
    target: Vec3,
    distance: f32,
    min_distance: f32,
    max_distance: f32,
    yaw: f32,
    pitch: f32,
    rotate_sensitivity: f32,
    zoom_sensitivity: f32,
}

impl OrbitCamera {
    fn translation(&self) -> Vec3 {
        let x = self.distance * self.yaw.sin() * self.pitch.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.yaw.cos() * self.pitch.cos();
        self.target + Vec3::new(x, y, z)
    }
}

#[derive(Component)]
struct OrbitMainCamera;

#[derive(Debug)]
struct DecodedImages {
    mode: ParallaxMode,
    rgb_bytes: Vec<u8>,
    image_meta: Vec<ImageMetaCpu>,
    hogel_lookup: Vec<u32>,
    grid_width: u32,
    grid_height: u32,
}

fn is_png_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
}

fn parse_numeric_groups(path: &Path) -> Vec<u32> {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };

    let mut values = Vec::new();
    let mut current = String::new();

    for ch in stem.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            if let Ok(value) = current.parse::<u32>() {
                values.push(value);
            }
            current.clear();
        }
    }

    if !current.is_empty() {
        if let Ok(value) = current.parse::<u32>() {
            values.push(value);
        }
    }

    values
}

fn push_image_data(
    path: &Path,
    rgb_bytes: &mut Vec<u8>,
    image_meta: &mut Vec<ImageMetaCpu>,
) -> u32 {
    let image_index = image_meta.len() as u32;
    let (width, height, decoded_rgba) = decode_png_rgba8(path);
    let offset_bytes = rgb_bytes.len() as u64;

    for rgba in decoded_rgba.chunks_exact(4) {
        rgb_bytes.extend_from_slice(&rgba[..3]);
    }

    image_meta.push(ImageMetaCpu {
        offset_bytes,
        width,
        height,
    });
    image_index
}

fn pack_rgb_bytes_to_layered_texture(
    rgb_bytes: &[u8],
    image_meta: &[ImageMetaCpu],
) -> (Image, Vec<[u32; 4]>, UVec4) {
    let words_per_row = usize::try_from(PIXEL_TEXTURE_WORDS_PER_ROW)
        .unwrap_or_else(|_| panic!("invalid PIXEL_TEXTURE_WORDS_PER_ROW"));
    let rows_per_layer = usize::try_from(PIXEL_TEXTURE_ROWS_PER_LAYER)
        .unwrap_or_else(|_| panic!("invalid PIXEL_TEXTURE_ROWS_PER_LAYER"));
    let words_per_layer = words_per_row
        .checked_mul(rows_per_layer)
        .unwrap_or_else(|| panic!("pixel words-per-layer overflow"));
    let bytes_per_layer = words_per_layer
        .checked_mul(4)
        .unwrap_or_else(|| panic!("pixel bytes-per-layer overflow"));

    if words_per_layer == 0 {
        panic!("pixel texture layer layout cannot be zero-sized");
    }

    let aligned_total_bytes = rgb_bytes.len().div_ceil(4).max(1) * 4;
    let layer_count = aligned_total_bytes.div_ceil(bytes_per_layer).max(1);
    let tex_byte_len = layer_count
        .checked_mul(bytes_per_layer)
        .unwrap_or_else(|| panic!("pixel texture byte size overflow"));
    let mut tex_data = vec![0u8; tex_byte_len];
    tex_data[..rgb_bytes.len()].copy_from_slice(rgb_bytes);

    let bytes_per_layer_u64 = u64::try_from(bytes_per_layer)
        .unwrap_or_else(|_| panic!("bytes-per-layer cannot fit in u64"));
    let mut shader_meta = Vec::with_capacity(image_meta.len());
    for meta in image_meta {
        let base_layer_u64 = meta.offset_bytes / bytes_per_layer_u64;
        let base_byte_u64 = meta.offset_bytes % bytes_per_layer_u64;
        let base_layer = u32::try_from(base_layer_u64)
            .unwrap_or_else(|_| panic!("base pixel layer index exceeds u32"));
        let base_byte_in_layer = u32::try_from(base_byte_u64)
            .unwrap_or_else(|_| panic!("base byte-in-layer exceeds u32"));
        shader_meta.push([base_byte_in_layer, meta.width, meta.height, base_layer]);
    }

    let layer_count_u32 = u32::try_from(layer_count)
        .unwrap_or_else(|_| panic!("pixel texture layer count exceeds u32"));
    let bytes_per_layer_u32 = u32::try_from(bytes_per_layer)
        .unwrap_or_else(|_| panic!("pixel bytes-per-layer exceeds u32"));

    let size = Extent3d {
        width: PIXEL_TEXTURE_WORDS_PER_ROW,
        height: PIXEL_TEXTURE_ROWS_PER_LAYER,
        depth_or_array_layers: layer_count_u32,
    };
    let mut image = Image::new(
        size,
        TextureDimension::D2,
        tex_data,
        TextureFormat::Rgba8Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        array_layer_count: Some(layer_count_u32),
        ..Default::default()
    });

    let pixel_layout = UVec4::new(
        PIXEL_TEXTURE_WORDS_PER_ROW,
        PIXEL_TEXTURE_ROWS_PER_LAYER,
        bytes_per_layer_u32,
        layer_count_u32,
    );
    (image, shader_meta, pixel_layout)
}

fn load_half_parallax_images(directory: &Path) -> DecodedImages {
    let entries = fs::read_dir(directory).unwrap_or_else(|err| {
        panic!(
            "failed to read image directory '{}': {err}",
            directory.display()
        )
    });

    let mut ordered_paths: Vec<(u32, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| is_png_path(path))
        .filter_map(|path| {
            let groups = parse_numeric_groups(&path);
            groups.last().copied().map(|order| (order, path))
        })
        .collect();

    ordered_paths.sort_by_key(|(order, _)| *order);

    if ordered_paths.is_empty() {
        panic!(
            "no valid .png files with trailing numeric order found in '{}'",
            directory.display()
        );
    }

    let mut rgb_bytes = Vec::new();
    let mut image_meta = Vec::with_capacity(ordered_paths.len());
    let mut hogel_lookup = Vec::with_capacity(ordered_paths.len());

    for (_order, path) in ordered_paths {
        let image_index = push_image_data(&path, &mut rgb_bytes, &mut image_meta);
        hogel_lookup.push(image_index);
    }

    let grid_width = u32::try_from(hogel_lookup.len())
        .unwrap_or_else(|_| panic!("too many half-parallax images for u32 grid width"));

    DecodedImages {
        mode: ParallaxMode::Half,
        rgb_bytes,
        image_meta,
        hogel_lookup,
        grid_width,
        grid_height: 1,
    }
}

fn load_full_parallax_images(directory: &Path) -> DecodedImages {
    let entries = fs::read_dir(directory).unwrap_or_else(|err| {
        panic!(
            "failed to read image directory '{}': {err}",
            directory.display()
        )
    });

    let mut indexed_paths: Vec<(u32, u32, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| is_png_path(path))
        .filter_map(|path| {
            let groups = parse_numeric_groups(&path);
            if groups.len() < 2 {
                return None;
            }
            let col = groups[groups.len() - 2];
            let row = groups[groups.len() - 1];
            if col % 2 == 0 || row % 2 == 0 {
                return None;
            }
            Some((col/2, row/2, path))
        })
        .collect();

    if indexed_paths.is_empty() {
        panic!(
            "no valid full-parallax PNGs found in '{}' (need two numeric indices: col,row)",
            directory.display()
        );
    }

    indexed_paths.sort_by_key(|(col, row, _)| (*row, *col));

    let min_col = indexed_paths.iter().map(|(col, _, _)| *col).min().unwrap();
    let max_col = indexed_paths.iter().map(|(col, _, _)| *col).max().unwrap();
    let min_row = indexed_paths.iter().map(|(_, row, _)| *row).min().unwrap();
    let max_row = indexed_paths.iter().map(|(_, row, _)| *row).max().unwrap();

    let grid_width = max_col - min_col + 1;
    let grid_height = max_row - min_row + 1;
    let grid_len = usize::try_from(grid_width)
        .ok()
        .and_then(|w| usize::try_from(grid_height).ok().map(|h| w * h))
        .unwrap_or_else(|| panic!("full-parallax grid is too large: {grid_width}x{grid_height}"));

    let mut rgb_bytes = Vec::new();
    let mut image_meta = Vec::with_capacity(indexed_paths.len());
    let mut hogel_lookup = vec![INVALID_IMAGE_INDEX; grid_len];

    for (col, row, path) in indexed_paths {
        let image_index = push_image_data(&path, &mut rgb_bytes, &mut image_meta);

        let local_col = col - min_col;
        let local_row = row - min_row;
        let lookup_index = usize::try_from(local_row * grid_width + local_col)
            .unwrap_or_else(|_| panic!("grid index overflow for '{}'", path.display()));

        if hogel_lookup[lookup_index] != INVALID_IMAGE_INDEX {
            panic!(
                "duplicate full-parallax cell for col={}, row={} (file '{}')",
                col,
                row,
                path.display()
            );
        }

        hogel_lookup[lookup_index] = image_index;
    }

    DecodedImages {
        mode: ParallaxMode::Full,
        rgb_bytes,
        image_meta,
        hogel_lookup,
        grid_width,
        grid_height,
    }
}

fn load_images_for_mode(directory: &Path, mode: ParallaxMode) -> DecodedImages {
    match mode {
        ParallaxMode::Half => load_half_parallax_images(directory),
        ParallaxMode::Full => load_full_parallax_images(directory),
    }
}

fn decode_png_rgba8(path: &Path) -> (u32, u32, Vec<u8>) {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|err| panic!("failed to open '{}': {err}", path.display()));
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

    let mut reader = decoder
        .read_info()
        .unwrap_or_else(|err| panic!("failed to parse PNG header '{}': {err}", path.display()));
    let output_size = reader.output_buffer_size().unwrap_or_else(|| {
        panic!(
            "failed to determine PNG output size for '{}'",
            path.display()
        )
    });
    let mut buffer = vec![0; output_size];
    let frame_info = reader
        .next_frame(&mut buffer)
        .unwrap_or_else(|err| panic!("failed to decode PNG '{}': {err}", path.display()));
    let data = &buffer[..frame_info.buffer_size()];

    let mut rgba = Vec::with_capacity((frame_info.width * frame_info.height * 4) as usize);
    match frame_info.color_type {
        png::ColorType::Rgba => rgba.extend_from_slice(data),
        png::ColorType::Rgb => {
            for rgb in data.chunks_exact(3) {
                rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
        }
        png::ColorType::Grayscale => {
            for gray in data {
                rgba.extend_from_slice(&[*gray, *gray, *gray, 255]);
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for ga in data.chunks_exact(2) {
                rgba.extend_from_slice(&[ga[0], ga[0], ga[0], ga[1]]);
            }
        }
        png::ColorType::Indexed => {
            panic!(
                "unexpected indexed PNG output after expansion for '{}'",
                path.display()
            );
        }
    }

    (frame_info.width, frame_info.height, rgba)
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    cli: Res<CliArgs>,
) {
    let orbit = OrbitCamera {
        target: Vec3::ZERO,
        distance: 3.0,
        min_distance: 0.5,
        max_distance: 20.0,
        yaw: -0.5,
        pitch: 0.35,
        rotate_sensitivity: 0.005,
        zoom_sensitivity: 0.15,
    };

    let decoded = load_images_for_mode(&cli.image_directory, cli.mode);
    let (pixel_image, shader_image_meta, pixel_layout) =
        pack_rgb_bytes_to_layered_texture(&decoded.rgb_bytes, &decoded.image_meta);

    let image_meta_buffer = storage_buffers.add(ShaderStorageBuffer::from(shader_image_meta));
    let image_pixels_texture = images.add(pixel_image);
    let hogel_lookup_buffer = storage_buffers.add(ShaderStorageBuffer::from(decoded.hogel_lookup));

    let initial_camera = orbit.translation();
    let mut face_dir = Vec3::new(initial_camera.x, 0.0, initial_camera.z);
    if face_dir.length_squared() < 1e-8 {
        face_dir = Vec3::Z;
    } else {
        face_dir = face_dir.normalize();
    }
    let yaw = face_dir.x.atan2(face_dir.z);
    let plane_rotation =
        Quat::from_rotation_y(yaw) * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let plane_center_world = Vec3::ZERO;
    let plane_normal_world = (plane_rotation * Vec3::Y).normalize();
    let plane_right_world = (plane_rotation * Vec3::X).normalize();
    let plane_up_world = (plane_rotation * Vec3::Z).normalize();

    commands.spawn((
        Mesh3d(
            meshes.add(
                Plane3d::default()
                    .mesh()
                    .size(cli.plane_width, cli.plane_height),
            ),
        ),
        MeshMaterial3d(materials.add(CustomMaterial {
            camera_world_pos: orbit.translation().extend(1.0),
            config: UVec4::new(
                decoded.mode.as_u32(),
                decoded.grid_width,
                decoded.grid_height,
                0,
            ),
            plane_center_world: plane_center_world.extend(1.0),
            plane_normal_world: plane_normal_world.extend(0.0),
            plane_right_world: plane_right_world.extend(0.0),
            plane_up_world: plane_up_world.extend(0.0),
            image_meta: image_meta_buffer,
            image_pixels: image_pixels_texture,
            hogel_lookup: hogel_lookup_buffer,
            plane_size: Vec4::new(cli.plane_width, cli.plane_height, 0.0, 0.0),
            pixel_layout,
            alpha_mode: AlphaMode::Opaque,
        })),
        Transform::from_rotation(plane_rotation),
    ));

    commands.spawn((
        Camera3d::default(),
        Tonemapping::None,
        Transform::from_translation(orbit.translation()).looking_at(orbit.target, Vec3::Y),
        orbit,
        OrbitMainCamera,
    ));
}

fn orbit_camera_controls(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    mut query: Query<(&mut OrbitCamera, &mut Transform), With<OrbitMainCamera>>,
) {
    let Ok((mut orbit, mut transform)) = query.single_mut() else {
        return;
    };

    let mouse_delta = mouse_motion.delta;

    if mouse_buttons.pressed(MouseButton::Left) {
        orbit.yaw -= mouse_delta.x * orbit.rotate_sensitivity;
        orbit.pitch += mouse_delta.y * orbit.rotate_sensitivity;
        orbit.pitch = orbit.pitch.clamp(-1.5, 1.5);
    }

    let scroll_delta = mouse_scroll.delta.y;

    if scroll_delta != 0.0 {
        orbit.distance *= 1.0 - scroll_delta * orbit.zoom_sensitivity;
        orbit.distance = orbit.distance.clamp(orbit.min_distance, orbit.max_distance);
    }

    transform.translation = orbit.translation();
    transform.look_at(orbit.target, Vec3::Y);
}

fn sync_camera_uniform(
    camera_query: Query<&Transform, (With<Camera3d>, With<OrbitMainCamera>)>,
    mut materials: ResMut<Assets<CustomMaterial>>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };

    let camera_position = camera_transform.translation.extend(1.0);
    for (_, material) in materials.iter_mut() {
        material.camera_world_pos = camera_position;
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct CustomMaterial {
    #[uniform(0)]
    camera_world_pos: Vec4,
    #[uniform(1)]
    config: UVec4,
    #[storage(2, read_only)]
    image_meta: Handle<ShaderStorageBuffer>,
    #[storage_texture(
        3,
        access = ReadOnly,
        image_format = Rgba8Uint,
        dimension = "2d_array",
        visibility(fragment)
    )]
    image_pixels: Handle<Image>,
    #[uniform(4)]
    plane_center_world: Vec4,
    #[uniform(5)]
    plane_normal_world: Vec4,
    #[uniform(6)]
    plane_right_world: Vec4,
    #[uniform(7)]
    plane_up_world: Vec4,
    #[storage(8, read_only)]
    hogel_lookup: Handle<ShaderStorageBuffer>,
    #[uniform(9)]
    plane_size: Vec4,
    #[uniform(10)]
    pixel_layout: UVec4,
    alpha_mode: AlphaMode,
}

impl Material for CustomMaterial {
    fn vertex_shader() -> ShaderRef {
        VERTEX_SHADER_ASSET_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        FRAGMENT_SHADER_ASSET_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
}
