use std::io;
use std::num::NonZeroU32;
use std::path::Path;

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub async fn new<'a, P: AsRef<Path>>(
        path: P,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        swap_chain_desc: &'a wgpu::SwapChainDescriptor,
    ) -> Result<Texture, io::Error> {
        let image = crate::read_png(path).await?;
        Ok(Self::from_png(image, device, queue, swap_chain_desc).await?)
    }

    pub async fn from_bytes(
        bytes: &[u8],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        swap_chain_desc: &wgpu::SwapChainDescriptor,
    ) -> Result<Texture, io::Error> {
        let image = crate::convert_to_png(bytes).await?;
        Ok(Self::from_png(image, device, queue, swap_chain_desc).await?)
    }

    /// WARNING: Png must be RGBA
    pub async fn from_png<'a>(
        image: crate::Png,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        swap_chain_desc: &wgpu::SwapChainDescriptor,
    ) -> Result<Texture, io::Error> {
        let size = wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size,
            label: None,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: swap_chain_desc.format,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });
        log::info!("Created texture");

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            image.data.as_slice(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * image.width),
                rows_per_image: NonZeroU32::new(image.height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }
}
