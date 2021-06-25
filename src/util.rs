use png::{BitDepth, ColorType};
use std::io::{self, ErrorKind};
use std::path::Path;

pub struct Png {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub color_type: ColorType,
    pub bit_depth: BitDepth,
    pub line_size: usize,
}

pub async fn read_png<P: AsRef<Path>>(path: P) -> Result<Png, io::Error> {
    let bytes = tokio::fs::read(path).await?;
    let decoder = png::Decoder::new(bytes.as_slice());
    let (info, mut reader) = decoder.read_info().map_err(|_| ErrorKind::InvalidData)?;
    let mut data = vec![0; info.buffer_size()];
    reader
        .next_frame(&mut data)
        .map_err(|_| ErrorKind::InvalidData)?;

    Ok(Png {
        data,
        width: info.width,
        height: info.height,
        color_type: info.color_type,
        bit_depth: info.bit_depth,
        line_size: info.line_size,
    })
}

pub async fn convert_to_png(bytes: &[u8]) -> Result<Png, io::Error> {
    let decoder = png::Decoder::new(bytes);
    let (info, mut reader) = decoder.read_info().map_err(|_| ErrorKind::InvalidData)?;
    let mut data = vec![0; info.buffer_size()];
    reader
        .next_frame(&mut data)
        .map_err(|_| ErrorKind::InvalidData)?;
    Ok(Png {
        data,
        width: info.width,
        height: info.height,
        color_type: info.color_type,
        bit_depth: info.bit_depth,
        line_size: info.line_size,
    })
}
