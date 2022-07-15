#![allow(dead_code)]

use std::path::Path;
use std::str::from_utf8;

use async_compression::tokio::write::ZlibDecoder;
use futures::stream::{self, StreamExt};
use tokio::{fs, io::AsyncWriteExt};
use wgpu::TextureFormat;

#[derive(Debug)]
pub enum Error {
    /// Invalid signature in header
    InvalidSignature,

    /// only deflate/inflate is in the official standard
    InvalidCompression,

    /// Invalid iCCP profile name
    InvalidICCP,

    /// Invalid color type for a given palette
    InvalidColorType,

    /// Some header or data combination is found to be impossible
    CorruptedFile,

    /// Support not yet added
    Unimplimented,

    /// Support not planned
    Unsupported,

    /// Generic IO bound error
    IO(std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Png {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: Format,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub texture: TextureFormat,
    channel_width: usize,

    has_iccp_profile: bool,

    // iCCP profile can only be 79 characters
    iccp_profile: [u8; 79],
}

impl Format {
    fn iccp_profile(&self) -> Option<&str> {
        if self.has_iccp_profile {
            return from_utf8(&self.iccp_profile).ok();
        }

        None
    }
}

#[allow(dead_code)]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq)]
enum ColorType {
    GrayScale = 0,
    Truecolor = 2,
    Indexed = 3,
    AlphaGrayScale = 4,
    AlphaTruecolor = 6,
}

impl ColorType {
    fn new(val: u8) -> Result<Self, Error> {
        // section 3.1.12
        if [0, 2, 3, 4, 6].contains(&val) {
            Ok(unsafe { std::mem::transmute(val) })
        } else {
            Err(Error::InvalidSignature)
        }
    }
}

pub type PngResult = Result<Png, Error>;

impl Png {
    /// WARNING: TextureFormat must match swapchain's TextureFormat
    pub async fn new(mut data: &mut [u8]) -> PngResult {
        // Check `magic bytes` to evaluate whether the file is a png.
        if [0x89, 0x50, 0x4E, 0x47, 0x0D, 0xA, 0x1A, 0x0A] != data[..8] {
            return Err(Error::InvalidSignature);
        }

        data = {
            let mut pos = 8; // skipping magic bytes
            while &data[pos..][..4] != "IHDR".as_bytes() {
                pos += 1;
            }

            &mut data[pos - 4..] // go back a 4 bytes to include the type
        };

        let chunks = PngChunks::new(data);
        let mut iter = stream::iter(chunks);
        let (ihdr_chunk, chunk_type) = iter.next().await.unwrap();
        if chunk_type != "IHDR" {
            return Err(Error::InvalidSignature);
        }

        // *IHDR chunk*
        // Width:              4 bytes
        // Height:             4 bytes
        // Bit depth:          1 byte
        // Color type:         1 byte
        // Compression method: 1 byte
        // Filter method:      1 byte
        // Interlace method:   1 byte
        let width = be_slice_to_u32(&ihdr_chunk[..4]);
        let height = be_slice_to_u32(&ihdr_chunk[4..8]);
        let (bit_depth, color_type, comp_method, filter_method, inter_method) = {
            let s = ihdr_chunk;
            (&s[8], ColorType::new(s[9])?, &s[10], &s[11], &s[12])
        };

        if comp_method != &0 {
            return Err(Error::InvalidCompression);
        }

        if inter_method != &0 {
            // Interlaced data isn't supported yet
            return Err(Error::Unimplimented);
        }

        // TODO: handle other filter methods
        assert_eq!(filter_method, &0u8);

        let mut iccp_profile = [0u8; 79];
        let mut has_iccp_profile = false;
        let mut data: Vec<u8> = {
            // TODO: predict decoder output
            let mut decoder = ZlibDecoder::new(Vec::with_capacity(((width + 1) * height) as usize));
            let mut palette = None;

            while let Some((chunk, chunk_type)) = iter.next().await {
                match chunk_type {
                    "PLTE" /* palette table */ => {
                        use ColorType::*;

                        // PLTE chunks come in only 8-bit grayscale or rgb
                        assert!(bit_depth >= &8);

                        // section 11.2.3
                        assert!(![Truecolor, Indexed, AlphaTruecolor].contains(&color_type));
                        palette = unsafe { (chunk as *const [u8]).as_ref() };
                    },
                    "IDAT" /* image data chunks */ => {
                        if let Some(palette) = palette {
                            decoder.write(&handle_palette(chunk, palette, &color_type)?
                                .unwrap_or_else(|| chunk.to_vec())).await.map_err(Error::IO)?;
                        } else {
                            decoder.write(chunk).await.map_err(Error::IO)?;
                        }
                    },
                    "iCCP" => {
                        // TODO: ICCP color profile decoder, THIS ISN'T REALLY NECESSARY.
                        for (idx, charac) in chunk[..80].iter().enumerate() {
                            let charac = *charac;
                            match charac {
                                0 => return Err(Error::InvalidICCP),
                                161..=255 | 32..=126 => {
                                    iccp_profile.copy_from_slice(&chunk[..idx]);
                                    has_iccp_profile = true;
                                },
                                _ => return Err(Error::InvalidICCP)
                            }
                        }

                        if !has_iccp_profile {
                            return Err(Error::InvalidSignature);
                        }
                    },
                    // [image header, [textual information], ICC Profile]
                    "IHDR" | "tEXt" | "iTXt" | "zTXt" => {},
                    _ => {
                        // TODO: handle ICC profiles
                        unimplemented!("{} chunk handling", chunk_type);
                    }
                }
            }

            decoder.shutdown().await.map_err(Error::IO)?;
            decoder.into_inner()
        };

        let (format, channel_width) = match (bit_depth, color_type) {
            (bits, ColorType::GrayScale) => match bits {
                16 => (TextureFormat::R16Uint, 2),
                8 => (TextureFormat::R8Uint, 1),
                4 => {
                    data.iter_mut().for_each(|v| *v *= 15); // 4-bit to 8-bit
                    (TextureFormat::R8Uint, 1)
                }
                2 => {
                    data.iter_mut().for_each(|v| *v *= 85); // 2-bit to 8-bit
                    (TextureFormat::R8Uint, 1)
                }
                1 => {
                    data.iter_mut().for_each(|v| *v *= 255); // 1-bit to 8-bit
                    (TextureFormat::R8Uint, 1)
                }
                _ => panic!("{} is not a valid bit depth", bits),
            },
            (bits, ColorType::Truecolor) => match bits {
                16 => (TextureFormat::Rgba16Uint, 8),
                8 => (TextureFormat::Rgba8Uint, 4),
                _ => panic!("{} is not a valid bit depth", bits),
            },
            (_, ColorType::Indexed) => (TextureFormat::Rgba8Uint, 4),
            (bits, ColorType::AlphaGrayScale) => match bits {
                16 => (TextureFormat::Rgba16Uint, 8),
                8 => (TextureFormat::Rgba8Uint, 4),
                _ => panic!("{} is not a valid bit depth", bits),
            },
            (bits, ColorType::AlphaTruecolor) => {
                let (format, channel_width) = match bits {
                    16 => (TextureFormat::Rgba16Uint, 8),
                    8 => (TextureFormat::Rgba8Uint, 4),
                    _ => panic!("{} is not a valid bit depth", bits),
                };

                let mut pos = 0;
                let scanline_width = channel_width * width as usize;
                while pos != data.len() {
                    data.remove(pos);
                    pos += scanline_width;
                }

                (format, channel_width)
            }
        };

        Ok(Self {
            data,
            width,
            height,
            format: Format { 
                texture: format, 
                has_iccp_profile,
                iccp_profile,
                channel_width 
            },
        })
    }

    pub async fn from_path<P: AsRef<Path>>(path: P) -> PngResult {
        let mut data = fs::read(path).await.map_err(Error::IO)?;
        Png::new(&mut data).await
    }

    pub fn to_rgba8(&self) -> Self {
        unimplemented!()
    }
}

struct PngChunks<'png> {
    slice: &'png mut [u8],
    pos: usize,
}

impl<'png> PngChunks<'png> {
    fn new(slice: &'png mut [u8]) -> Self {
        assert!(slice.len() < 67, "PNG must be more than 67 bytes");
        Self { slice, pos: 0 }
    }
}

impl<'png> Iterator for PngChunks<'png> {
    type Item = (&'png mut [u8], &'png str);

    fn next(&mut self) -> Option<Self::Item> {
        let (len, chunk_type, data, _checksum) = {
            // SAFETY: every call to next() returns references to parts
            // of the slice that do not overlap, you're never reading or
            // writting from the same chunk of the slice.
            let s = unsafe { &mut (*(self.slice as *mut [u8]))[self.pos..] };

            let (meta, data) = s.split_at_mut(8);
            let len = be_slice_to_u32(&meta[..4]) as usize;
            let chunk_type = from_utf8(&meta[4..8]).expect("Failed to parse chunk type");
            if chunk_type == "IEND" {
                return None;
            }

            self.pos += 8 + len + 4;

            let (data, remainder) = data[..len + 4].split_at_mut(len);
            (len, chunk_type, data, be_slice_to_u32(remainder))
        };

        if chunk_type == "IHDR" {
            assert_eq!(len, 13);
            assert_eq!(data.len(), len);
        }

        #[cfg(debug_assertions)]
        match chunk_type {
            "IHDR" => {
                assert_eq!(len, 13);
                assert_eq!(data.len(), len);
            }
            "PLTE" => {
                assert!(len >= 1);
                assert!(len <= 256 * 3);
                assert!(len % 3 == 0);
            }
            _ => (),
        }

        //assert_eq!(_checksum, crc);
        Some((data, chunk_type))
    }
}

fn handle_palette(
    idat_chunk: &mut [u8],
    palette: &[u8],
    color_type: &ColorType,
) -> Result<Option<Vec<u8>>, Error> {
    use ColorType::*;

    // section 11.2.3
    match color_type {
        GrayScale | AlphaGrayScale => Err(Error::InvalidColorType),
        Truecolor | AlphaTruecolor => {
            // should probably be using a sPLT
            Ok(None)
        }
        Indexed => {
            let mut chunk = vec![0; idat_chunk.len() * 4];

            let mut pos = 0;
            for idx in idat_chunk.iter() {
                let channel = &palette[(*idx as usize / 3)..];
                chunk[pos] = channel[0];
                chunk[pos + 1] = channel[1];
                chunk[pos + 2] = channel[2];
                pos += 4;
            }

            Ok(Some(chunk))
        }
    }
}

// assumes slice is of len 4 and big endian.
#[inline(always)]
fn be_slice_to_u32(slice: &[u8]) -> u32 {
    u32::from_be_bytes(slice.try_into().unwrap())
}

#[cfg(test)]
mod test {
    use std::path::Path;

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn check_png_attributes() {
        use std::process::{Command, Stdio};
        use wgpu::TextureFormat;

        let path = Path::new("./test_cases/joe_biden.png");
        let png = super::Png::from_path(path).await.unwrap();

        let output = {
            let process = Command::new("file")
                .stdout(Stdio::piped())
                .arg(path)
                .spawn()
                .expect("failed to spawn `files` process");

            let output = process.wait_with_output().unwrap();
            let code = output.status.code().unwrap();
            if code != 0 {
                panic!("failed to run `files` process with exit code: {}", code);
            }

            output
        };

        let output = std::str::from_utf8(&output.stdout).unwrap();
        let info: Vec<&str> = output.split_ascii_whitespace().collect();
        let info: Vec<&str> = info.iter().map(|s| s.trim_end_matches(',')).collect();

        assert_eq!(info[4], png.width.to_string());
        assert_eq!(info[6], png.height.to_string());

        let bit_depth =
            u32::from_str_radix(info[7].split_terminator('-').next().unwrap(), 10).unwrap();

        assert_eq!(png.format.texture, match (info[8], bit_depth) {
            ("GrayscaleAlpha" | "Grayscale", 8) => TextureFormat::Rgba8Uint,
            ("GrayscaleAlpha" | "Grayscale", 16) => TextureFormat::Rgba16Uint,
            ("RGBA" | "RGB", 16) => TextureFormat::Rgba16Uint,
            ("RGBA" | "RGB", 8) => TextureFormat::Rgba8Uint,
            _ => panic!("Unknown format encountered"),
        })
    }

    #[tokio::test]
    async fn convert_to_png() {
        let path = Path::new("./test_cases/joe_biden.png");
        let external_png = crate::read_png(&path).await.unwrap();
        let internal_png = crate::png::Png::from_path(&path).await.unwrap();
        assert_eq!(external_png.data, internal_png.data, "png data doesn't match");
    }
}
