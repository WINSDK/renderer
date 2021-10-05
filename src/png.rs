/// Indexed-color => image representation in which each pixel of the original image is represented
/// by a single index into a palette.
///
/// Indexing => representing an image by a palette, an alpha table and
/// an array of indices pointing to entries in the palette and alpha table.
///
/// Palette => indexed table of three 8-bit sample values, red, green, and blue.
/// In an indexed image it defines the red, green, and blue sample values of the reference image.
/// In other cases, the palette may be a suggested palette that viewers may use on old limited.
/// Alpha samples may be defined for palette entries via the alpha table to reconstruct alpha samples.
///
/// Sample Depth => number of bits used to represent a sample. In an indexed-color image, samples
/// are stored in the palette thus the sample depth will always be 8, for any other PNG image
/// it is the same as the bit depth.
///
/// Sample Depth Scaling => mapping of a range of sample values onto the full range of the sample
/// depth of a PNG image
use std::borrow::Cow;
use std::convert::TryInto;
use std::io;
use std::path::Path;
use std::str::from_utf8;

use crate::crc::Hasher;

use async_compression::tokio::write::ZlibDecoder;
use tokio::{fs, io::AsyncWriteExt};
use tokio_stream::{self as stream, StreamExt};
use wgpu::TextureFormat;

#[derive(Debug, Clone, PartialEq)]
pub struct Png {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: Format,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Format {
    pub texture: TextureFormat,
    channel_width: usize,
}

#[derive(Debug)]
pub enum Error {
    InvalidSignature,
    CorruptedFile,
    Unimplimented(Cow<'static, str>),
    Unsupported(Cow<'static, str>),
    IoError(io::Error),
    Other(Cow<'static, str>),
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
            Error::invalid_signature()
        }
    }
}

pub type PngResult = Result<Png, Error>;

impl Png {
    /// WARNING: TextureFormat must match swapchain's TextureFormat
    pub async fn new(mut data: &mut [u8]) -> PngResult {
        // Check `magic bytes` to evaluate whether the file is a png.
        if &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0xA, 0x1A, 0x0A] != &data[..8] {
            return Error::invalid_signature();
        }

        data = {
            let mut pos = 8; // skipping magic bytes
            while &data[pos..][..4] != "IHDR".as_bytes() {
                pos += 1;
            }

            &mut data[pos - 4..] // go back a 4 bytes to include the type
        };

        let chunks = PngChunks::new(&mut data);
        let mut iter = stream::iter(chunks);
        let (ihdr_chunk, chunk_type) = iter.next().await.unwrap();
        if chunk_type != "IHDR" {
            return Error::invalid_signature();
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
            // only deflate/inflate is in the official standard. so only 0
            return Error::unsupported("Invalid compression method used in PNG");
        }

        if inter_method != &0 {
            return Error::unimplimented("Interlaced data isn't supported yet");
        }

        // TODO: handle other filter methods
        assert_eq!(filter_method, &0u8);

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
                        palette = Some(unsafe { &*(chunk.as_ref() as *const _) });
                    },
                    "IDAT" /* image data chunks */ => {
                        if let Some(palette) = palette {
                            decoder.write(&handle_palette(chunk, palette, &color_type)?
                                .unwrap_or(chunk.to_vec())).await.or_else(Error::ioerror)?;
                        } else {
                            decoder.write(chunk).await.or_else(Error::ioerror)?;
                        }
                    },
                    _ => {
                        // [image header, textual information]
                        let exceptions = ["IHDR", "tEXt"];
                        if !exceptions.contains(&chunk_type) {
                            unimplemented!("{} chunk handling", chunk_type);
                        }
                    }
                }
            }

            decoder.shutdown().await.or_else(Error::ioerror)?;
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

        Ok(Self { data, width, height, format: Format { texture: format, channel_width } })
    }

    pub async fn from_path<P: AsRef<Path>>(path: P) -> PngResult {
        let mut data = fs::read(path).await.map_err(Error::IoError)?;
        Png::new(&mut data).await
    }
}

struct PngChunks<'png> {
    slice: &'png mut [u8],
    pos: usize,
}

impl<'png> PngChunks<'png> {
    fn new(slice: &'png mut [u8]) -> Self {
        if slice.len() < 67 {
            panic!("PNG must be more than 67 bytes");
        }

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

        let mut crc = Hasher::new();
        crc.update(chunk_type.as_bytes());
        crc.update(data);

        assert_eq!(_checksum, crc.finalize());
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
        GrayScale | AlphaGrayScale => return Error::other("Invalid color type for palette"),
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
                chunk[pos + 3] = 0;
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

macro_rules! error_fn {
    ($name:ident, $enum:expr, $into:ty) => {
        #[allow(dead_code)]
        #[inline(always)]
        pub fn $name<U, T: Into<$into>>(s: T) -> Result<U, Self> {
            Err($enum(s.into()))
        }
    };
    ($name:ident, $enum:expr) => {
        #[allow(dead_code)]
        #[inline(always)]
        pub fn $name<U>() -> Result<U, Self> {
            Err($enum)
        }
    };
}

impl Error {
    error_fn!(invalid_signature, Error::InvalidSignature);

    error_fn!(corrupted_file, Error::CorruptedFile);

    error_fn!(unimplimented, Error::Unimplimented, Cow<'static, str>);

    error_fn!(unsupported, Error::Unsupported, Cow<'static, str>);

    error_fn!(ioerror, Error::IoError, io::Error);

    error_fn!(other, Error::Other, Cow<'static, str>);
}

#[cfg(all(target_family = "unix", test))]
mod test {
    use super::Png;
    use std::path::Path;

    #[tokio::test]
    async fn check_png_attributes() {
        use std::process::{Command, Stdio};
        use wgpu::TextureFormat;

        let path = Path::new("./res/joe_biden.png");
        let png = Png::from_path(path).await.unwrap();

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
            ("GrayscaleAlpha" | "Grayscale", 8) => TextureFormat::Rgba8Unorm,
            ("GrayscaleAlpha" | "Grayscale", 16) => TextureFormat::Rgba16Float,
            ("RGBA" | "RGB", 16) => TextureFormat::Rgba16Float,
            ("RGBA" | "RGB", 8) => TextureFormat::Rgba8Unorm,
            ("Indexed", 8) => unimplemented!("Indexed color is not yet supported"),
            _ => panic!("Unknown format encountered"),
        })
    }

    #[tokio::test]
    async fn convert_to_png() {
        let path = Path::new("./test_cases/joe_biden.png");
        let external_png = crate::read_png(&path).await.unwrap();
        let internal_png = crate::png::Png::from_path(&path).await.unwrap();
        assert_eq!(external_png.data, internal_png.data);
    }
}