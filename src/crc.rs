// Algorithm based on Adler-32, credits go to Mark Adler for creating the original.
const CRCTAB: [u32; 256] = [
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
    0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
    0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
    0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
    0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
    0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
    0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
    0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e, 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
    0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
    0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
    0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
    0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a, 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
    0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
    0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
    0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
    0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
    0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
    0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
    0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
    0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
    0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
    0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94, 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d,
];

const DEFAULT_HASH: u32 = 0xffffffff;
const CRC_BITS: usize = 32;
const CHUNK_SIZE: usize = 256;

#[derive(Default)]
pub struct Hasher<'hash> {
    streams: Vec<&'hash [u8]>,
}

impl<'hash> Hasher<'hash> {
    #[inline]
    pub fn update(&mut self, data: &'hash [u8]) {
        let size = (data.len() as f64 / CHUNK_SIZE as f64).ceil() as usize;

        self.streams.reserve(size);
        self.streams.extend(data.chunks(CHUNK_SIZE))
    }

    /// Calculates current crc32 hash, takes &self as the hash can continue to be updated.
    pub async fn finalize(&self) -> u32 {
        use futures::stream::{FuturesOrdered, StreamExt};
        use std::sync::{Arc, Mutex};

        match self.streams.len() {
            0 => return DEFAULT_HASH,
            1 => return crc32_sync(self.streams[0]),
            _ => {}
        }

        // A Vec with the (pos, pad_byte_count)
        let padded_chunks = Arc::new(Mutex::new(Vec::new()));
        let mut futures = FuturesOrdered::new();

        for (idx, stream) in self.streams.iter().enumerate() {
            let sender = Arc::clone(&padded_chunks);

            futures.push(async move {
                let mut bytes = Vec::new();
                let bytes: &[u8] = if stream.len() == CHUNK_SIZE {
                    stream
                } else {
                    let len = stream.len();

                    bytes.reserve(CHUNK_SIZE);
                    bytes.extend(*stream);

                    // TODO: add safe method to pad to `CHUNK_SIZE` bytes
                    unsafe {
                        bytes.set_len(CHUNK_SIZE);
                        bytes[len + 1..].fill(0);
                    }

                    sender.lock().unwrap().push((idx, CHUNK_SIZE - len));
                    &bytes
                };

                crc32_sync(bytes)
            })
        }

        let futures: Vec<u32> = futures.collect().await;
        let mut pad_pos = 0;
        let mut crc = 0;

        // Move out of Arc<Mutex<_>>
        let padded_chunks = Arc::try_unwrap(padded_chunks).unwrap().into_inner().unwrap();
        for (idx, next) in futures.iter().enumerate() {
            crc = crc32_combine(crc, *next, CHUNK_SIZE);

            if padded_chunks.get(pad_pos).map(|v| v.0) == Some(idx) {
                crc = crc32_remove_zeros(crc, padded_chunks[pad_pos].1);

                pad_pos += 1;
            }
        }

        crc
    }
}

/// Calculate's CRC32, requires `data` to be a multiple of 4
#[inline]
pub fn crc32_sync(data: &[u8]) -> u32 {
    debug_assert!(data.len() % 4 == 0);
    crc32_sync_impl(DEFAULT_HASH, data)
}

/// Calculate's CRC32
#[inline]
pub async fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Hasher::default();
    hasher.update(data);
    hasher.finalize().await
}

#[inline]
fn crc32_sync_impl(crc: u32, data: &[u8]) -> u32 {
    #[cfg(target_feature = "sse2")]
    unsafe {
        crc32_simd_impl(crc, data)
    }
    #[cfg(not(target_feature = "sse2"))]
    crc32_sync_impl(crc, data)
}

#[inline]
fn crc32_impl(mut crc: u32, data: &[u8]) -> u32 {
    if data.is_empty() {
        return crc;
    }

    for byte in data {
        crc = CRCTAB[((crc ^ *byte as u32) & 0xff) as usize] ^ (crc >> 8);
    }

    crc ^ !0
}

unsafe fn crc32_simd_impl(crc: u32, mut data: &[u8]) -> u32 {
    // based on the http://intel.ly/2ySEwL0 paper
    use crate::intrinsics::*;
    use std::mem::transmute;

    // Buffer has to be at least 64 bytes and a multiple of 16...
    if data.len() < 64 {
        return crc32_impl(DEFAULT_HASH, data);
    }

    let remainder = {
        let remainder = data.len() % 16;
        if remainder == 0 {
            None
        } else {
            data = &data[..data.len() - remainder];
            Some(&data[data.len() - remainder..])
        }
    };

    // CRC32+Barrett polynomials.
    const K1K2: __m128i = unsafe { transmute([0x0154442bd4u64, 0x01c6e41596u64]) };
    const K3K4: __m128i = unsafe { transmute([0x01751997d0u64, 0x00ccaa009eu64]) };
    const K5K0: __m128i = unsafe { transmute([0x0163cd6124u64, 0x0000000000u64]) };
    const POLY: __m128i = unsafe { transmute([0x01db710641u64, 0x01f7011641u64]) };

    let (
        mut x0,
        mut x1,
        mut x2,
        mut x3,
        mut x4,
        mut x5,
        mut x6,
        mut x7,
        mut x8,
        mut y5,
        mut y6,
        mut y7,
        mut y8,
    );

    // there's at least one block of 64.
    x1 = _mm_loadu_si128(data.as_ptr().offset(0x00) as *const __m128i);
    x2 = _mm_loadu_si128(data.as_ptr().offset(0x10) as *const __m128i);
    x3 = _mm_loadu_si128(data.as_ptr().offset(0x20) as *const __m128i);
    x4 = _mm_loadu_si128(data.as_ptr().offset(0x30) as *const __m128i);

    x1 = _mm_xor_si128(x1, _mm_cvtsi32_si128(crc as i32));
    x0 = _mm_load_si128(&K1K2);

    data = &data[64..];

    // parallel fold blocks of 64, if any.
    while data.len() >= 64 {
        x5 = _mm_clmulepi64_si128::<0x00>(x1, x0);
        x6 = _mm_clmulepi64_si128::<0x00>(x2, x0);
        x7 = _mm_clmulepi64_si128::<0x00>(x3, x0);
        x8 = _mm_clmulepi64_si128::<0x00>(x4, x0);

        x1 = _mm_clmulepi64_si128::<0x11>(x1, x0);
        x2 = _mm_clmulepi64_si128::<0x11>(x2, x0);
        x3 = _mm_clmulepi64_si128::<0x11>(x3, x0);
        x4 = _mm_clmulepi64_si128::<0x11>(x4, x0);

        y5 = _mm_loadu_si128(data.as_ptr().offset(0x10) as *const __m128i);
        y6 = _mm_loadu_si128(data.as_ptr().offset(0x10) as *const __m128i);
        y7 = _mm_loadu_si128(data.as_ptr().offset(0x20) as *const __m128i);
        y8 = _mm_loadu_si128(data.as_ptr().offset(0x30) as *const __m128i);

        x1 = _mm_xor_si128(x1, x5);
        x2 = _mm_xor_si128(x2, x6);
        x3 = _mm_xor_si128(x3, x7);
        x4 = _mm_xor_si128(x4, x8);

        x1 = _mm_xor_si128(x1, y5);
        x2 = _mm_xor_si128(x2, y6);
        x3 = _mm_xor_si128(x3, y7);
        x4 = _mm_xor_si128(x4, y8);

        data = &data[64..];
    }

    // fold into 128-bits.
    x0 = _mm_load_si128(&K3K4);

    x5 = _mm_clmulepi64_si128::<0x00>(x1, x0);
    x1 = _mm_clmulepi64_si128::<0x11>(x1, x0);
    x1 = _mm_xor_si128(x1, x2);
    x1 = _mm_xor_si128(x1, x5);

    x5 = _mm_clmulepi64_si128::<0x00>(x1, x0);
    x1 = _mm_clmulepi64_si128::<0x11>(x1, x0);
    x1 = _mm_xor_si128(x1, x3);
    x1 = _mm_xor_si128(x1, x5);

    x5 = _mm_clmulepi64_si128::<0x00>(x1, x0);
    x1 = _mm_clmulepi64_si128::<0x11>(x1, x0);
    x1 = _mm_xor_si128(x1, x4);
    x1 = _mm_xor_si128(x1, x5);

    // single fold blocks of 16, if any.
    while data.len() >= 16 {
        x2 = _mm_loadu_si128(data.as_ptr() as *const __m128i);

        x5 = _mm_clmulepi64_si128::<0x00>(x1, x0);
        x1 = _mm_clmulepi64_si128::<0x11>(x1, x0);
        x1 = _mm_xor_si128(x1, x2);
        x1 = _mm_xor_si128(x1, x5);

        data = &data[16..];
    }

    // fold 128-bits to 64-bits.
    x2 = _mm_clmulepi64_si128::<0x10>(x1, x0);
    x3 = _mm_setr_epi32(!0, 0, !0, 0);
    x1 = _mm_srli_si128::<8>(x1);
    x1 = _mm_xor_si128(x1, x2);

    x0 = _mm_loadl_epi64(&K5K0);

    x2 = _mm_srli_si128::<4>(x1);
    x1 = _mm_and_si128(x1, x3);
    x1 = _mm_clmulepi64_si128::<0x00>(x1, x0);
    x1 = _mm_xor_si128(x1, x2);

    // barret reduce to 32-bits.
    x0 = _mm_load_si128(&POLY);

    x2 = _mm_and_si128(x1, x3);
    x2 = _mm_clmulepi64_si128::<0x10>(x2, x0);
    x2 = _mm_and_si128(x2, x3);
    x2 = _mm_clmulepi64_si128::<0x00>(x2, x0);
    x1 = _mm_xor_si128(x1, x2);

    let mut crc = _mm_extract_epi32::<1>(x1) as u32;
    if let Some(bytes) = remainder {
        for byte in bytes {
            crc = CRCTAB[((crc ^ *byte as u32) & 0xff) as usize] ^ (crc >> 8);
        }
    }

    crc ^ !0
}

/// Merge two's CRC32 such that result = crc32(dataB, lengthB, crc32(dataA, lengthA))
fn crc32_combine(mut acrc: u32, bcrc: u32, mut blen: usize) -> u32 {
    // based on Jean-loup Gailly and Mark Adler's crc_combine from
    // https://github.com/madler/zlib/blob/master/crc32.c

    // degenerated case
    if blen == 0 {
        return acrc;
    }

    let mut odd = [0u32; CRC_BITS]; // odd-power-of-two zeros operator
    let mut even = [0u32; CRC_BITS]; // even-power-of-two zeros operator

    // put operator for one zero bit in odd
    odd[0] = 0xEDB88320; // CRC-32 polynomial

    let mut row = 1;
    for odd_one in odd.iter_mut().take(CRC_BITS).skip(1) {
        *odd_one = row;
        row <<= 1;
    }

    // put operator for two zero bits in even
    for (square, mat) in even.iter_mut().zip(odd) {
        *square = gf2_matrix_times(&odd, mat);
    }

    // put operator for four zero bits in odd
    for (square, mat) in odd.iter_mut().zip(even) {
        *square = gf2_matrix_times(&even, mat);
    }

    let even = &mut even;
    let odd = &mut odd;
    while blen != 0 {
        for (square, mat) in even.iter_mut().zip(odd.iter()) {
            *square = gf2_matrix_times(odd, *mat);
        }

        if (blen & 1) != 0 {
            acrc = gf2_matrix_times(even, acrc);
        }

        blen >>= 1;
        std::mem::swap(even, odd);
    }

    acrc ^ bcrc
}

#[inline]
fn gf2_matrix_times(mat: &[u32; CRC_BITS], mut vec: u32) -> u32 {
    let (mut sum, mut idx) = (0, 0);
    while vec != 0 {
        if (vec & 1) != 0 {
            sum ^= mat[idx];
        }

        vec >>= 1;
        idx += 1;
    }

    sum
}

/// Remove's padding from stream
#[inline]
fn crc32_remove_zeros(mut crc: u32, mut n: usize) -> u32 {
    crc = !crc;

    while n != 0 {
        for _ in 0..8 {
            crc = if (crc & 0x80000000) == 0 { crc << 1 } else { (crc << 1) ^ 0xdb710641u32 }
        }

        n -= 1;
    }

    !crc
}

// #[inline]
// fn log2(val: usize) -> usize {
//     unsafe {
//         let res: usize;
//         asm!("bsr eax, edi", out("eax") res, in("edi") val);
//         res
//     }
// }

#[cfg(test)]
mod test {
    use rand::{thread_rng, Rng};

    #[test]
    fn crc32() {
        let random_data = {
            let mut rng = thread_rng();
            vec![rng.gen::<u8>(); 1024]
        };

        let base_hash = {
            let mut hasher = crc32fast::Hasher::new();

            hasher.update(random_data.as_slice());
            hasher.finalize()
        };

        assert_eq!(base_hash, super::crc32_sync(random_data.as_slice()))
    }

    #[tokio::test]
    async fn crc32_hasher() {
        let mut rng = thread_rng();
        let (mut base_hasher, mut impl_hasher) =
            (crc32fast::Hasher::new(), super::Hasher::default());

        let sample_data = [16, 32, 64, 128, 256, 512, 1024].map(|v| vec![rng.gen::<u8>(); v]);
        for sample in sample_data.iter() {
            base_hasher.update(sample);
            impl_hasher.update(sample);
        }

        assert_eq!(base_hasher.finalize(), impl_hasher.finalize().await)
    }

    #[test]
    fn crc32_join() {
        let mut rng = thread_rng();

        // Generate sample's where the data and slice's are a multiple of 4
        let sample = vec![rng.gen::<u8>(); rng.gen_range(128 / 4..1024 / 4) * 4];
        let (left, right) = sample.split_at(rng.gen_range(1..sample.len() / 4) * 4);

        let (left_crc, right_crc) = (super::crc32_sync(left), super::crc32_sync(right));

        assert_eq!(
            super::crc32_sync(&sample[..]),
            super::crc32_combine(left_crc, right_crc, right.len())
        )
    }

    #[test]
    fn remove_zeros() {
        let mut rng = thread_rng();
        let sample = {
            let mut joined: Vec<u8> = Vec::with_capacity(512);

            joined.extend(vec![rng.gen::<u8>(); 356]);
            joined.extend(vec![0; 156]);
            joined
        };

        assert_eq!(
            super::crc32_remove_zeros(super::crc32_sync(&sample[..]), 156),
            super::crc32_sync(&sample[..512 - 156])
        );
    }
}
