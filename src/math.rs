use crate::intrinsics::*;

use std::mem::transmute;
use std::ops::Mul;

pub type Radians = f64;

pub type Point3 = Vector3;

pub type Matrix2<T> = Matrix<T, 4>;
pub type Matrix3<T> = Matrix<T, 9>;
pub type Matrix4<T> = Matrix<T, 16>;

#[repr(simd)]
#[derive(Debug, Clone, Copy)]
pub struct Vector2 {
    pub x: f64,
    pub y: f64,
}

#[repr(simd)]
#[derive(Debug, Clone, Copy)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[repr(simd)]
#[derive(Debug, Clone, Copy)]
pub struct Vector4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub h: f64,
}

// NOTE: `S` specifies the width**2
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
pub struct Matrix<T, const S: usize> {
    pub data: [T; S],
    pub width: usize,
}

#[macro_export]
macro_rules! vector {
    [$x:expr, $y:expr $(,)?] => {
        $crate::math::Vector2 { x:$x, y:$y }
    };

    [$x:expr, $y:expr, $z:expr $(,)?] => {
        $crate::math::Vector3 { x:$x, y:$y, z:$z }
    };

    [$x:expr, $y:expr, $z:expr, $h:expr $(,)?] => {
        $crate::math::Vector4 { x:$x, y:$y, z:$z, h:$h }
    };
}

#[macro_export]
macro_rules! matrix {
    [$a:expr, $b:expr,
     $c:expr, $d:expr $(,)?] => {
        $crate::math::Matrix {
            data: [$a, $b, $c, $d],
            width: 2,
        }
    };

    [$a:expr, $b:expr, $c:expr,
     $d:expr, $e:expr, $f:expr,
     $g:expr, $h:expr, $i:expr $(,)?] => {
        $crate::math::Matrix {
            data: [$a, $b, $c, $d, $e, $f, $g, $h, $i],
            width: 3,
        }
    };

    [$a:expr, $b:expr, $c:expr, $d:expr,
     $e:expr, $f:expr, $g:expr, $h:expr,
     $i:expr, $j:expr, $k:expr, $l:expr,
     $m:expr, $n:expr, $o:expr, $p:expr $(,)?] => {
        $crate::math::Matrix {
            data: [$a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p],
            width: 3,
        }
    };
}

impl PartialEq<Self> for Vector2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Mul<Self> for Vector2 {
    type Output = Self;

    #[inline]
    fn mul(self, vect: Self) -> Self::Output {
        unsafe {
            let a = _mm_load_pd(&self as *const Self as *const _);
            let b = _mm_load_pd(&vect as *const Self as *const _);

            transmute(_mm_mul_pd(a, b))
        }
    }
}

impl PartialEq<Self> for Vector3 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl Mul<Self> for Vector3 {
    type Output = Self;

    #[inline]
    fn mul(self, vect: Self) -> Self::Output {
        unsafe {
            let a = _mm256_load_pd(&self as *const Self as *const _);
            let b = _mm256_load_pd(&vect as *const Self as *const _);

            transmute(_mm256_mul_pd(a, b))
        }
    }
}

impl PartialEq<Self> for Vector4 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z && self.h == other.h
    }
}

impl Mul<Self> for Vector4 {
    type Output = Self;

    #[inline]
    fn mul(self, vect: Self) -> Self::Output {
        unsafe {
            let a = _mm256_load_pd(&self as *const Self as *const _);
            let b = _mm256_load_pd(&vect as *const Self as *const _);

            transmute(_mm256_mul_pd(a, b))
        }
    }
}

impl<T: PartialEq, const S: usize> PartialEq<Self> for Matrix<T, S> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Mul<Output = T> + Copy> Mul<Self> for Matrix<T, 4> {
    type Output = Self;

    #[inline]
    fn mul(self, v: Self) -> Self::Output {
        let (a, b) = (self.data, v.data);

        matrix![a[0] * b[0], a[1] * b[2], a[2] * b[0], a[3] * b[1]]
    }
}

impl Mul<Self> for Matrix<f64, 16> {
    type Output = Self;

    fn mul(self, vect: Self) -> Self::Output {
        if !is_x86_feature_detected!("avx") {
            panic!("x86 AVX doesn't seem to be supported");
        }

        unsafe {
            // Transpose matrix so we can just do a dot product on the rows
            let ax = _mm256_load_pd(self.data.as_ptr());
            let ay = _mm256_load_pd(self.data.as_ptr().add(4));
            let az = _mm256_load_pd(self.data.as_ptr().add(8));
            let ah = _mm256_load_pd(self.data.as_ptr().add(12));

            let bx = _mm256_load_pd(vect.data.as_ptr());
            let by = _mm256_load_pd(vect.data.as_ptr().add(4));
            let bz = _mm256_load_pd(vect.data.as_ptr().add(8));
            let bh = _mm256_load_pd(vect.data.as_ptr().add(12));

            let tbx = _mm256_unpacklo_pd(bx, by);
            let tby = _mm256_unpackhi_pd(bx, by);
            let tbz = _mm256_unpacklo_pd(bz, bh);
            let tbh = _mm256_unpackhi_pd(bz, bh);

            let bx = _mm256_permute2f128_pd(tbx, tbz, 0x20);
            let by = _mm256_permute2f128_pd(tby, tbh, 0x20);
            let bz = _mm256_permute2f128_pd(tbx, tbz, 0x31);
            let bh = _mm256_permute2f128_pd(tby, tbh, 0x31);

            // Perform a dot product on the rows of `self` and `other`
            //
            // NOTE: data must be stored in a `matrix` as `_mm256_store_pd`
            // requires `data` to be 32 byte aligned.
            let mut matrix = Matrix { data: [0f64; 16], width: 4 };
            let ptr = matrix.data.as_mut_ptr();

            for (idx, &src) in [ax, ay, az, ah].iter().enumerate() {
                let xx = _mm256_mul_pd(src, bx);
                let xy = _mm256_mul_pd(src, by);
                let xz = _mm256_mul_pd(src, bz);
                let xh = _mm256_mul_pd(src, bh);

                let t1 = _mm256_hadd_pd(xx, xy);
                let t2 = _mm256_hadd_pd(xz, xh);

                let swapped = _mm256_permute2f128_pd(t1, t2, 0x21);
                let blended = _mm256_blend_pd(t1, t2, 0b1100);
                let dotproduct = _mm256_add_pd(swapped, blended);

                _mm256_store_pd(ptr.add(idx * 4), dotproduct);
                _mm256_store_pd(ptr.add(idx * 4), dotproduct);
            }

            matrix
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn vector_mul() {
        let a = vector![10.0, 20.0];
        let b = vector![80.0, 90.0];
        let c = vector![800.0, 1800.0];

        assert_eq!(a * b, c);

        let a = vector![10.0, 20.0, 32.0];
        let b = vector![80.0, 90.0, 60.8];
        let c = vector![800.0, 1800.0, 1945.6];

        assert_eq!(a * b, c);

        let a = vector![10.0, 20.0, 32.0, 123.0];
        let b = vector![80.0, 90.0, 60.8, 11.2];
        let c = vector![800.0, 1800.0, 1945.6, 1377.6];

        assert_eq!(a * b, c);
    }

    #[rustfmt::skip]
    #[test]
    #[no_mangle]
    fn matrix_mul() {
        let a: Matrix4<f64> = matrix![
            5.0, 7.0,  9.0, 10.0,
            2.0, 3.0,  3.0, 8.0,
            8.0, 10.0, 2.0, 3.0,
            3.0, 3.0,  4.0, 8.0
        ];

        let b: Matrix4<f64> = matrix![
            3.0,  10.0, 12.0, 18.0,
            12.0, 1.0,  4.0,  9.0,
            9.0,  10.0, 12.0, 2.0,
            3.0,  12.0, 4.0,  10.0,
        ];

        let c: Matrix4<f64> = matrix![
            210.0, 267.0, 236.0, 271.0,
            93.0,  149.0, 104.0, 149.0,
            171.0, 146.0, 172.0, 268.0,
            105.0, 169.0, 128.0, 169.0,
        ];

        assert_eq!(a * b, c);
    }
}
