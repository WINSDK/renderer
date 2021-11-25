use num_traits::float::Float;

#[repr(transparent)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Radians<T: Float>(pub T);

impl<T: Float> From<T> for Radians<T> {
    fn from(val: T) -> Self {
        Self(val)
    }
}
