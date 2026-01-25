use crate::backend::Backend;
use std::marker::PhantomData;
pub trait ScalarOps:
    Clone
    + Copy
    + Send
    + Sync
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn exp(self) -> Self;
}

// === Реализации для f64 (и f32, если нужно) ===
impl ScalarOps for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn exp(self) -> Self {
        self.exp()
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Scalar<B: Backend> {
    pub(crate) data: B::Scalar,
    pub(crate) backend: PhantomData<B>,
}

impl<B: Backend> Scalar<B> {
    pub fn new(f: f64) -> Self {
        Self {
            data: B::scalar_f64(f),
            backend: PhantomData,
        }
    }

    pub fn exp(&self) -> Self {
        Self {
            data: self.data.exp(),
            backend: PhantomData,
        }
    }
}

// Арифметические операции через std::ops
impl<B: Backend + Copy> std::ops::Add for Scalar<B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            data: self.data + rhs.data,
            backend: PhantomData,
        }
    }
}

impl<B: Backend> std::ops::Sub for Scalar<B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            data: self.data - rhs.data,
            backend: PhantomData,
        }
    }
}

impl<B: Backend> std::ops::Mul for Scalar<B> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            data: self.data * rhs.data,
            backend: PhantomData,
        }
    }
}

impl<B: Backend> std::ops::Div for Scalar<B> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self {
            data: self.data / rhs.data,
            backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend; // Убедись, что CpuBackend доступен

    #[test]
    fn test_scalar_ops_f64() {
        let a = 4.0f64;
        assert_eq!(a.sqrt(), 2.0);
        assert_eq!(a.abs(), 4.0);
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(f64::from_f64(3.14), 3.14);
        assert_eq!(3.14f64.to_f64(), 3.14);
        assert_eq!(1.0f64.exp(), std::f64::consts::E);
    }

    #[test]
    fn test_scalar_new_and_exp() {
        let s: Scalar<CpuBackend> = Scalar::new(1.0);
        assert_eq!(s.data, 1.0);

        let e = s.exp();
        assert!((e.data - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_arithmetic() {
        let a: Scalar<CpuBackend> = Scalar::new(5.0);
        let b: Scalar<CpuBackend> = Scalar::new(2.0);

        let sum = a + b;
        assert_eq!(sum.data, 7.0);

        let diff = sum - Scalar::new(3.0);
        assert_eq!(diff.data, 4.0);

        let prod = diff * Scalar::new(0.5);
        assert_eq!(prod.data, 2.0);

        let quot = prod / Scalar::new(4.0);
        assert_eq!(quot.data, 0.5);
    }

    #[test]
    fn test_scalar_type_safety() {
        // Этот код не должен компилироваться:
        // let cpu_scalar: Scalar<CpuBackend> = Scalar::new(1.0);
        // let ndarray_scalar: Scalar<NdarrayBackend> = Scalar::new(2.0);
        // let _ = cpu_scalar + ndarray_scalar; // <-- ошибка типов

        // Но мы можем проверить, что арифметика работает только внутри одного бэкенда.
        // Тест проходит, если компиляция вышеуказанных строк невозможна — это гарантия Rust.
        // Поэтому просто убедимся, что операции с одним бэкендом работают.
        let x: Scalar<CpuBackend> = Scalar::new(10.0);
        let y: Scalar<CpuBackend> = Scalar::new(3.0);
        let _ = x / y; // должно компилироваться
    }
}
