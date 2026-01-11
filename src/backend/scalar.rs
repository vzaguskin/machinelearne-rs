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
}

// === Реализации для f64 (и f32, если нужно) ===
impl ScalarOps for f64 {
    fn sqrt(self) -> Self { self.sqrt() }
    fn abs(self) -> Self { self.abs() }
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn from_f64(v: f64) -> Self { v }
    fn to_f64(self) -> f64 { self }
    
}

#[derive(Clone, Debug, Copy)]
pub struct Scalar <B: Backend> 
{
    pub(crate) data: B::Scalar,
    pub(crate) backend: PhantomData<B>,
}

impl <B: Backend> Scalar<B>{
    pub fn new(f: f64) -> Self{
        Self{data: B::scalar_f64(f), backend: PhantomData}
    }
}

// Арифметические операции через std::ops
impl<B: Backend+Copy> std::ops::Add for Scalar<B> {
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