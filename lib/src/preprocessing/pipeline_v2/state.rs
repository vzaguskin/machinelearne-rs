//! Type-state markers for enforcing preprocessing order at compile time.
//!
//! The state machine:
//! ```text
//! Raw --> Imputed --> Scaled --> Encoded
//!  |        |          |           |
//!  |        |          |           +--> Model (ready)
//!  |        |          |
//!  |        |          +--> Encoder (optional)
//!  |        |
//!  |        +--> Scaler (required for most models)
//!  |
//!  +--> Imputer (handles missing values)
//! ```

#![allow(dead_code)] // Experimental module - some types not yet used

use std::marker::PhantomData;

/// Marker trait for data states.
pub trait State: Clone + Default + 'static {}

/// Marker trait for states that have been imputed.
pub trait RequiresImputation: State {}

/// Marker trait for states that have been scaled.
pub trait RequiresScaling: State {}

/// Marker trait for states that have been encoded.
pub trait RequiresEncoding: State {}

/// Raw data - no preprocessing applied yet.
#[derive(Clone, Copy, Default, Debug)]
pub struct Raw;

impl State for Raw {}

/// Data has been imputed (missing values filled).
#[derive(Clone, Copy, Default, Debug)]
pub struct Imputed;

impl State for Imputed {}
impl RequiresImputation for Imputed {}

/// Data has been scaled (normalized/standardized).
#[derive(Clone, Copy, Default, Debug)]
pub struct Scaled;

impl State for Scaled {}
impl RequiresImputation for Scaled {}
impl RequiresScaling for Scaled {}

/// Data has been encoded (categorical features converted).
#[derive(Clone, Copy, Default, Debug)]
pub struct Encoded;

impl State for Encoded {}
impl RequiresImputation for Encoded {}
impl RequiresScaling for Encoded {}
impl RequiresEncoding for Encoded {}

/// Type alias for data ready for model training/inference.
/// Requires imputation, scaling, and encoding.
pub type Ready = Encoded;

/// Type alias for data that's imputed and scaled (most common for numeric models).
pub type Preprocessed = Scaled;

/// Phantom data wrapper for state tracking.
#[derive(Clone, Copy, Default, Debug)]
pub struct StateMarker<S: State>(PhantomData<S>);

impl<S: State> StateMarker<S> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

/// Helper trait to check if a state transition is valid.
pub trait ValidTransition<From, To> {}

// Valid transitions
impl ValidTransition<Raw, Imputed> for () {}
impl ValidTransition<Imputed, Scaled> for () {}
impl ValidTransition<Scaled, Encoded> for () {}
impl ValidTransition<Raw, Scaled> for () {} // Allow skipping imputation if no missing values
impl ValidTransition<Imputed, Encoded> for () {} // Allow skipping scaling
impl ValidTransition<Raw, Encoded> for () {} // Allow both skips
