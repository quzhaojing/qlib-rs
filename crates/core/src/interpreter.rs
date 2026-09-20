//! Generic reinforcement-learning interpreter contracts and Gym-style validation.

use std::{error::Error, fmt};

use indexmap::IndexMap;

/// Unconstrained policy-observation type.
pub type ObsType<T> = T;

/// Unconstrained policy-action type.
pub type PolicyActType<T> = T;

/// Marker shared by both directions of interpreter.
pub trait Interpreter {}

/// A policy space capable of validating one sample.
pub trait SampleSpace<Sample> {
    type Error;

    /// Validate `sample` against this space.
    ///
    /// # Errors
    /// Returns the space-specific validation failure.
    fn validate(&self, sample: &Sample) -> Result<(), Self::Error>;
}

/// Converts simulator state into a validated policy observation.
pub trait StateInterpreter<State>: Interpreter {
    type Observation;
    type ObservationSpace: SampleSpace<Self::Observation, Error = Self::Error>;
    type Error;

    fn observation_space(&self) -> &Self::ObservationSpace;

    /// Convert one simulator state into a policy observation.
    ///
    /// # Errors
    /// Returns interpretation failures.
    fn interpret(&self, simulator_state: &State) -> Result<Self::Observation, Self::Error>;

    /// Validate an observation against `observation_space`.
    ///
    /// # Errors
    /// Returns the space-specific validation failure.
    fn validate(&self, observation: &Self::Observation) -> Result<(), Self::Error> {
        self.observation_space().validate(observation)
    }

    /// Interpret and then validate one simulator state.
    ///
    /// # Errors
    /// Returns interpretation or validation failures.
    fn call(&self, simulator_state: &State) -> Result<Self::Observation, Self::Error> {
        let observation = self.interpret(simulator_state)?;
        self.validate(&observation)?;
        Ok(observation)
    }
}

/// Validates a policy action and converts it into a simulator action.
pub trait ActionInterpreter<State, PolicyAction>: Interpreter {
    type Action;
    type ActionSpace: SampleSpace<PolicyAction, Error = Self::Error>;
    type Error;

    fn action_space(&self) -> &Self::ActionSpace;

    /// Convert one validated policy action into a simulator action.
    ///
    /// # Errors
    /// Returns interpretation failures.
    fn interpret(
        &self,
        simulator_state: &State,
        action: &PolicyAction,
    ) -> Result<Self::Action, Self::Error>;

    /// Validate an action against `action_space`.
    ///
    /// # Errors
    /// Returns the space-specific validation failure.
    fn validate(&self, action: &PolicyAction) -> Result<(), Self::Error> {
        self.action_space().validate(action)
    }

    /// Validate the policy action and then interpret it.
    ///
    /// # Errors
    /// Returns validation or interpretation failures.
    fn call(
        &self,
        simulator_state: &State,
        action: &PolicyAction,
    ) -> Result<Self::Action, Self::Error> {
        self.validate(action)?;
        self.interpret(simulator_state, action)
    }
}

/// The leaf predicate used by a recursive Gym-style space.
pub trait LeafSpace<Value> {
    fn contains(&self, value: &Value) -> bool;
}

/// A recursive Gym-style space whose leaves supply their own membership predicate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GymSpace<Leaf> {
    Dict(IndexMap<String, Self>),
    Tuple(Vec<Self>),
    Leaf(Leaf),
}

/// A recursive sample. Lists and arrays are distinct inputs but are promoted to tuples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GymSample<Value> {
    Dict(IndexMap<String, Self>),
    Tuple(Vec<Self>),
    List(Vec<Self>),
    Array(Vec<Self>),
    Leaf(Value),
}

impl<Leaf: fmt::Display> fmt::Display for GymSpace<Leaf> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dict(spaces) => {
                formatter.write_str("Dict(")?;
                write_entries(formatter, spaces.iter())?;
                formatter.write_str(")")
            }
            Self::Tuple(spaces) => {
                formatter.write_str("Tuple(")?;
                write_sequence(formatter, spaces)?;
                formatter.write_str(")")
            }
            Self::Leaf(space) => space.fmt(formatter),
        }
    }
}

impl<Value: fmt::Display> fmt::Display for GymSample<Value> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dict(values) => write_entries(formatter, values.iter()),
            Self::Tuple(values) | Self::List(values) | Self::Array(values) => {
                write_sequence(formatter, values)
            }
            Self::Leaf(value) => value.fmt(formatter),
        }
    }
}

fn write_entries<'a, Value: fmt::Display + 'a>(
    formatter: &mut fmt::Formatter<'_>,
    entries: impl Iterator<Item = (&'a String, &'a Value)>,
) -> fmt::Result {
    formatter.write_str("{")?;
    for (index, (key, value)) in entries.enumerate() {
        if index != 0 {
            formatter.write_str(", ")?;
        }
        write!(formatter, "{key}: {value}")?;
    }
    formatter.write_str("}")
}

fn write_sequence<Value: fmt::Display>(
    formatter: &mut fmt::Formatter<'_>,
    values: &[Value],
) -> fmt::Result {
    formatter.write_str("(")?;
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            formatter.write_str(", ")?;
        }
        value.fmt(formatter)?;
    }
    formatter.write_str(")")
}

/// Gym-style validation failure retaining the exact current space and sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GymSpaceValidationError<Leaf, Value> {
    pub message: String,
    pub space: Box<GymSpace<Leaf>>,
    pub sample: Box<GymSample<Value>>,
    source: Option<Box<Self>>,
}

impl<Leaf, Value> GymSpaceValidationError<Leaf, Value> {
    fn new(message: impl Into<String>, space: &GymSpace<Leaf>, sample: &GymSample<Value>) -> Self
    where
        Leaf: Clone,
        Value: Clone,
    {
        Self {
            message: message.into(),
            space: Box::new(space.clone()),
            sample: Box::new(sample.clone()),
            source: None,
        }
    }

    fn with_source(
        message: impl Into<String>,
        space: &GymSpace<Leaf>,
        sample: &GymSample<Value>,
        source: Self,
    ) -> Self
    where
        Leaf: Clone,
        Value: Clone,
    {
        Self {
            message: message.into(),
            space: Box::new(space.clone()),
            sample: Box::new(sample.clone()),
            source: Some(Box::new(source)),
        }
    }

    #[must_use]
    pub fn cause(&self) -> Option<&Self> {
        self.source.as_deref()
    }
}

impl<Leaf: fmt::Display, Value: fmt::Display> fmt::Display
    for GymSpaceValidationError<Leaf, Value>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}\n  Space: {}\n  Sample: {}",
            self.message, self.space, self.sample
        )
    }
}

impl<Leaf, Value> Error for GymSpaceValidationError<Leaf, Value>
where
    Leaf: Clone + fmt::Debug + fmt::Display + 'static,
    Value: Clone + fmt::Debug + fmt::Display + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

impl<Leaf, Value> SampleSpace<GymSample<Value>> for GymSpace<Leaf>
where
    Leaf: Clone + LeafSpace<Value>,
    Value: Clone,
{
    type Error = GymSpaceValidationError<Leaf, Value>;

    fn validate(&self, sample: &GymSample<Value>) -> Result<(), Self::Error> {
        gym_space_contains(self, sample)
    }
}

/// Strengthened Gym membership check with recursive failure context.
///
/// # Errors
/// Returns the failure at the current space and chains a nested failure when present.
pub fn gym_space_contains<Leaf, Value>(
    space: &GymSpace<Leaf>,
    sample: &GymSample<Value>,
) -> Result<(), GymSpaceValidationError<Leaf, Value>>
where
    Leaf: Clone + LeafSpace<Value>,
    Value: Clone,
{
    match space {
        GymSpace::Dict(spaces) => {
            let GymSample::Dict(values) = sample else {
                return Err(GymSpaceValidationError::new(
                    "Sample must be a dict with same length as space.",
                    space,
                    sample,
                ));
            };
            if values.len() != spaces.len() {
                return Err(GymSpaceValidationError::new(
                    "Sample must be a dict with same length as space.",
                    space,
                    sample,
                ));
            }
            for (key, subspace) in spaces {
                let Some(value) = values.get(key) else {
                    return Err(GymSpaceValidationError::new(
                        format!("Key {key} not found in sample."),
                        space,
                        sample,
                    ));
                };
                if let Err(source) = gym_space_contains(subspace, value) {
                    return Err(GymSpaceValidationError::with_source(
                        format!("Subspace of key {key} validation error."),
                        space,
                        sample,
                        source,
                    ));
                }
            }
            Ok(())
        }
        GymSpace::Tuple(spaces) => {
            let (GymSample::Tuple(values) | GymSample::List(values) | GymSample::Array(values)) =
                sample
            else {
                return Err(GymSpaceValidationError::new(
                    "Sample must be a tuple with same length as space.",
                    space,
                    sample,
                ));
            };
            if values.len() != spaces.len() {
                let normalized = GymSample::Tuple(values.clone());
                return Err(GymSpaceValidationError::new(
                    "Sample must be a tuple with same length as space.",
                    space,
                    &normalized,
                ));
            }
            for (index, (subspace, value)) in spaces.iter().zip(values).enumerate() {
                if let Err(source) = gym_space_contains(subspace, value) {
                    let normalized = GymSample::Tuple(values.clone());
                    return Err(GymSpaceValidationError::with_source(
                        format!("Subspace of index {index} validation error."),
                        space,
                        &normalized,
                        source,
                    ));
                }
            }
            Ok(())
        }
        GymSpace::Leaf(leaf) => match sample {
            GymSample::Leaf(value) if leaf.contains(value) => Ok(()),
            _ => Err(GymSpaceValidationError::new(
                "Validation error reported by gym.",
                space,
                sample,
            )),
        },
    }
}
