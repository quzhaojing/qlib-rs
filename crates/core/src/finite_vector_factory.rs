//! Typed construction and plugin registration for finite vector environments.

use std::{marker::PhantomData, str::FromStr, time::Duration};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use strum::{Display, EnumString};
use thiserror::Error;

use crate::{
    BoxedFiniteEnvironment, EnvironmentPluginError, FiniteDummyBackend, FiniteDummyBuildError,
    FiniteObservationPredicate, FiniteShmemBackend, FiniteShmemBuildError, FiniteSubprocessBackend,
    FiniteSubprocessBackendBuildError, FiniteSubprocessProgram, FiniteSubprocessProgramFactory,
    FiniteVectorBackend, FiniteVectorEnv, FiniteVectorLogger,
};

pub const FINITE_ENVIRONMENT_DESCRIPTOR_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Display, EnumString, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum FiniteEnvironmentKind {
    Dummy,
    Subproc,
    Shmem,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteEnvironmentDescriptor<Config> {
    pub version: u16,
    pub factory: String,
    pub config: Config,
}

impl<Config> FiniteEnvironmentDescriptor<Config> {
    #[must_use]
    pub fn new(factory: impl Into<String>, config: Config) -> Self {
        Self {
            version: FINITE_ENVIRONMENT_DESCRIPTOR_VERSION,
            factory: factory.into(),
            config,
        }
    }

    fn validate(&self) -> Result<(), FiniteVectorFactoryError> {
        if self.version != FINITE_ENVIRONMENT_DESCRIPTOR_VERSION {
            return Err(FiniteVectorFactoryError::UnsupportedDescriptorVersion {
                expected: FINITE_ENVIRONMENT_DESCRIPTOR_VERSION,
                actual: self.version,
            });
        }
        if self.factory.is_empty() {
            return Err(FiniteVectorFactoryError::EmptyFactoryName);
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum FiniteVectorFactoryError {
    #[error("unknown finite environment kind {value:?}")]
    UnknownEnvironmentKind { value: String },
    #[error("finite environment descriptor version {actual} is unsupported; expected {expected}")]
    UnsupportedDescriptorVersion { expected: u16, actual: u16 },
    #[error("finite environment descriptor factory name cannot be empty")]
    EmptyFactoryName,
    #[error("finite environment backend {kind} is not registered")]
    MissingBackend { kind: FiniteEnvironmentKind },
    #[error("finite environment factory {factory:?} is not registered for backend {kind}")]
    MissingFactory {
        kind: FiniteEnvironmentKind,
        factory: String,
    },
    #[error(
        "finite environment factory {factory:?} failed for backend {kind} worker {environment_id}: {source}"
    )]
    WorkerFactory {
        kind: FiniteEnvironmentKind,
        factory: String,
        environment_id: usize,
        #[source]
        source: EnvironmentPluginError,
    },
    #[error("finite dummy backend construction failed: {0}")]
    DummyBuild(#[source] FiniteDummyBuildError),
    #[error("finite subprocess backend construction failed: {0}")]
    SubprocessBuild(#[source] FiniteSubprocessBackendBuildError),
    #[error("finite shared-memory backend construction failed: {0}")]
    SharedMemoryBuild(#[source] FiniteShmemBuildError),
}

type BoxedBackend<Observation, Action, Reward, Info> =
    Box<dyn FiniteVectorBackend<Observation, Action, Reward, Info>>;
type BoxedBackendPlugin<Config, Observation, Action, Reward, Info> =
    Box<dyn FiniteVectorBackendPlugin<Config, Observation, Action, Reward, Info>>;
type BoxedDummyResolver<Config, Observation, Action, Reward, Info> =
    Box<dyn FiniteDummyEnvironmentResolver<Config, Observation, Action, Reward, Info>>;

pub trait FiniteVectorBackendPlugin<Config, Observation, Action, Reward, Info>: Send {
    /// # Errors
    /// Returns descriptor, registry, worker, or backend-construction failures.
    fn build(
        &mut self,
        descriptor: &FiniteEnvironmentDescriptor<Config>,
        concurrency: usize,
    ) -> Result<BoxedBackend<Observation, Action, Reward, Info>, FiniteVectorFactoryError>;
}

impl<Config, Observation, Action, Reward, Info, Plugin>
    FiniteVectorBackendPlugin<Config, Observation, Action, Reward, Info> for Plugin
where
    Plugin: FnMut(
            &FiniteEnvironmentDescriptor<Config>,
            usize,
        )
            -> Result<BoxedBackend<Observation, Action, Reward, Info>, FiniteVectorFactoryError>
        + Send,
{
    fn build(
        &mut self,
        descriptor: &FiniteEnvironmentDescriptor<Config>,
        concurrency: usize,
    ) -> Result<BoxedBackend<Observation, Action, Reward, Info>, FiniteVectorFactoryError> {
        self(descriptor, concurrency)
    }
}

pub struct FiniteVectorBackendRegistry<Config, Observation, Action, Reward, Info> {
    plugins: IndexMap<
        FiniteEnvironmentKind,
        BoxedBackendPlugin<Config, Observation, Action, Reward, Info>,
    >,
}

impl<Config, Observation, Action, Reward, Info> Default
    for FiniteVectorBackendRegistry<Config, Observation, Action, Reward, Info>
{
    fn default() -> Self {
        Self {
            plugins: IndexMap::new(),
        }
    }
}

impl<Config, Observation, Action, Reward, Info>
    FiniteVectorBackendRegistry<Config, Observation, Action, Reward, Info>
{
    #[must_use]
    pub fn register(
        &mut self,
        kind: FiniteEnvironmentKind,
        plugin: impl FiniteVectorBackendPlugin<Config, Observation, Action, Reward, Info> + 'static,
    ) -> bool {
        self.plugins.insert(kind, Box::new(plugin)).is_some()
    }

    pub fn registered_kinds(&self) -> impl Iterator<Item = FiniteEnvironmentKind> + '_ {
        self.plugins.keys().copied()
    }

    fn build(
        &mut self,
        kind: FiniteEnvironmentKind,
        descriptor: &FiniteEnvironmentDescriptor<Config>,
        concurrency: usize,
    ) -> Result<BoxedBackend<Observation, Action, Reward, Info>, FiniteVectorFactoryError> {
        self.plugins
            .get_mut(&kind)
            .ok_or(FiniteVectorFactoryError::MissingBackend { kind })?
            .build(descriptor, concurrency)
    }
}

/// Selects `dummy`, `subproc`, or `shmem` through a stable plugin registry and constructs the
/// transport-neutral finite vector coordinator.
///
/// # Errors
/// Returns invalid kind/descriptor, missing plugin/factory, worker, or backend failures.
pub fn vectorize_env<Config, Observation, Action, Reward, Info>(
    descriptor: &FiniteEnvironmentDescriptor<Config>,
    environment_kind: &str,
    concurrency: usize,
    registry: &mut FiniteVectorBackendRegistry<Config, Observation, Action, Reward, Info>,
    predicate: Box<dyn FiniteObservationPredicate<Observation>>,
    loggers: Vec<Box<dyn FiniteVectorLogger<Observation, Reward, Info>>>,
) -> Result<FiniteVectorEnv<Observation, Action, Reward, Info>, FiniteVectorFactoryError>
where
    Observation: Clone + 'static,
    Action: Clone + 'static,
    Reward: Clone + 'static,
    Info: Clone + 'static,
{
    let kind = FiniteEnvironmentKind::from_str(environment_kind).map_err(|_| {
        FiniteVectorFactoryError::UnknownEnvironmentKind {
            value: environment_kind.to_owned(),
        }
    })?;
    descriptor.validate()?;
    let backend = registry.build(kind, descriptor, concurrency)?;
    Ok(FiniteVectorEnv::new(backend, predicate, loggers))
}

pub trait FiniteDummyEnvironmentResolver<Config, Observation, Action, Reward, Info>: Send {
    /// # Errors
    /// Returns a named in-process environment construction failure.
    fn create(
        &mut self,
        environment_id: usize,
        config: &Config,
    ) -> Result<BoxedFiniteEnvironment<Observation, Action, Reward, Info>, EnvironmentPluginError>;
}

impl<Config, Observation, Action, Reward, Info, Resolver>
    FiniteDummyEnvironmentResolver<Config, Observation, Action, Reward, Info> for Resolver
where
    Resolver: FnMut(
            usize,
            &Config,
        ) -> Result<
            BoxedFiniteEnvironment<Observation, Action, Reward, Info>,
            EnvironmentPluginError,
        > + Send,
{
    fn create(
        &mut self,
        environment_id: usize,
        config: &Config,
    ) -> Result<BoxedFiniteEnvironment<Observation, Action, Reward, Info>, EnvironmentPluginError>
    {
        self(environment_id, config)
    }
}

pub struct FiniteDummyBackendPlugin<Config, Observation, Action, Reward, Info> {
    factories: IndexMap<String, BoxedDummyResolver<Config, Observation, Action, Reward, Info>>,
}

impl<Config, Observation, Action, Reward, Info> Default
    for FiniteDummyBackendPlugin<Config, Observation, Action, Reward, Info>
{
    fn default() -> Self {
        Self {
            factories: IndexMap::new(),
        }
    }
}

impl<Config, Observation, Action, Reward, Info>
    FiniteDummyBackendPlugin<Config, Observation, Action, Reward, Info>
{
    #[must_use]
    pub fn register(
        &mut self,
        name: impl Into<String>,
        factory: impl FiniteDummyEnvironmentResolver<Config, Observation, Action, Reward, Info>
        + 'static,
    ) -> bool {
        self.factories
            .insert(name.into(), Box::new(factory))
            .is_some()
    }
}

impl<Config, Observation, Action, Reward, Info>
    FiniteVectorBackendPlugin<Config, Observation, Action, Reward, Info>
    for FiniteDummyBackendPlugin<Config, Observation, Action, Reward, Info>
where
    Config: Send,
    Observation: Clone + Send + 'static,
    Action: Send + 'static,
    Reward: Clone + Send + 'static,
    Info: Clone + Send + 'static,
{
    fn build(
        &mut self,
        descriptor: &FiniteEnvironmentDescriptor<Config>,
        concurrency: usize,
    ) -> Result<BoxedBackend<Observation, Action, Reward, Info>, FiniteVectorFactoryError> {
        let factory = self.factories.get_mut(&descriptor.factory).ok_or_else(|| {
            FiniteVectorFactoryError::MissingFactory {
                kind: FiniteEnvironmentKind::Dummy,
                factory: descriptor.factory.clone(),
            }
        })?;
        let environments = (0..concurrency)
            .map(|environment_id| {
                factory
                    .create(environment_id, &descriptor.config)
                    .map_err(|source| FiniteVectorFactoryError::WorkerFactory {
                        kind: FiniteEnvironmentKind::Dummy,
                        factory: descriptor.factory.clone(),
                        environment_id,
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        FiniteDummyBackend::new(environments)
            .map(|backend| Box::new(backend) as BoxedBackend<Observation, Action, Reward, Info>)
            .map_err(FiniteVectorFactoryError::DummyBuild)
    }
}

pub trait FiniteSubprocessProgramResolver<Config>: Send {
    /// # Errors
    /// Returns a named subprocess-program resolution failure.
    fn resolve(
        &mut self,
        environment_id: usize,
        config: &Config,
    ) -> Result<FiniteSubprocessProgram, EnvironmentPluginError>;
}

impl<Config, Resolver> FiniteSubprocessProgramResolver<Config> for Resolver
where
    Resolver:
        FnMut(usize, &Config) -> Result<FiniteSubprocessProgram, EnvironmentPluginError> + Send,
{
    fn resolve(
        &mut self,
        environment_id: usize,
        config: &Config,
    ) -> Result<FiniteSubprocessProgram, EnvironmentPluginError> {
        self(environment_id, config)
    }
}

type SubprocessTypes<Action, Observation, Reward, Info, Payload> =
    fn() -> (Action, Observation, Reward, Info, Payload);

pub struct FiniteSubprocessBackendPlugin<Config, Action, Observation, Reward, Info, Payload> {
    factories: IndexMap<String, Box<dyn FiniteSubprocessProgramResolver<Config>>>,
    frame_limit: usize,
    operation_timeout: Duration,
    types: PhantomData<SubprocessTypes<Action, Observation, Reward, Info, Payload>>,
}

impl<Config, Action, Observation, Reward, Info, Payload>
    FiniteSubprocessBackendPlugin<Config, Action, Observation, Reward, Info, Payload>
{
    #[must_use]
    pub fn new(frame_limit: usize, operation_timeout: Duration) -> Self {
        Self {
            factories: IndexMap::new(),
            frame_limit,
            operation_timeout,
            types: PhantomData,
        }
    }

    #[must_use]
    pub fn register(
        &mut self,
        name: impl Into<String>,
        resolver: impl FiniteSubprocessProgramResolver<Config> + 'static,
    ) -> bool {
        self.factories
            .insert(name.into(), Box::new(resolver))
            .is_some()
    }
}

pub struct FiniteShmemBackendPlugin<Config, Action, Observation, Reward, Info, Payload> {
    factories: IndexMap<String, Box<dyn FiniteSubprocessProgramResolver<Config>>>,
    observation_capacity: usize,
    frame_limit: usize,
    operation_timeout: Duration,
    types: PhantomData<SubprocessTypes<Action, Observation, Reward, Info, Payload>>,
}

impl<Config, Action, Observation, Reward, Info, Payload>
    FiniteShmemBackendPlugin<Config, Action, Observation, Reward, Info, Payload>
{
    #[must_use]
    pub fn new(
        observation_capacity: usize,
        frame_limit: usize,
        operation_timeout: Duration,
    ) -> Self {
        Self {
            factories: IndexMap::new(),
            observation_capacity,
            frame_limit,
            operation_timeout,
            types: PhantomData,
        }
    }

    #[must_use]
    pub fn register(
        &mut self,
        name: impl Into<String>,
        resolver: impl FiniteSubprocessProgramResolver<Config> + 'static,
    ) -> bool {
        self.factories
            .insert(name.into(), Box::new(resolver))
            .is_some()
    }
}

impl<Config, Action, Observation, Reward, Info, Payload>
    FiniteVectorBackendPlugin<Config, Observation, Action, Reward, Info>
    for FiniteShmemBackendPlugin<Config, Action, Observation, Reward, Info, Payload>
where
    Config: Send,
    Action: Clone + Serialize + Send + Sync + 'static,
    Observation: DeserializeOwned + Serialize + Send + 'static,
    Reward: DeserializeOwned + Serialize + Send + 'static,
    Info: DeserializeOwned + Serialize + Send + 'static,
    Payload: Default + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn build(
        &mut self,
        descriptor: &FiniteEnvironmentDescriptor<Config>,
        concurrency: usize,
    ) -> Result<BoxedBackend<Observation, Action, Reward, Info>, FiniteVectorFactoryError> {
        let resolver = self.factories.get_mut(&descriptor.factory).ok_or_else(|| {
            FiniteVectorFactoryError::MissingFactory {
                kind: FiniteEnvironmentKind::Shmem,
                factory: descriptor.factory.clone(),
            }
        })?;
        let programs = (0..concurrency)
            .map(|environment_id| {
                resolver
                    .resolve(environment_id, &descriptor.config)
                    .map_err(|source| FiniteVectorFactoryError::WorkerFactory {
                        kind: FiniteEnvironmentKind::Shmem,
                        factory: descriptor.factory.clone(),
                        environment_id,
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        FiniteShmemBackend::<Action, Observation, Reward, Info, Payload>::from_programs(
            programs,
            self.observation_capacity,
            self.frame_limit,
            self.operation_timeout,
        )
        .map(|backend| Box::new(backend) as BoxedBackend<Observation, Action, Reward, Info>)
        .map_err(FiniteVectorFactoryError::SharedMemoryBuild)
    }
}

impl<Config, Action, Observation, Reward, Info, Payload>
    FiniteVectorBackendPlugin<Config, Observation, Action, Reward, Info>
    for FiniteSubprocessBackendPlugin<Config, Action, Observation, Reward, Info, Payload>
where
    Config: Send,
    Action: Clone + Serialize + Send + Sync + 'static,
    Observation: DeserializeOwned + Send + 'static,
    Reward: DeserializeOwned + Send + 'static,
    Info: DeserializeOwned + Send + 'static,
    Payload: Default + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn build(
        &mut self,
        descriptor: &FiniteEnvironmentDescriptor<Config>,
        concurrency: usize,
    ) -> Result<BoxedBackend<Observation, Action, Reward, Info>, FiniteVectorFactoryError> {
        let resolver = self.factories.get_mut(&descriptor.factory).ok_or_else(|| {
            FiniteVectorFactoryError::MissingFactory {
                kind: FiniteEnvironmentKind::Subproc,
                factory: descriptor.factory.clone(),
            }
        })?;
        let programs = (0..concurrency)
            .map(|environment_id| {
                resolver
                    .resolve(environment_id, &descriptor.config)
                    .map_err(|source| FiniteVectorFactoryError::WorkerFactory {
                        kind: FiniteEnvironmentKind::Subproc,
                        factory: descriptor.factory.clone(),
                        environment_id,
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut factory =
            FiniteSubprocessProgramFactory::new(programs, self.frame_limit, self.operation_timeout);
        FiniteSubprocessBackend::<Action, Observation, Reward, Info, Payload>::from_factory(
            concurrency,
            &mut factory,
        )
        .map(|backend| Box::new(backend) as BoxedBackend<Observation, Action, Reward, Info>)
        .map_err(FiniteVectorFactoryError::SubprocessBuild)
    }
}
