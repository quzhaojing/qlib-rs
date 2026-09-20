//! Native Qlib recurrent features over Candle layers and automatic differentiation.
//! This backend boundary uses tensors; it is not a stable external plugin ABI.

use crate::training_vessel_runner::TrainingPolicyMode;
use candle_core::{DType, Error, IndexOp, Module, Result, Shape, Tensor, Var};
use candle_nn::{GRU, GRUConfig, Init, LSTM, LSTMConfig, Linear, RNN, VarBuilder};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RecurrentKind {
    Rnn,
    Lstm,
    Gru,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct RecurrentConfig {
    pub data_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub kind: RecurrentKind,
    pub layers: usize,
}
impl RecurrentConfig {
    #[must_use]
    pub const fn new(data_dim: usize) -> Self {
        Self {
            data_dim,
            hidden_dim: 64,
            output_dim: 32,
            kind: RecurrentKind::Gru,
            layers: 1,
        }
    }
}

/// Batched full-history tensors. Step/tick indices are explicitly I64 vectors;
/// floating observations retain their dtype (private features follow source F32 casts).
#[derive(Clone)]
pub struct RecurrentObservation {
    pub data_processed: Tensor,
    pub cur_tick: Tensor,
    pub cur_step: Tensor,
    pub position_history: Tensor,
    pub target: Tensor,
    pub num_step: Tensor,
    pub acquiring: Tensor,
}

impl RecurrentObservation {
    /// Select minibatch positions consistently across every observation field.
    /// Keeps index order and duplicates; actor evaluation later detaches inputs.
    /// # Errors
    /// Propagates invalid index dtype/rank/range or inconsistent field dimensions.
    pub fn select_batch(&self, indices: &Tensor) -> Result<Self> {
        Ok(Self {
            data_processed: self.data_processed.index_select(indices, 0)?,
            cur_tick: self.cur_tick.index_select(indices, 0)?,
            cur_step: self.cur_step.index_select(indices, 0)?,
            position_history: self.position_history.index_select(indices, 0)?,
            target: self.target.index_select(indices, 0)?,
            num_step: self.num_step.index_select(indices, 0)?,
            acquiring: self.acquiring.index_select(indices, 0)?,
        })
    }

    /// Qlib heads convert a Batch through Tianshou's deep-copying `to_torch`.
    /// These immutable tensor views preserve its input-gradient boundary.
    #[must_use]
    pub fn detached(&self) -> Self {
        Self {
            data_processed: self.data_processed.detach(),
            cur_tick: self.cur_tick.detach(),
            cur_step: self.cur_step.detach(),
            position_history: self.position_history.detach(),
            target: self.target.detach(),
            num_step: self.num_step.detach(),
            acquiring: self.acquiring.detach(),
        }
    }
}

// Candle drops empty matmul dependencies. Empty slices followed by reduction
// keep exact zero gradients (not absent gradients), including non-finite inputs.
pub(crate) fn empty_link(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    left.flatten_all()?.narrow(0, 0, 0)?.sum_all()?
        + right.flatten_all()?.narrow(0, 0, 0)?.sum_all()?
}

// Torch retains signed zero and NaN, and chooses derivative zero at x == 0.
// Candle's relu uses max(x, 0) with a >= 0 derivative; native comparisons and
// selection preserve the source rule without custom tensor kernels/autograd.
fn relu(input: &Tensor) -> Result<Tensor> {
    let inactive = input
        .lt(0.)?
        .where_cond(&input.zeros_like()?, &input.detach())?;
    input.le(0.)?.where_cond(&inactive, input)
}

// Explicit flatten/unflatten handles empty leading dimensions which Candle's
// inferred reshape cannot recover. CPU reduced-precision GEMM accumulates in F32.
pub(crate) fn linear_forward(layer: &Linear, input: &Tensor) -> Result<Tensor> {
    let mut shape = input.dims().to_vec();
    let channels = shape
        .pop()
        .ok_or_else(|| Error::Msg("linear input must have a dimension".into()))?;
    let rows = shape.iter().product::<usize>();
    let output_dim = layer.weight().dim(0)?;
    if channels != layer.weight().dim(1)? || input.dtype() != layer.weight().dtype() {
        return Err(Error::Msg(
            "linear input and weight shape/dtype mismatch".into(),
        ));
    }
    let flat = input.reshape((rows, channels))?;
    let output = if rows == 0 || channels == 0 || output_dim == 0 {
        let mut zero = empty_link(input, layer.weight())?;
        if let Some(bias) = layer.bias() {
            zero = empty_link(&zero, bias)?;
        }
        let output = zero.broadcast_as((rows, output_dim))?;
        match layer.bias() {
            Some(bias) => output.broadcast_add(bias)?,
            None => output,
        }
    } else if matches!(input.dtype(), DType::F16 | DType::BF16) {
        let weight = layer.weight().to_dtype(DType::F32)?;
        let bias = layer
            .bias()
            .map(|bias| bias.to_dtype(DType::F32))
            .transpose()?;
        Linear::new(weight, bias)
            .forward(&flat.to_dtype(DType::F32)?)?
            .to_dtype(input.dtype())?
    } else {
        layer.forward(&flat)?
    };
    shape.push(output_dim);
    output.reshape(shape)
}

pub trait CandleFeatureExtractor: Send + Sync {
    fn output_dim(&self) -> usize;
    fn parameters(&self) -> &IndexMap<String, Tensor>;
    /// # Errors
    /// Returns observation-shape, indexing, dtype or backend computation failures.
    fn forward(&self, observation: &RecurrentObservation) -> Result<Tensor>;

    /// Rebuild the same architecture around the supplied named parameter handles.
    /// Implementations must retain these handles (including aliases), and copy
    /// any other model-local state rather than sharing mutable online state.
    /// # Errors
    /// Returns unsupported reconstruction or invalid parameter/backend errors.
    fn rebuild(
        &self,
        _parameters: IndexMap<String, Tensor>,
    ) -> Result<Arc<dyn CandleFeatureExtractor>> {
        Err(Error::Msg(
            "feature extractor does not implement target reconstruction".into(),
        ))
    }

    /// Native recurrent layers have no dropout/batch-normalization mode state.
    /// Stateful extractors can override this linked mode boundary.
    fn set_mode(&self, _mode: TrainingPolicyMode) {}
}

/// Copy logical parameters once per live identity, preserving whole-tensor aliases.
/// Detaching before Var construction is essential: `from_tensor` on a Var aliases it.
pub(crate) fn independent_parameters(
    parameters: &IndexMap<String, Tensor>,
) -> Result<IndexMap<String, Tensor>> {
    let mut memo = HashMap::new();
    parameters
        .iter()
        .map(|(name, tensor)| {
            let copied = if let Some(value) = memo.get(&tensor.id()) {
                Tensor::clone(value)
            } else {
                let value = if tensor.is_variable() {
                    Var::from_tensor(&tensor.detach())?.into_inner()
                } else {
                    tensor.detach().copy()?
                };
                memo.insert(tensor.id(), value.clone());
                value
            };
            Ok((name.clone(), copied))
        })
        .collect()
}

pub(crate) struct Parameters<'a> {
    pub builder: VarBuilder<'a>,
    pub tensors: IndexMap<String, Tensor>,
}
impl<'a> Parameters<'a> {
    pub fn new(builder: VarBuilder<'a>) -> Self {
        Self {
            builder,
            tensors: IndexMap::new(),
        }
    }
    fn get(&mut self, name: &str, shape: impl Into<Shape>, init: Init) -> Result<Tensor> {
        let tensor = self.builder.get_with_hints(shape, name, init)?;
        self.tensors.insert(name.into(), tensor.clone());
        Ok(tensor)
    }
    pub fn linear(&mut self, name: &str, input: usize, output: usize) -> Result<Linear> {
        let init = uniform(input);
        let weight = self.get(&format!("{name}.weight"), (output, input), init)?;
        let bias = self.get(&format!("{name}.bias"), output, init)?;
        Ok(Linear::new(weight, Some(bias)))
    }
}

// Torch calculates this initializer bound in floating point too. The dimension
// conversion is used only for that bound, never for shape/index arithmetic.
#[allow(clippy::cast_precision_loss)]
fn uniform(input: usize) -> Init {
    if input == 0 {
        return Init::Const(0.);
    }
    let bound = 1. / (input as f64).sqrt();
    Init::Uniform {
        lo: -bound,
        up: bound,
    }
}

enum Cell {
    Plain { input: Linear, hidden: Linear },
    Lstm(LSTM),
    Gru(GRU),
}

fn recurrent_stack(
    parameters: &mut Parameters<'_>,
    name: &str,
    config: RecurrentConfig,
) -> Result<Vec<Cell>> {
    let gates = match config.kind {
        RecurrentKind::Rnn => 1,
        RecurrentKind::Lstm => 4,
        RecurrentKind::Gru => 3,
    };
    let width = config
        .hidden_dim
        .checked_mul(gates)
        .ok_or_else(|| Error::Msg("recurrent gate dimension overflow".into()))?;
    let mut cells = Vec::new();
    for layer in 0..config.layers {
        let init = uniform(config.hidden_dim);
        let mut local = HashMap::new();
        for (parameter, shape) in [
            ("weight_ih", Shape::from((width, config.hidden_dim))),
            ("weight_hh", Shape::from((width, config.hidden_dim))),
            ("bias_ih", Shape::from(width)),
            ("bias_hh", Shape::from(width)),
        ] {
            let tensor = parameters.get(&format!("{name}.{parameter}_l{layer}"), shape, init)?;
            local.insert(format!("{parameter}_l0"), tensor);
        }
        // Candle cells are single-layer. Rebind the same live tensors to their
        // local layer-0 names; registration above retains Qlib's actual layer names.
        let cell = match config.kind {
            RecurrentKind::Rnn => Cell::Plain {
                input: Linear::new(
                    local["weight_ih_l0"].clone(),
                    Some(local["bias_ih_l0"].clone()),
                ),
                hidden: Linear::new(
                    local["weight_hh_l0"].clone(),
                    Some(local["bias_hh_l0"].clone()),
                ),
            },
            RecurrentKind::Lstm => Cell::Lstm(candle_nn::lstm(
                config.hidden_dim,
                config.hidden_dim,
                LSTMConfig::default(),
                VarBuilder::from_tensors(
                    local,
                    parameters.builder.dtype(),
                    parameters.builder.device(),
                ),
            )?),
            RecurrentKind::Gru => Cell::Gru(candle_nn::gru(
                config.hidden_dim,
                config.hidden_dim,
                GRUConfig::default(),
                VarBuilder::from_tensors(
                    local,
                    parameters.builder.dtype(),
                    parameters.builder.device(),
                ),
            )?),
        };
        cells.push(cell);
    }
    Ok(cells)
}

fn sequence(cells: &[Cell], input: &Tensor) -> Result<Tensor> {
    let (batch, length, hidden_dim) = input.dims3()?;
    if length == 0 {
        return Err(Error::Msg("recurrent sequence must not be empty".into()));
    }
    let mut output = input.clone();
    for cell in cells {
        let states = match cell {
            Cell::Plain { input, hidden } => {
                let mut state =
                    Tensor::zeros((batch, hidden_dim), output.dtype(), output.device())?;
                let mut states = Vec::new();
                for step in 0..length {
                    state = (linear_forward(input, &output.i((.., step, ..))?.contiguous()?)?
                        + linear_forward(hidden, &state)?)?
                    .tanh()?;
                    states.push(state.clone());
                }
                states
            }
            Cell::Lstm(cell) => cell
                .seq(&output)?
                .into_iter()
                .map(|state| state.h)
                .collect(),
            Cell::Gru(cell) => cell
                .seq(&output)?
                .into_iter()
                .map(|state| state.h)
                .collect(),
        };
        // Stack, not GRU::states_to_tensor's concatenation: [batch, time, hidden].
        output = Tensor::stack(&states, 1)?;
    }
    Ok(output)
}

fn select_step(sequence: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let (batch, length, hidden) = sequence.dims3()?;
    let length_i64 = i64::try_from(length).map_err(Error::wrap)?;
    let mut indices = indices.to_vec1::<i64>()?;
    if indices.len() != batch {
        return Err(Error::Msg("step indices must match batch size".into()));
    }
    for index in &mut indices {
        if *index < 0 {
            *index += length_i64;
        }
        if *index < 0 || *index >= length_i64 {
            return Err(Error::Msg("step index out of range".into()));
        }
    }
    let indices = Tensor::from_vec(indices, (batch, 1, 1), sequence.device())?
        .broadcast_as((batch, 1, hidden))?
        .contiguous()?;
    sequence.gather(&indices, 1)?.squeeze(1)
}

pub struct RecurrentFeatures {
    config: RecurrentConfig,
    parameters: IndexMap<String, Tensor>,
    raw: Vec<Cell>,
    _previous: Vec<Cell>,
    private: Vec<Cell>,
    raw_fc: Linear,
    private_fc: Linear,
    direction: [Linear; 2],
    output: [Linear; 2],
}
impl RecurrentFeatures {
    /// Creates live layers using the supplied Candle parameter source. Registration
    /// retains the unused `prev_rnn` bank for checkpoint compatibility.
    /// # Errors
    /// Returns invalid configuration or parameter shape/source/backend failures.
    pub fn new(config: RecurrentConfig, builder: VarBuilder<'_>) -> Result<Self> {
        if config.hidden_dim == 0 || config.layers == 0 {
            return Err(Error::Msg("hidden_dim and layers must be positive".into()));
        }
        let sources = config
            .hidden_dim
            .checked_mul(3)
            .ok_or_else(|| Error::Msg("source feature dimension overflow".into()))?;
        let mut parameters = Parameters::new(builder);
        let raw = recurrent_stack(&mut parameters, "raw_rnn", config)?;
        let previous = recurrent_stack(&mut parameters, "prev_rnn", config)?;
        let private = recurrent_stack(&mut parameters, "pri_rnn", config)?;
        let raw_fc = parameters.linear("raw_fc.0", config.data_dim, config.hidden_dim)?;
        let private_fc = parameters.linear("pri_fc.0", 2, config.hidden_dim)?;
        let direction = [
            parameters.linear("dire_fc.0", 2, config.hidden_dim)?,
            parameters.linear("dire_fc.2", config.hidden_dim, config.hidden_dim)?,
        ];
        let output = [
            parameters.linear("fc.0", sources, config.hidden_dim)?,
            parameters.linear("fc.2", config.hidden_dim, config.output_dim)?,
        ];
        Ok(Self {
            config,
            parameters: parameters.tensors,
            raw,
            _previous: previous,
            private,
            raw_fc,
            private_fc,
            direction,
            output,
        })
    }

    /// Public/private/direction features plus the full public recurrent sequence.
    /// # Errors
    /// Returns malformed observation, dtype, index or native computation failures.
    pub fn source_features(&self, obs: &RecurrentObservation) -> Result<([Tensor; 3], Tensor)> {
        let (batch, _, channels) = obs.data_processed.dims3()?;
        let device = obs.data_processed.device();
        let padding = Tensor::zeros((batch, 1, channels), DType::F32, device)?
            .to_dtype(obs.data_processed.dtype())?;
        let data = Tensor::cat(&[&padding, &obs.data_processed], 1)?;
        let position = obs
            .position_history
            .broadcast_div(&obs.target.unsqueeze(1)?)?
            .to_dtype(DType::F32)?;
        let steps = Tensor::arange(
            0_i64,
            i64::try_from(position.dim(1)?).map_err(Error::wrap)?,
            device,
        )?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?
        .broadcast_as(position.shape())?
        .broadcast_div(&obs.num_step.to_dtype(DType::F32)?.unsqueeze(1)?)?;
        let private_input = Tensor::stack(&[position, steps], 2)?;
        let public_sequence = sequence(&self.raw, &relu(&linear_forward(&self.raw_fc, &data)?)?)?;
        let public = select_step(&public_sequence, &obs.cur_tick)?;
        let private_sequence = sequence(
            &self.private,
            &relu(&linear_forward(&self.private_fc, &private_input)?)?,
        )?;
        let private = select_step(&private_sequence, &obs.cur_step)?;
        let acquiring = obs.acquiring.to_dtype(DType::F32)?;
        let direction_input = Tensor::stack(&[&acquiring, &acquiring.affine(-1., 1.)?], 1)?;
        let direction = relu(&linear_forward(
            &self.direction[1],
            &relu(&linear_forward(&self.direction[0], &direction_input)?)?,
        )?)?;
        Ok(([public, private, direction], public_sequence))
    }
}
impl CandleFeatureExtractor for RecurrentFeatures {
    fn output_dim(&self) -> usize {
        self.config.output_dim
    }
    fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.parameters
    }
    fn rebuild(
        &self,
        parameters: IndexMap<String, Tensor>,
    ) -> Result<Arc<dyn CandleFeatureExtractor>> {
        let builder = VarBuilder::from_tensors(
            parameters.into_iter().collect(),
            self.raw_fc.weight().dtype(),
            self.raw_fc.weight().device(),
        );
        Ok(Arc::new(Self::new(self.config, builder)?))
    }
    fn forward(&self, observation: &RecurrentObservation) -> Result<Tensor> {
        let (sources, _) = self.source_features(observation)?;
        relu(&linear_forward(
            &self.output[1],
            &relu(&linear_forward(
                &self.output[0],
                &Tensor::cat(&sources, 1)?,
            )?)?,
        )?)
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_network.rs"]
mod tests;
