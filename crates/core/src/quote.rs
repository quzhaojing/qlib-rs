//! Arrow-backed equivalent of `qlib.backtest.high_performance_ds.PandasQuote`.

use std::{collections::BTreeMap, str::FromStr, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_cast::cast;
use arrow_schema::{ArrowError, DataType, Schema};
use thiserror::Error;

use crate::{
    AggregationArguments, BuiltInAggregation, FrequencyError, FrequencyUnit, LastValidAggregator,
    TimeRange, TimeSeriesIndex, TimeSeriesIndexOrder, TimeSeriesMethod, TimeSeriesResampleError,
    TimeSeriesSelectionError, resample_time_series, select_time_series,
    time_series_aggregation::prepare_time_series_groups,
};

/// String method accepted by Qlib's quote access boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuoteMethod {
    /// Return the selected time series as `SingleData`-compatible Float64 data.
    #[default]
    Selection,
    /// Invoke one of the supported Pandas string reductions.
    BuiltIn(BuiltInAggregation),
    /// Invoke Qlib's `ts_data_last` callable adapter.
    LastValid,
}

impl QuoteMethod {
    /// Return the exact optional method name used by Python's quote API.
    #[must_use]
    pub const fn python_name(self) -> Option<&'static str> {
        match self {
            Self::Selection => None,
            Self::BuiltIn(BuiltInAggregation::All) => Some("all"),
            Self::BuiltIn(BuiltInAggregation::Sum) => Some("sum"),
            Self::BuiltIn(BuiltInAggregation::Mean) => Some("mean"),
            Self::BuiltIn(BuiltInAggregation::Product) => Some("prod"),
            Self::BuiltIn(BuiltInAggregation::First) => Some("first"),
            Self::BuiltIn(BuiltInAggregation::Last) => Some("last"),
            Self::LastValid => Some("ts_data_last"),
        }
    }

    /// Convert Python's optional string method into the typed Rust union.
    ///
    /// # Errors
    ///
    /// Returns [`QuoteError::UnsupportedMethod`] for an unknown method name.
    pub fn from_python_name(method: Option<&str>) -> Result<Self, QuoteError> {
        let Some(method) = method else {
            return Ok(Self::Selection);
        };
        if method == "ts_data_last" {
            return Ok(Self::LastValid);
        }
        BuiltInAggregation::from_str(method)
            .map(Self::BuiltIn)
            .map_err(|_| QuoteError::UnsupportedMethod {
                method: method.to_owned(),
            })
    }
}

/// Result union corresponding to Python's scalar or `SingleData` return.
#[derive(Debug, Clone)]
pub enum QuoteData {
    /// One numeric or Boolean value, represented by a one-element Arrow array.
    Scalar(ArrayRef),
    /// Explicit datetime plus one Float64 value column.
    Series(RecordBatch),
}

/// Failures from constructing or querying an Arrow quote.
#[derive(Debug, Error)]
pub enum QuoteError {
    /// The textual Qlib frequency is malformed.
    #[error(transparent)]
    Frequency(#[from] FrequencyError),
    /// `NumpyQuote` only accepts minute and day units.
    #[error("unsupported NumpyQuote frequency {input}: unit {unit}")]
    UnsupportedNumpyFrequency {
        /// Original frequency spelling.
        input: String,
        /// Parsed but unsupported unit.
        unit: FrequencyUnit,
    },
    /// Python requires both timestamp bounds on the Numpy fast-path boundary.
    #[error("NumpyQuote requires the {bound} timestamp bound")]
    MissingRangeBound {
        /// Missing bound name.
        bound: &'static str,
    },
    /// Region validation is deferred until an existing stock is queried.
    #[error("unsupported NumpyQuote region: {region}")]
    UnsupportedRegion {
        /// Rejected region spelling.
        region: String,
    },
    /// A value column cannot reproduce `NumPy`'s Float64 matrix conversion.
    #[error("field {field} with Arrow type {data_type} cannot be converted to NumpyQuote Float64")]
    Float64Conversion {
        /// Rejected field.
        field: String,
        /// Original Arrow type.
        data_type: DataType,
        /// Arrow conversion failure.
        #[source]
        source: ArrowError,
    },
    /// The shared index validator rejected the input.
    #[error(transparent)]
    Selection(#[from] TimeSeriesSelectionError),
    /// Stock identifiers must match Qlib's string contract.
    #[error("instrument column {column} must be Utf8, not {data_type}")]
    InvalidInstrumentType {
        /// Instrument field name.
        column: String,
        /// Rejected Arrow type.
        data_type: DataType,
    },
    /// Python `dict` lookup would raise `KeyError` for this stock.
    #[error("stock {stock} is not present in the quote")]
    MissingStock {
        /// Requested identifier.
        stock: String,
    },
    /// Python `DataFrame` column lookup would raise `KeyError`.
    #[error("field {field} is not present for stock {stock}")]
    MissingField {
        /// Requested stock.
        stock: String,
        /// Requested field.
        field: String,
    },
    /// Duplicate `DataFrame` column labels do not form the required Series input.
    #[error("field {field} is ambiguous for stock {stock}")]
    AmbiguousField {
        /// Requested stock.
        stock: String,
        /// Duplicate field label.
        field: String,
    },
    /// A dynamic method is absent from the migrated method set.
    #[error("unsupported quote aggregation method: {method}")]
    UnsupportedMethod {
        /// Rejected name.
        method: String,
    },
    /// Selection or aggregation failed in the unified resampler.
    #[error(transparent)]
    Resample(#[from] TimeSeriesResampleError),
    /// `SingleData` cannot represent this field as its Float64 underlayer.
    #[error("field {field} with Arrow type {data_type} cannot be converted to SingleData")]
    UnsupportedSeriesType {
        /// Requested field.
        field: String,
        /// Rejected type.
        data_type: DataType,
    },
    /// `PandasQuote` rejects non-number callable/method return values.
    #[error("quote aggregation returned unsupported scalar Arrow type {data_type}")]
    UnsupportedScalarType {
        /// Rejected type.
        data_type: DataType,
    },
}

/// Object-safe quote boundary for in-memory, remote, or future plugin adapters.
pub trait Quote: Send + Sync {
    /// Return stock identifiers in stable Qlib/Pandas group-key order.
    fn get_all_stock(&self) -> Vec<String>;

    /// Select and optionally aggregate one stock field.
    ///
    /// # Errors
    ///
    /// Returns typed stock, field, selection, method, shape, or dtype errors.
    fn get_data(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError>;
}

/// In-memory Arrow implementation of Qlib's `PandasQuote` behavior.
#[derive(Debug, Clone)]
pub struct ArrowQuote {
    datetime: String,
    stocks: BTreeMap<String, RecordBatch>,
}

impl ArrowQuote {
    /// Partition a two-level quote batch by non-null string instrument.
    ///
    /// # Errors
    ///
    /// Returns an index validation or instrument dtype error.
    ///
    /// # Panics
    ///
    /// Panics only if Arrow violates sorting, slicing, or column-removal
    /// invariants established from the same validated batch.
    pub fn try_new(
        batch: &RecordBatch,
        instrument: impl Into<String>,
        datetime: impl Into<String>,
    ) -> Result<Self, QuoteError> {
        let instrument = instrument.into();
        let datetime = datetime.into();
        let index = TimeSeriesIndex::InstrumentDatetime {
            instrument: instrument.clone(),
            datetime: datetime.clone(),
            order: TimeSeriesIndexOrder::InstrumentDatetime,
        };
        let _ = select_time_series(batch, &index, TimeRange::default())?;
        let instrument_index = batch
            .schema_ref()
            .index_of(&instrument)
            .expect("the selection validator resolved the instrument field");
        if batch.column(instrument_index).data_type() != &DataType::Utf8 {
            return Err(QuoteError::InvalidInstrumentType {
                column: instrument,
                data_type: batch.column(instrument_index).data_type().clone(),
            });
        }

        let prepared = prepare_time_series_groups(batch, &index);
        let instrument_index = prepared
            .instrument_index
            .expect("a two-level quote index always has an instrument column");
        let instruments = prepared.batch.column(instrument_index);
        let instruments = instruments
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("the instrument type was validated as Utf8");
        let mut stocks = BTreeMap::new();
        for range in prepared.ranges {
            let stock = instruments.value(range.start).to_owned();
            let mut group = prepared.batch.slice(range.start, range.len());
            let _ = group.remove_column(instrument_index);
            stocks.insert(stock, group);
        }
        Ok(Self { datetime, stocks })
    }

    fn field_index(
        &self,
        stock: &str,
        batch: &RecordBatch,
        field: &str,
    ) -> Result<usize, QuoteError> {
        if field == self.datetime {
            return Err(QuoteError::MissingField {
                stock: stock.to_owned(),
                field: field.to_owned(),
            });
        }
        let mut matches = batch
            .schema_ref()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.name() == field)
            .map(|(index, _)| index);
        let Some(index) = matches.next() else {
            return Err(QuoteError::MissingField {
                stock: stock.to_owned(),
                field: field.to_owned(),
            });
        };
        if matches.next().is_some() {
            return Err(QuoteError::AmbiguousField {
                stock: stock.to_owned(),
                field: field.to_owned(),
            });
        }
        Ok(index)
    }

    pub(crate) fn contains_stock(&self, stock: &str) -> bool {
        self.stocks.contains_key(stock)
    }

    pub(crate) fn get_series(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
    ) -> Result<Option<RecordBatch>, QuoteError> {
        let Some(output) = self.resampled_data(stock, range, field, QuoteMethod::Selection)? else {
            return Ok(None);
        };
        selection_result(&output, &self.datetime, field).map(Some)
    }

    fn resampled_data(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<RecordBatch>, QuoteError> {
        let batch = self
            .stocks
            .get(stock)
            .ok_or_else(|| QuoteError::MissingStock {
                stock: stock.to_owned(),
            })?;
        let datetime_index = batch
            .schema_ref()
            .index_of(&self.datetime)
            .expect("constructor partitioning preserves the datetime field");
        let field_index = self.field_index(stock, batch, field)?;
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                batch.schema_ref().field(datetime_index).clone(),
                batch.schema_ref().field(field_index).clone(),
            ],
            batch.schema_ref().metadata().clone(),
        ));
        let input = RecordBatch::try_new(
            schema,
            vec![
                batch.column(datetime_index).clone(),
                batch.column(field_index).clone(),
            ],
        )
        .expect("two columns from one batch retain equal row counts");
        let index = TimeSeriesIndex::Datetime {
            datetime: self.datetime.clone(),
        };
        let time_series_method = match method {
            QuoteMethod::Selection => TimeSeriesMethod::Selection,
            QuoteMethod::BuiltIn(method) => TimeSeriesMethod::BuiltIn(method),
            QuoteMethod::LastValid => TimeSeriesMethod::Callable(&LastValidAggregator),
        };
        resample_time_series(
            &input,
            &index,
            range,
            time_series_method,
            &AggregationArguments::new(),
        )
        .map_err(Into::into)
    }
}

impl Quote for ArrowQuote {
    fn get_all_stock(&self) -> Vec<String> {
        self.stocks.keys().cloned().collect()
    }

    fn get_data(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError> {
        let output = self.resampled_data(stock, range, field, method)?;
        let Some(output) = output else {
            return Ok(None);
        };
        if method == QuoteMethod::Selection {
            return selection_result(&output, &self.datetime, field)
                .map(QuoteData::Series)
                .map(Some);
        }
        scalar_result(&output).map(Some)
    }
}

fn selection_result(
    output: &RecordBatch,
    datetime: &str,
    field: &str,
) -> Result<RecordBatch, QuoteError> {
    let values = output
        .column_by_name(field)
        .expect("the resampler preserves the selected value field");
    if !values.data_type().is_numeric() && values.data_type() != &DataType::Boolean {
        return Err(QuoteError::UnsupportedSeriesType {
            field: field.to_owned(),
            data_type: values.data_type().clone(),
        });
    }
    let values = cast(values.as_ref(), &DataType::Float64)
        .expect("Arrow numeric and Boolean arrays cast to Float64");
    let datetime_values = output
        .column_by_name(datetime)
        .expect("the resampler preserves the datetime field")
        .clone();
    let fields = vec![
        output
            .schema_ref()
            .field_with_name(datetime)
            .expect("the datetime field belongs to this schema")
            .clone(),
        output
            .schema_ref()
            .field_with_name(field)
            .expect("the selected field belongs to this schema")
            .clone()
            .with_data_type(DataType::Float64)
            .with_nullable(true),
    ];
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        output.schema_ref().metadata().clone(),
    ));
    Ok(RecordBatch::try_new(schema, vec![datetime_values, values])
        .expect("the cast value column preserves its selected row count"))
}

fn scalar_result(output: &RecordBatch) -> Result<QuoteData, QuoteError> {
    let value = output.column(0);
    if !matches!(
        value.data_type(),
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    ) {
        return Err(QuoteError::UnsupportedScalarType {
            data_type: value.data_type().clone(),
        });
    }
    Ok(QuoteData::Scalar(value.clone()))
}
