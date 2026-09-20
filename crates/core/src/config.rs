//! Typed configuration contracts derived from `qlib.config`.

use serde::{Deserialize, Serialize};

use crate::Region;

/// Trading defaults selected by a Qlib market region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionConfig {
    /// Minimum number of units traded in one lot.
    pub trade_unit: u32,
    /// Default daily price-limit threshold, or no limit for the market.
    pub limit_threshold: Option<f64>,
    /// Default Qlib feature used as the execution price.
    pub deal_price: String,
}

impl Region {
    /// Return an owned copy of the defaults applied by `QlibConfig.set_region`.
    #[must_use]
    pub fn defaults(self) -> RegionConfig {
        let (trade_unit, limit_threshold) = match self {
            Self::Cn => (100, Some(0.095)),
            Self::Us => (1, None),
            Self::Tw => (1_000, Some(0.1)),
        };

        RegionConfig {
            trade_unit,
            limit_threshold,
            deal_price: "close".to_owned(),
        }
    }
}
