//! NumPy-compatible widths for integer-containing native numeric data blocks.
use arrow_schema::DataType;

const SIGNED: [DataType; 4] = [
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
];
const UNSIGNED: [DataType; 4] = [
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
];

fn integer(dtype: &DataType) -> Option<(bool, usize)> {
    match dtype {
        DataType::Int8 => Some((true, 0)),
        DataType::Int16 => Some((true, 1)),
        DataType::Int32 => Some((true, 2)),
        DataType::Int64 => Some((true, 3)),
        DataType::UInt8 => Some((false, 0)),
        DataType::UInt16 => Some((false, 1)),
        DataType::UInt32 => Some((false, 2)),
        DataType::UInt64 => Some((false, 3)),
        _ => None,
    }
}

fn with_float(dtype: &DataType, rank: usize) -> Option<DataType> {
    match dtype {
        DataType::Float16 => Some(match rank {
            0 => DataType::Float16,
            1 => DataType::Float32,
            _ => DataType::Float64,
        }),
        DataType::Float32 => Some(if rank <= 1 {
            DataType::Float32
        } else {
            DataType::Float64
        }),
        DataType::Float64 => Some(DataType::Float64),
        _ => None,
    }
}

pub(super) fn common(left: &DataType, right: &DataType) -> Option<DataType> {
    if left == &DataType::Boolean && right.is_integer() {
        return Some(right.clone());
    }
    if right == &DataType::Boolean && left.is_integer() {
        return Some(left.clone());
    }
    match (integer(left), integer(right)) {
        (Some((ls, lr)), Some((rs, rr))) => {
            let result = if ls == rs {
                if ls {
                    SIGNED[lr.max(rr)].clone()
                } else {
                    UNSIGNED[lr.max(rr)].clone()
                }
            } else {
                let (signed, unsigned) = if ls { (lr, rr) } else { (rr, lr) };
                if signed > unsigned {
                    SIGNED[signed].clone()
                } else if unsigned < 3 {
                    SIGNED[unsigned + 1].clone()
                } else {
                    DataType::Float64
                }
            };
            Some(result)
        }
        (Some((_, rank)), None) => with_float(right, rank),
        (None, Some((_, rank))) => with_float(left, rank),
        (None, None) => None,
    }
}
