//! Incremental replacement-field framing. Value formatting belongs to third-party engines.

pub(crate) struct Field<'a> {
    pub name: &'a str,
    pub conversion: Option<char>,
    pub spec: &'a str,
}

// Consume only this field: eagerly parsing subsequent fields changes custom-object effects.
pub(crate) fn field<'a>(remaining: &mut &'a str) -> Result<Field<'a>, String> {
    let input = *remaining;
    let mut chars = input.char_indices();
    let (end, delimiter) = loop {
        match chars.next() {
            Some((_, '[')) => {
                if !chars.any(|(_, ch)| ch == ']') {
                    return Err("missing ']' in checkpoint field".into());
                }
            }
            Some((index, ch @ ('}' | ':' | '!'))) => break (index, ch),
            Some((_, '{')) => return Err("unexpected '{' in checkpoint field".into()),
            Some(_) => {}
            None => return Err("unclosed checkpoint field".into()),
        }
    };
    let mut result = Field {
        name: &input[..end],
        conversion: None,
        spec: "",
    };
    let mut tail = &input[end + 1..];
    if delimiter == '}' {
        *remaining = tail;
        return Ok(result);
    }
    if delimiter == '!' {
        let conversion = tail.chars().next().ok_or("missing checkpoint conversion")?;
        result.conversion = Some(conversion);
        tail = &tail[conversion.len_utf8()..];
        if let Some(after) = tail.strip_prefix('}') {
            *remaining = after;
            return Ok(result);
        }
        tail = tail
            .strip_prefix(':')
            .ok_or("expected ':' after checkpoint conversion")?;
    }
    let mut depth = 1_usize;
    for (index, ch) in tail.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    result.spec = &tail[..index];
                    *remaining = &tail[index + 1..];
                    return Ok(result);
                }
            }
            _ => {}
        }
    }
    Err("unclosed checkpoint format specification".into())
}

// Get the next attribute/item only when the previous lookup has succeeded. In particular,
// a malformed later component must not suppress an earlier custom getattr/getitem call.
pub(crate) fn part<'a>(remaining: &mut &'a str) -> Result<(bool, &'a str), String> {
    let input = *remaining;
    let (attribute, name, tail) = if let Some(after) = input.strip_prefix('.') {
        let end = after.find(['.', '[']).unwrap_or(after.len());
        (true, &after[..end], &after[end..])
    } else if let Some(after) = input.strip_prefix('[') {
        let end = after.find(']').ok_or("missing ']' in checkpoint field")?;
        (false, &after[..end], &after[end + 1..])
    } else {
        return Err("only '.' or '[' may follow a checkpoint item".into());
    };
    if name.is_empty() {
        return Err("empty checkpoint attribute or index".into());
    }
    *remaining = tail;
    Ok((attribute, name))
}
