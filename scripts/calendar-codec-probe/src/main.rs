use encoding_rs::GBK;

fn main() {
    let values: Vec<_> = (0_u16..=u16::MAX).map(|value| {
        GBK.decode_without_bom_handling_and_without_replacement(&value.to_be_bytes())
            .map(|text| text.chars().map(u32::from).collect::<Vec<_>>())
    }).collect();
    println!("{}", serde_json::to_string(&values).unwrap());
}
