use bullet_core::Price;

pub fn simple_moving_average(values: &[Price], length: usize) -> Option<Price> {
    let window = values.get(values.len().checked_sub(length)?..)?;
    Some(window.iter().sum::<Price>() / length as Price)
}

#[cfg(test)]
mod tests {
    use super::simple_moving_average;

    #[test]
    fn calculates_latest_window_average() {
        assert_eq!(simple_moving_average(&[10, 20, 30], 2), Some(25));
    }
}
