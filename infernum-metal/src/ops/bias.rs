//! BiasOps implementation for Metal — row-wise bias addition.

use infernum::backend::BiasOps;
use infernum::tensor::Tensor;
use infernum::Result;

use crate::tensor::MetalTensor;
use crate::MetalBackend;

impl BiasOps for MetalBackend {
    fn bias_add_inplace(input: &mut MetalTensor, bias: &MetalTensor) -> Result<()> {
        let shape = input.shape().to_vec();
        let cols = *shape.last().unwrap();
        let rows = input.numel() / cols;

        assert_eq!(
            bias.numel(),
            cols,
            "bias_add: bias len {} != cols {cols}",
            bias.numel()
        );

        let mut data = input.as_f32_slice().to_vec();
        let bias_data = bias.as_f32_slice();

        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                data[offset + c] += bias_data[c];
            }
        }

        *input = MetalTensor::from_f32(input.context(), &shape, &data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;
    use infernum::backend::{TensorDataOps, TensorFactory};

    fn ctx() -> MetalContext {
        MetalContext::new()
    }

    #[test]
    fn test_bias_add_1d() {
        let c = ctx();
        let mut input = MetalBackend::from_f32_slice(&c, &[3], &[1.0, 2.0, 3.0]).unwrap();
        let bias = MetalBackend::from_f32_slice(&c, &[3], &[0.1, 0.2, 0.3]).unwrap();
        MetalBackend::bias_add_inplace(&mut input, &bias).unwrap();
        let result = MetalBackend::to_f32_vec(&input).unwrap();
        for (a, b) in result.iter().zip([1.1, 2.2, 3.3].iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn test_bias_add_2d() {
        let c = ctx();
        let mut input =
            MetalBackend::from_f32_slice(&c, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let bias = MetalBackend::from_f32_slice(&c, &[3], &[10.0, 20.0, 30.0]).unwrap();
        MetalBackend::bias_add_inplace(&mut input, &bias).unwrap();
        let result = MetalBackend::to_f32_vec(&input).unwrap();
        assert_eq!(result, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }
}
