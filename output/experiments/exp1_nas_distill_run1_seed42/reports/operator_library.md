# NAS Operator Library

- `std_k3`: standard 1D convolution residual op, kernel_size=3
- `std_k5`: standard 1D convolution residual op, kernel_size=5
- `std_k7`: standard 1D convolution residual op, kernel_size=7
- `sep_k3`: depthwise-separable 1D convolution residual op, kernel_size=3
- `sep_k5`: depthwise-separable 1D convolution residual op, kernel_size=5
- `sep_k7`: depthwise-separable 1D convolution residual op, kernel_size=7
- `sep_k9`: depthwise-separable 1D convolution residual op, kernel_size=9
- `dil_k3`: dilated 1D convolution residual op, kernel_size=3, dilation_factor=2
- `dil_k5`: dilated 1D convolution residual op, kernel_size=5, dilation_factor=2
- `dil_k9`: dilated 1D convolution residual op, kernel_size=9, dilation_factor=2
- `pw_bottleneck_k3`: pointwise bottleneck residual op with kernel_size=3 temporal mixing
- `skip`: identity residual op

Constraints:

- `count(skip) <= 2`
- `skip` forces `use_se=false` because the current skip op is identity.
