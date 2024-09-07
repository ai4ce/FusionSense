import torch

class MaskedParameterModule(torch.nn.Module):
    def __init__(self, param_size, mask):
        super(MaskedParameterModule, self).__init__()
        
        # Initialize parameters
        self.params = torch.nn.Parameter(torch.randn(param_size))
        
        # Store the mask as a buffer so it won't be considered a parameter
        self.register_buffer('mask', mask)

    def forward(self, x):
        # Apply the mask: gradients only for the unmasked parts
        masked_params = self.params * self.mask + self.params.detach() * (1 - self.mask)
        return x * masked_params

# Example usage
param_size = (5, 5)
mask = torch.tensor([[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 0],
                     [0, 0, 1, 1, 0],
                     [0, 0, 0, 0, 1]], dtype=torch.float32)

model = MaskedParameterModule(param_size, mask)

# Forward pass with some input
input_data = torch.randn(5, 5)
output = model(input_data)

# Only the parameters corresponding to mask=1 will have gradients
output.sum().backward()

print(model.params.grad)  # Check gradients