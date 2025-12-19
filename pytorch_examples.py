import torch

tensor0d = torch.tensor(1) # 0D tensor, scalar from a python int
tensor1d = torch.tensor([1, 2, 3]) # 1D tensor (vector) from a python list
tensor2d = torch.tensor([[1, 2],
                         [3, 4]]) # 2D tensor from a nested python list
tensor3d = torch.tensor([[[1, 2], [3, 4]],
                        [[5, 6], [7, 8]]]) # 3D tensor from a nested python list

print(tensor1d.dtype) # torch.int64

# float Tensors
floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype) # torch.float32

floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)

tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
print(tensor2d)
# print the shape of the tensor
print(tensor2d.shape) # torch.Size([2, 3]) meaning the tensor has 2 rows and 3 columns

# we can reshape the tensor to 3x2 
print(tensor2d.reshape(3, 2)) # this does not care about the size it will repeat the data if needed
# a more common command for reshaping is the .view()
print(tensor2d.view(3, 2)) # must be the same size to work

# flip a tensor across its diagonal
print(tensor2d.T)  #tensor([[1, 4],
                   #        [2, 5],
                   #        [3, 6]])

# multiplying 2 matrices in pytorch
print(tensor2d.matmul(tensor2d.T))  # tensor([[14, 32],
                                    #        [32, 77]])


"""
  z = x1 * w1 + b  # z = 1.1 * 2.2 + 0.0 = 2.42

  This is the standard neural network formula:
  z = (input x weight) + bias

  Then:
  a = torch.sigmoid(2.42)  # ≈ 0.918 (probability between 0 and 1)
  loss = F.binary_cross_entropy(0.918, 1.0)  # How far 0.918 is from target 1.0

  This is modeling a single neuron doing binary classification:
  1. Take input (1.1)
  2. Multiply by weight (2.2)
  3. Add bias (0.0)
  4. Apply sigmoid activation
  5. Compare prediction to true label (1.0)
  6. Calculate loss to see how wrong we were
"""
import torch.nn.functional as F

y = torch.tensor([1.0]) # True label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2]) # weight parameter
b = torch.tensor([0.0])  # bias unit
z = x1 * w1 + b # net input 
a = torch.sigmoid(z) # Activation and output
loss = F.binary_cross_entropy(a, y)
                                   