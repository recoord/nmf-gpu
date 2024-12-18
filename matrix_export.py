import numpy as np
import struct

# # target X matrix
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
# # initial W matrix
# W = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
# # initial H matrix
# H = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

X = np.random.rand(4096, 3500).astype(np.float32)
W = np.random.rand(4096, 128).astype(np.float32)
H = np.random.rand(128, 3500).astype(np.float32)

path = "./"

def write_matrix_to_file(matrix, filename):
    with open(filename, "wb") as f:
        f.write(struct.pack("ii", *matrix.shape))
        f.write(matrix.tobytes())
        print(f"wrote file with {matrix.size} elements")

write_matrix_to_file(X, f"{path}X.bin")
write_matrix_to_file(W, f"{path}W.bin")
write_matrix_to_file(H, f"{path}H.bin")
