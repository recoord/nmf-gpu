import numpy as np
import struct

np.random.seed(0)
X = np.random.rand(4096, 350).astype(np.float32)
W = np.random.rand(4096, 128).astype(np.float32)
H = np.random.rand(128, 350).astype(np.float32)

def write_matrix_to_file(matrix, filename):
    with open(filename, "wb") as f:
        f.write(struct.pack("ii", *matrix.shape))
        f.write(matrix.tobytes())
        print(f"wrote file with {matrix.size} elements")

write_matrix_to_file(X, "X.bin")
write_matrix_to_file(W, "W.bin")
write_matrix_to_file(H, "H.bin")
