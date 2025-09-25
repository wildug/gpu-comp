import numpy as np
from tqdm import tqdm
import struct
import sys


np.random.seed(20250310)
w = 512
sigma = 1/np.sqrt(w)

n = 20
entropy = 7 # upper bound on data entropy

quantized_matrices = np.empty((n,w,w), dtype=np.int8)
w_deltas = np.zeros(n)
for i in tqdm(range(n)):
    matrix = np.random.randn(w, w)* sigma
    max_value = (2**(entropy-1))-1 # for signed integer maximum value is (2**(entropy-1))-1, neglecting -2**(entropy-1)
    w_delta =  np.abs(matrix).max()/max_value  # quantize matrices to int8 grid
    w_deltas[i] = w_delta # store as weight delta
    quantized_matrices[i,...] = np.round(np.round( matrix/ w_delta ) * 127 / max_value).astype(np.int8)

print(f'mins: {quantized_matrices.min(axis=(1, 2))}')
print(f'maxs: {quantized_matrices.max(axis=(1, 2))}')

vector = np.round(np.random.randn(w)).astype(np.int8)

def hash_int8_array(arr):
    """Simple hasher adopted from the "FxHash" rust crate.
    Not cryptographically secure, but _accidental_ collisions are very unlikely."""

    assert arr.dtype == np.int8

    hash = 0
    for byte in arr.astype(np.uint8).ravel():
        # rotate left by 5 bits
        hash = ((hash >> 27) | (hash << 5)) & 0xffff_ffff
        hash = ((hash ^ byte.item()) * 0x2722_0a95) & 0xffff_ffff

    return hash


deltas_a = np.zeros(n)
v_delta = 1
v_int8 = vector.copy()
for i,mat in enumerate(quantized_matrices):
    # .astype(np.int32) to not overflow while accumulating
    v_int32 = (mat.astype(float) @ v_int8.astype(float))
    
    # times w_deltas and v_delta to scale back to original values
    v_f32 = v_int32 * w_deltas[i] * v_delta

    # requantization to align again to  int8 
    v_delta = np.abs(v_f32).max() / 127 

    v_int8 = (v_f32 / v_delta)
    v_int8 = np.round(v_int8) # rounding for quantization

final_result = v_int8
print(f"Output vector:{v_int8}")
result_hash = hash_int8_array(np.array(final_result,dtype=np.int8))
print(f"Output hash: {result_hash}")

def serialize_file_header_raw(file, num_matrices): # omitting max word count
    print(f"Num_matrices: {num_matrices}")
    file.write(struct.pack('<L', num_matrices))

def serialize_vector_raw(file, vec):
    print(f"len_v: {len(vec)}")
    file.write(struct.pack('<L', len(vec)))
    # whatever you system does, we like little endian
    if sys.byteorder == 'little': 
        vec.astype(np.int8).tofile(file)
    else:
        vec.astype(np.int8).byteswap().tofile(file)

    file.write(b'\0' * (3 - (len(vec) + 3) % 4))

def serialize_raw_matrix(file, matrix):

    file.write(struct.pack(
        f'<LLf',
        *matrix.shape,1.))

    if sys.byteorder == 'little': 
        matrix.flatten(order="C").astype(np.int8).tofile(file) # C for C-Style saves matrix in row major format
    else:
        matrix.flatten(order="C").astype(np.int8).byteswap().tofile(file)

    if matrix.size % 2 != 0: 
        file.write(b'\0') # padding for if matrix does not have an even amount of 


with open(f'raw-matrices_{w}.bin', 'wb') as file:
    serialize_file_header_raw(file, len(quantized_matrices))
    serialize_vector_raw(file, vector)
    for matrix in quantized_matrices:
        serialize_raw_matrix(file, matrix)

print(f"wrote raw-matrices_{w}.bin")

def create_entropy_model(data, precision):
    min_value = data.min().item()
    max_value = data.max().item()

    if min_value == max_value:
        # Special case: all values are the same. Due to a limitation of ANS coding, we have
        # to create a dummy extra grid point with the smallest possible nonzero probability.
        grid_size = 2
        cdf = np.array([0, (1 << precision) - 1, 1 << precision], dtype=np.uint16)
        entropy = 0
        # To avoid loss of precision, we use `log1p(x) := log(1 + x)`, and thus:
        # cross_entropy = - log2((2^p - 1) / 2^p) = - log2(1 - 1 / 2^p) = - log1p(-1 / 2^p) / log(2)
        cross_entropy = - np.log1p(-1 / (1 << precision)) / np.log(2)
    else:
        grid_size = max_value - min_value + 1
        values, counts = np.unique(data, return_counts=True)

        # Sort descendingly by `counts`.
        order = np.argsort(counts)[::-1]
        values = values[order]
        counts = counts[order]

        scale = (1 << precision) / data.size
        probs = np.maximum(1, np.round(counts * scale).astype(np.uint16))

        excess = sum(probs).item() - (1 << precision)
        if excess > 0:
            # Shave off some probability mass from the most probable entries
            assert excess <= len(counts)
            while excess > 0:
                num_reducible_entries = (probs > 1).sum()
                num_reduce = min(excess, num_reducible_entries)
                assert num_reduce > 0
                probs[:num_reduce] -= 1
                excess -= num_reduce
            probs[:excess] -= 1
        elif excess < 0:
            # Spread some probability mass to the least probable entries.
            assert -excess <=len(counts)
            probs[excess:] += 1

        assert probs.sum() == 1 << precision

        entropy = np.log2(data.size) - (counts @ np.log2(counts)) / data.size
        cross_entropy = precision - (counts @ np.log2(probs)) / data.size

        padded_pmf = np.zeros(grid_size + 1, dtype=np.uint16)
        for value, prob in zip(values, probs):
            padded_pmf[value - min_value + 1] = prob

        cdf = np.cumsum(padded_pmf, dtype=np.uint16)
        assert cdf[0] == 0
        assert cdf[-1] == 1 << precision

    return min_value, grid_size, cdf, entropy, cross_entropy


def create_ppf(cdf, p_precision= 8):
    # handcrafet for 8 bit precision (weights && probs)
    # TODO generalize to arbitrary weight and prob precisions

    ppf = np.zeros((1<<p_precision), dtype=np.uint8) # ppf is of size 2**p_prec
    # for each possible probability save its corresponding value of the cdf
    for p in range ((1<< p_precision)):
        for r in range(len(cdf)-1,0,-1):
            if (cdf[r-1] & 0xFF).astype(np.uint8) <=p:
                ppf[p]= r-1
                break
            


    # ppf[1<<p_precision] = len(cdf) -1
        

    assert ppf[-1] == len(cdf)-2 # for 8-bit integer weights
    assert ppf[0] == 0
    return ppf
            

class AnsCoder:
    def __init__(self, precision, word_size,  compressed=[], head_size=2):
        # int head_size determines the size of the coder head as multiples of word_size bits
        self.precision = precision
        self.word_size = word_size
        self.word_mask = (1 << word_size) - 1
        self.quantile_mask = (1 << precision) - 1
        self.bulk = compressed.copy()
        self.head = 0
        self.head_size = head_size
        self.head_mask = (1 << (word_size * head_size)) - 1
        if compressed == []:
            self.bulk = [0] * (head_size-1) + [1]
        if len(compressed)>= self.head_size:
            for i in range(self.head_size):
                self.head = ((self.head << word_size) | int(self.bulk.pop())) & self.head_mask

    def push(self, symbol, cdf):
        prob = (cdf[symbol + 1] - cdf[symbol]).item()
        # this condition is true if the bits-back-trick would overflow the (head_size*word_size) bits, 
        # since z (arbitrary point between left cdf[symbol] and right cdf[symbol]) < prob
        if (self.head >> (self.head_size * self.word_size - self.precision)) >= prob:
            # i = 0
            while ((self.head >> (self.word_size)) != 0):
                self.bulk.append(self.head & self.word_mask)
                self.head >>= self.word_size

        # print(f'pushing {symbol} with prob {prob} and cdf {cdf[symbol]} onto {self.head}')
        z = self.head % prob + cdf[symbol].item()
        self.head = (((self.head // prob) << self.precision) | z) & self.head_mask

    def pop(self, cdf, ppf):
        z = self.head & self.quantile_mask
        self.head >>= self.precision
        # symbol = cdf.searchsorted(z, side='right').item() - 1
        symbol = ppf[z]
        prob = ((cdf[symbol + 1] - cdf[symbol]).item()) & 0xFF
        self.head = self.head * prob + (z - cdf[symbol].item()) & self.head_mask
        if ((self.head >> (self.word_size)) == 0) and len(self.bulk) != 0:
            # this while loops runs head_size -1 times if bulk is nice, bulk should be nice
            while ((self.head >> (self.head_size-1)*(self.word_size)) == 0) and len(self.bulk) != 0:
                self.head = ((self.head << self.word_size ) | int(self.bulk.pop())) & self.head_mask

        return symbol

    
    def interrupt(self):
        for i in range(self.head_size):
            self.bulk.append(self.head & self.word_mask)
            self.head >>= self.word_size
        self.head = 1 << self.word_size  # restores invariant
        return len(self.bulk)

class CompressedMatrix:
    def __init__(self, rows, cols, grid_spacing, cursors, min_value, cdf,ppf, payload):
        self.rows = rows
        self.cols = cols
        self.grid_spacing = grid_spacing
        self.cursors = cursors
        self.min_value = min_value
        self.cdf = (cdf & 0xFF).astype(np.uint8) # only take lowest 8 bit
        self.ppf= (ppf & 0xFF).astype(np.uint8)

        self.payload = payload # unpadded; will be padded to an even length upon serialization.
    
    def compressed_word_count(self):
        evend_payload_size = len(self.payload) + len(self.payload) % 2
        return (4 + self.rows) * 2 + (3 + len(self.cdf)) // 2 + evend_payload_size

    def serialize(self, file):
        payload_size = len(self.payload) 
        file.write(struct.pack(
            f'<LLf{self.rows}LLbB{len(self.cdf)}B',
            self.rows,
            self.cols,
            self.grid_spacing,
            *self.cursors,
            payload_size,
            self.min_value,
            len(self.cdf) - 1,
            *(self.cdf),
        ))

        if len(self.cdf) % 2 == 1:
            file.write(b'\0')
        
        # writing ppf 
        file.write(struct.pack(
            f'<256B',*(self.ppf)
        ))

        if sys.byteorder == 'little':
            self.payload.tofile(file)
        else:
            self.payload.byteswap().tofile(file)

        if len(self.payload) % 2 == 1:
            file.write(b'\0\0')
    
    @staticmethod
    def deserialize(file):
        """ Reads binary data from a file and reconstructs a CompressedMatrix object """
        # read number of rows and columns and grid_spacing
        rows, cols, grid_spacing = struct.unpack("<LLf", file.read(12))

        # get cursors using number of rows
        cursors = np.fromfile(file, dtype=np.uint32, count=rows) #check
        
        # get payload size
        payload_size, min_value, G = struct.unpack("<LbB",file.read(6))
        
        # Read the CDF values
        cdf_len = G + 1
        cdf_data = np.fromfile(file, dtype=np.uint8, count=cdf_len)
        if cdf_len % 2 == 1:
            file.seek(1,1)

        ppf = np.fromfile(file, dtype=np.uint8, count=256)
        # read payload
        payload = np.fromfile(file, dtype=np.uint16, count=payload_size)

        # skip 2 bytes if payload is an uneven number of 
        if payload_size % 2 ==1:
            file.seek(2,1)           

        # If system is big-endian, swap bytes
        if sys.byteorder != 'little':
            payload = payload.byteswap()

        return CompressedMatrix(rows, cols, grid_spacing, cursors, min_value, cdf_data,ppf, payload)

def encode_matrix(matrix, precision = 8):
    min_value, _grid_size, cdf, _entropy, _cross_entropy = create_entropy_model(matrix, precision)
    print(f"Entropy of matrix:       {_entropy:.4f} bits" )
    print(f"Cross-entropy of matrix: {_cross_entropy:.4f} bits (overhead of {(_cross_entropy/_entropy)-1:.4%} to Entropy)" )
    ppf = create_ppf(cdf)
    coder = AnsCoder(precision, 16, compressed=[0,0,0,1])
    back_cursors = np.empty(matrix.shape[0], dtype=np.uint32)

    for row in range(matrix.shape[0] - 1, -1, -1): # iterates in reverse order 
        for i,entry in enumerate(matrix[row, ::-1]): # iterates in reverse order due to stack semantics of ANS
            coder.push(entry.item() - min_value, cdf)
        back_cursors[row] = coder.interrupt()
    
    payload = np.array(coder.bulk[::-1], dtype=np.uint16)
    bits_per_weight = len(payload)*16/ (matrix.shape[0]*matrix.shape[1])
    print(f"Bits per Weight:         {bits_per_weight:.4f} bits (overhead of {(bits_per_weight/_cross_entropy)-1:.4%} to Cross-Entropy)")
    cursors = len(payload) - back_cursors
    return CompressedMatrix(matrix.shape[0], matrix.shape[1], 1.0, cursors, min_value, cdf,ppf, payload) # 1 here is for debugging purposes?

encoded_matrices = [encode_matrix(matrix) for matrix in tqdm(quantized_matrices)]

def serialize_file_header(file, num_matrices, result_hash, max_word_count):
    print(f"Num_matrices: {num_matrices}, Result_hash: {result_hash}, Max_word_count: {max_word_count}")
    file.write(struct.pack('<LLL', num_matrices, result_hash, max_word_count))

def serialize_vector(file, vec):
    print(f"len_v: {len(vec)}")
    file.write(struct.pack('<L', len(vec)))
    # whatever you system does, we like little endian
    if sys.byteorder == 'little': 
        vec.astype(np.int8).tofile(file)
    else:
        vec.astype(np.int8).byteswap().tofile(file)

    file.write(b'\0' * (3 - (len(vec) + 3) % 4))

max_word_count = max(m.compressed_word_count() for m in encoded_matrices)

with open(f'compressed_matrices_{w}.bin', 'wb') as file:
    serialize_file_header(file, len(quantized_matrices), result_hash, max_word_count)
    serialize_vector(file, vector)
    for matrix in encoded_matrices:
        matrix.serialize(file)

print(f"finished writing compressed_matrices_{w}.bin")