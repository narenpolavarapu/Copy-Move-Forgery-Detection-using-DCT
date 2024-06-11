import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.fftpack import dct
import math
from scipy.stats import mode

# Number of packages and other parameters
num_packages = 64
offset_value = 4 
pixel_range = 256
package_size = pixel_range // num_packages

# Initialize packages dictionary
packages = {}
for i in range(1, num_packages + 1):
    package_name = f"PA{i}"
    pixel_range_start = (i - 1) * offset_value
    pixel_range_end = i * offset_value - 1
    package_info = {
        "Offset Value": offset_value,
        "Pixel Range Start": pixel_range_start,
        "Pixel Range End": pixel_range_end,
        "Blocks": [],  
        "Feature Vectors": [], 
        "BC": []
    }
    packages[package_name] = package_info

print("64 packages created")

# Read RGB image and convert to grayscale
rgb_image = imread('rgb.jpg')
gray_image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
print("Grayscale image converted")

# Resize the grayscale image to 64x64
gray_image_resized = gray_image[::int(gray_image.shape[0] / 64), ::int(gray_image.shape[1] / 64)]
print("Image resized to 64x64")

# Show the grayscale image after resizing
plt.figure(figsize=(8, 8))
plt.imshow(gray_image_resized, cmap='gray')
plt.title('Grayscale Image after 64x64 Conversion')
plt.axis('off')
plt.show()

# Extract blocks from the grayscale image
M, N = gray_image_resized.shape
b = 8
blocks = []
block_coordinates = []
for i in range(0, M - b + 1):
    for j in range(0, N - b + 1):
        block = gray_image_resized[i:i + b, j:j + b]
        block_coordinates.append((i, j))
        blocks.append(block)

print("Blocks extracted")

# Compute DCT coefficients for each block
dct_coefficient_matrices = []
for block in blocks:
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    dct_coefficient_matrices.append(dct_block)

print("DCT coefficients computed")

# Compute feature vectors for each block
sb = 4
epsilon = 1e-10 
sub_block = np.zeros((sb, sb))
all_feature_vectors = []
for dct_matrix in dct_coefficient_matrices:
    t_values = []
    positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
    for pos in positions:
        i, j = pos
        for r in range(i+sb):
            for s in range(j+sb):
                sub_block[i % sb][j % sb] = dct_matrix[i][j]
        sub_block_with_epsilon = sub_block + epsilon
        sum_of_powers = np.sum(sub_block_with_epsilon)
        tk = sum_of_powers / ((b/2)**2)
        t_values.append(tk)
    feature_vector = tuple(t_values)
    all_feature_vectors.append(feature_vector)

print("Feature vectors computed")

# Compute max pixel values for each block
max_pixel_values = []
for block in blocks:
    mode_value, _ = mode(block.flatten())
    max_pixel_values.append(mode_value)

print("Max pixel values computed")

# Assign blocks to packages based on max pixel values
for i in range(len(max_pixel_values)):
    package_index = math.ceil((max_pixel_values[i] / 4) + 1)
    package_index = max(1, min(package_index, num_packages))
    package_name = f"PA{package_index}"
    packages[package_name]['Blocks'].append(blocks[i])
    packages[package_name]['Feature Vectors'].append(all_feature_vectors[i])
    packages[package_name]['BC'].append(block_coordinates[i])

print("Blocks assigned to packages")

# Normalize feature vectors for all packages
def min_max_scaling(feature_vectors):
    min_val = min(feature_vectors)
    max_val = max(feature_vectors)
    if max_val == min_val:
        return feature_vectors
    scaled_features = [(fv - min_val) / (max_val - min_val) for fv in feature_vectors]
    return scaled_features

for package_name, package_details in packages.items():
    feature_vectors = package_details['Feature Vectors']
    normalized_feature_vectors = [min_max_scaling(fv) for fv in feature_vectors]
    package_details['Normalized Feature Vectors'] = normalized_feature_vectors

print("Feature vectors normalized")

# Create a grid of D and m values to test
D_values = [0.00003, 0.00004, 0.00002]
m_values = [5, 8, 1]

results = {}

for D in D_values:
    for m in m_values:
        detection_map = np.zeros((M, N), dtype=np.uint8)
        for package_name, package_details in packages.items():
            blocks = package_details['Blocks']
            normalized_feature_vectors = package_details['Normalized Feature Vectors']
            BC = package_details['BC']
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    a = b = 0
                    block_i = blocks[i]
                    block_j = blocks[j]
                    feature_vector_i = normalized_feature_vectors[i]
                    feature_vector_j = normalized_feature_vectors[j]
                    BC_i = BC[i]
                    BC_j = BC[j]
                    for r in range(4):
                        a = feature_vector_i[r] - feature_vector_j[r]
                        b = b + (a ** 2)
                    A =  b ** 0.5
                    x1, y1 = BC_i
                    x2, y2 = BC_j
                    O = (abs((x1 - x2) ** 2 + (y1 - y2) ** 2)) ** 0.5
                    if A < D and O > m:
                        for row in range(x1, x1 + 8):
                            for col in range(y1, y1 + 8):
                                detection_map[row][col] = 255
                        for row in range(x2, x2 + 8):
                            for col in range(y2, y2 + 8):
                                detection_map[row][col] = 255
        results[(D, m)] = detection_map

print("Comparison completed")

# Plot grayscale image and forgery detection maps
fig, axes = plt.subplots(len(D_values) + 1, len(m_values), figsize=(15, 15))

# Display grayscale image
axes[0, 0].imshow(gray_image_resized, cmap='gray')
axes[0, 0].set_title('Grayscale Image')
axes[0, 0].axis('off')

# Display forgery detection maps
for i, (D, m) in enumerate(results.keys()):
    ax = axes[(i + 1) // len(m_values), (i + 1) % len(m_values)]
    ax.imshow(results[(D, m)], cmap='gray')
    ax.set_title(f"D = {D}, m = {m}")
    ax.axis('off')

plt.tight_layout()
plt.show()
