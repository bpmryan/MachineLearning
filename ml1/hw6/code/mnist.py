import numpy as np
from matplotlib import pyplot as plt

PATH = '../media/t10k-images-idx3-ubyte'
RNG = np.random.default_rng(3)

with open(PATH, 'rb') as f:
    arr = bytearray(f.read())
    magic_number = int.from_bytes(arr[:4], byteorder='big')
    num_examples = int.from_bytes(arr[4:8], byteorder='big')
    rows = int.from_bytes(arr[8:12], byteorder='big')
    cols = int.from_bytes(arr[12:16], byteorder='big')
    data = np.array(arr[16:], dtype=np.int64).reshape(num_examples, rows, cols)

X = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

# K = 0.0001
ep, K = 3.5, 10
means = X[RNG.integers(0, X.shape[0], K)].astype(float)  # initial means

# <!-- remove from here
# fig = plt.figure(figsize=(12, 12))
# fig.suptitle('First 100 images arranged as a 10 by 10 table.')
# for i, cj in enumerate(X[:100]):
#     ax = fig.add_subplot(10, 10, i+1)
#     ax.imshow(cj.reshape(28, 28), cmap='grey')
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.tight_layout()
# plt.show()

# plt.title('Data Point 324: Label?')
# plt.imshow(data[324], cmap='gray')
# plt.tight_layout()
# plt.show()

iteration = 0
while True: 
    iteration += 1

    # assignment 
    distances = np.linalg.norm(X[:, np.newaxis] - means, axis=2)
    labels = np.argmin(distances, axis=1)

    # update
    new_means = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else means[k] for k in range(K)])

    # calculate convergence
    shift = np.mean(np.linalg.norm(new_means - means, axis=1))
    print(f"Iteration {iteration}: Shift = {shift:.4f}")

    means = new_means

    if shift < ep:
        print(f"Converged after {iteration} iterations.")
        break
# till here, and implement K-means -->

rows, columns = 2, 5
fig = plt.figure(figsize=(15, 7))
fig.suptitle('Converged centeroids.')
for i, cj in enumerate(means):
    ax = fig.add_subplot(rows, columns, i+1)
    ax.imshow(cj.reshape(28, 28), cmap='grey')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
