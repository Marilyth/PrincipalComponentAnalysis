import numpy as np
import numpy.linalg
import pandas
import time
import csv

def scale_x(X_data):
    print("Normalizing data...")
    X_data_copy = X_data.copy()

    for f in range(X_data_copy.shape[1]):
        mean = np.mean(X_data_copy[:, f])
        std = np.std(X_data_copy[:, f])

        X_data_copy[:, f] -= mean
        if std != 0:
            X_data_copy[:, f] /= std
        else:
            X_data_copy[:, f] = 0

    return X_data_copy

def PCA(X_scaled):
    print("Performing PCA...")
    S = np.cov(X_scaled.T)
    eig_vals, eig_vecs = numpy.linalg.eig(S)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs_sorted = eig_vecs[:, idx]
    eig_vals_sorted = eig_vals[idx]

    PVEs = [eig_val / sum(eig_vals_sorted) for eig_val in eig_vals_sorted]

    return eig_vals_sorted, eig_vecs_sorted, PVEs

def reduce(X_data, eig_vecs, dimensions=2, transform=True):
    if transform:
        print("Transforming...")
    else:
        print("Calculating column importance...")

    dimension_vec = []
    importance = np.zeros(eig_vecs[:, 0].shape)

    for dimension in range(dimensions):
        importance += abs(eig_vecs[:, dimension])

        if transform:
            dimension_vec.append([np.dot(eig_vecs[:, dimension], X_data.T)])

    return np.array(dimension_vec), importance

def read_file(file_name="circ_3_14.dat", features=None, n=None):
    print("Reading data...")
    if features is None:
        return pandas.read_csv(file_name, delim_whitespace=True, usecols=range(0, 14), nrows=n).to_numpy().astype(np.float)
    else:
        return pandas.read_csv(file_name, delim_whitespace=True, usecols=range(0, features), nrows=n).to_numpy().astype(np.float)

def write_file(dimensions, file_name="reduced.dat"):
    print(f"Saving to {file_name}")
    numpy.savetxt(file_name, dimensions[:, 0, :].reshape((dimensions.shape[2], dimensions.shape[0])), delimiter=" ", newline="\n")

start = time.time()
X = read_file(input("Enter the file you want to use the PCA on: "), int(input("Enter the number of features (columns) you want to use for the PCA: ")))

scaled_x = scale_x(X)
scaled_x = np.nan_to_num(scaled_x)
vals, vecs, pve = PCA(scaled_x)

print("")
dimension_recommended = 0
for i, ve in enumerate(pve):
    print(i+1, "Principal Components contribute to: ", round(sum(pve[:i]) + ve, 2), " total variance")
    if sum(pve[:i]) + ve > 0.7 and dimension_recommended == 0:
        dimension_recommended = i+1
print("")

dimensions, importance = reduce(scaled_x, vecs, dimension_recommended, False)

print("Importance of each feature (higher = more important):")
relative_importance = importance / max(importance)
for i in range(len(relative_importance)):
    print("Column ", i+1, ": ", round(relative_importance[i], 2))

print("\nRecommending at least", dimension_recommended, " principal components.")
if input("Would you like to save the principal components? (y/n): ") == "y":
    dimensions, importance = reduce(scaled_x, vecs, int(input("How many principal components would you like to save?: ")))
    write_file(dimensions)
