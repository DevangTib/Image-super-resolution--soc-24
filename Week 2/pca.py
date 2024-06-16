import numpy as np
import matplotlib.pyplot as plt

def data1():
    x = [np.random.rand() for i in range(1000)]
    y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

def data2():
    x = [np.random.rand() for i in range(1000)]
    y = [(x[i])**2 + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

# Standardisation of Data
def std_data(nparray):
    standardized_data = (nparray - nparray.mean(axis=1)[:, None]) / nparray.std(axis=1)[:, None]
    return standardized_data

def DimReduction(arr):
    data_set = np.array(arr)
    std_data_set = std_data(data_set)
    
    # Step 1: Compute the covariance matrix
    cov_matrix = np.cov(std_data_set)
    
    # Step 2: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 3: Sort the eigenvalues and eigenvectors
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    # Step 4: Use the principal component to find the best fit line
    principal_component = sorted_eigenvectors[:, 0]
    m = principal_component[1] / principal_component[0]
    c = np.mean(data_set[1]) - m * np.mean(data_set[0])
    
    # Displaying the result using matplotlib
    plt.scatter(data_set[0], data_set[1], color = "red")
    plt.plot(data_set[0], data_set[0]*m + c)
    print("Slope =", m, "Intercept =", c)
    plt.title("Best Fit Line")
    plt.show()

DimReduction(data1())
DimReduction(data2())
