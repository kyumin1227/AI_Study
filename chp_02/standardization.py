from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

scaler = StandardScaler()

x = np.arange(10)

mean = x.mean()

values = [ item - mean for item in x ]
print(values, sum(values))
