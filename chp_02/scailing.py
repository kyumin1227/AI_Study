from sklearn.preprocessing import StandardScaler
import numpy as np

values1 = np.array((160, 170, 190, 180)).reshape(-1, 1)
values2 = np.array((400000000, 70000000, 200000000, 30000000)).reshape(-1, 1)

scaler1 = StandardScaler()
scaler2 = StandardScaler()

fit_values1 = scaler1.fit_transform(values1)
fit_values2 = scaler2.fit_transform(values2)

print(fit_values1)
print(fit_values2)
