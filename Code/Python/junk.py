
import numpy as np

generator = np.random.default_rng(0)


covariance = generator.uniform(0.5, 1, (6, 6))
covariance = (covariance + covariance.T)/2
total_jacobian = generator.uniform(-1, 1, (4, 6))
total_jacobian[:, 3:6] = 0
jacobian1 = total_jacobian[0:2].copy()
jacobian2 = total_jacobian[2:4].copy()

innovations_covariance = total_jacobian @ covariance @ total_jacobian.T
gain_matrix = covariance @ total_jacobian.T @ np.linalg.inv(innovations_covariance)

innovations = generator.uniform(-1, 1, (4))

innovations_covariance1 = jacobian1 @ covariance @ jacobian1.T
gain_matrix1 = covariance @ jacobian1.T @ np.linalg.inv(innovations_covariance1)
innovations_covariance2 = jacobian2 @ covariance @ jacobian2.T
gain_matrix2 = covariance @ jacobian2.T @ np.linalg.inv(innovations_covariance2)
correction1 = gain_matrix1 @ innovations[0:2]
correction2 = gain_matrix2 @ innovations[2:4]
total_correction = correction1 + correction2

print(innovations_covariance)
print(innovations_covariance1)
print(innovations_covariance2)