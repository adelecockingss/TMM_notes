# TMM Lab 03 - Q9: the lecture's two-origin 'scissors' network
# Python translation of networkExampleLecture2021.m, with the SO-in-link-
# variables exercise completed. MATLAB version: q9_scissors.m
import numpy as np
from assign_utils import qp_small

A = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]], float)
a = np.array([2, 2, 1, 4, 4], float); b = np.ones(5)
M = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], float)            # OD-route incidence
d = np.array([1.0, 1.0])
B = np.array([[-1, 0, 0, -1, 0], [0, -1, 0, 0, -1], [1, 1, -1, 0, 0], [0, 0, 1, 1, 1]], float)
s = np.array([-d[0], -d[1], 0, d.sum()])

yUE, _ = qp_small(A.T @ np.diag(b) @ A, A.T @ a, M, d)        # UE, routes
ySO, _ = qp_small(2 * A.T @ np.diag(b) @ A, A.T @ a, M, d)    # SO, routes
xUE, _ = qp_small(np.diag(b), a, B, s)                         # UE, links
xSO, _ = qp_small(2 * np.diag(b), a, B, s)                     # SO, links (exercise)

print("UE link flows from routes:", (A @ yUE).round(4), " from links:", xUE.round(4))
print("SO link flows from routes:", (A @ ySO).round(4), " from links:", xSO.round(4))
cost = lambda x: x @ (a + b * x)
print("route costs at UE:", (A.T @ (a + b * xUE)).round(4))
print(f"system cost UE {cost(xUE):.4f}, SO {cost(xSO):.4f}, PoA {cost(xUE) / cost(xSO):.4f}")
# 4 routes, 5 links: here route flows CAN be recovered (A has full column rank)
print("route flows from link flows:", np.linalg.lstsq(A, xUE, rcond=None)[0].round(4))
