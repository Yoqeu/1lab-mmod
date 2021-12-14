import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from model import *

print('THEORETICAL STATS: ')

A = np.array([
    [-2, 1, 0, 0, 0, 0],
    [2, -3.5, 1, 0, 1, 0],
    [0, 2, -3.5, 1, 0, 1],
    [0, 0, 2, -1.5, 0, 0],
    [0, 0.5, 0, 0, -3, 0],
    [0, 0, 0.5, 0.5, 2, -1],
    [1, 1, 1, 1, 1, 1]
])

b = np.array([0, 0, 0, 0, 0, 0, 1])


def f(x):
    y = np.dot(A, x) - b
    return np.dot(y, y)


cons = (
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]},
    {'type': 'ineq', 'fun': lambda x: x[2]},
    {'type': 'ineq', 'fun': lambda x: x[3]},
    {'type': 'ineq', 'fun': lambda x: x[4]},
    {'type': 'ineq', 'fun': lambda x: x[5]},
)
t_probs = opt.minimize(f, [0, 0, 0, 0, 0, 0], method='SLSQP', constraints=cons, options={'disp': False}).x

p1 = t_probs[0]
p2 = t_probs[1]
p3 = t_probs[2]
p4 = t_probs[3]
p5 = t_probs[4]
p6 = t_probs[5]

print('terminal is free, 0 in queue | p = ', p1)
print('terminal is busy, 0 in queue | p = ', p2)
print('terminal is busy, 1 in queue | p = ', p3)
print('terminal is busy, 2 in queue | p = ', p4)
print('terminal is broken, 1 in queue | p = ', p5)
print('terminal is broken, 2 in queue | p = ', p6)

t_p_denial = p4 + p6
t_avg_queue_length = 1 * (p3 + p5) + 2 * (p4 + p6)
t_avg_processing = p2 + p3 + p4
t_Q = 1 - t_p_denial
t_A = 2 * t_Q
t_avg_waiting_time = t_avg_queue_length / 2
t_avg_processing_time = t_Q / 1
t_avg_total_time = t_avg_processing_time + t_avg_waiting_time
print('Probability of denial: ', t_p_denial)
print('Average queue length: ', t_avg_queue_length)
print('Average processing: ', t_avg_processing)
print('Relative throughput: ', t_Q)
print('Absolute throрugput', t_A)
print('Average processing time: ', t_avg_processing_time)
print('Average waiting time: ', t_avg_waiting_time)
print('Average time in system: ', t_avg_total_time)

print('_______________________________')
print('EMPIRICAL STATS: ')

n = 1
m = 2
lambda_ = 2
mu = 1
nu = 0.5
r = 1
model = Model(m, lambda_, mu, nu, r, 1000)
model.start()
e_probs = model.show_stats()
p1 = e_probs[0]
p2 = e_probs[1]
p3 = e_probs[2]
p4 = e_probs[3]
p5 = e_probs[4]
p6 = e_probs[5]

print('terminal is free, 0 in queue | p = ', p1)
print('terminal is busy, 0 in queue | p = ', p2)
print('terminal is busy, 1 in queue | p = ', p3)
print('terminal is busy, 2 in queue | p = ', p4)
print('terminal is broken, 1 in queue | p = ', p5)
print('terminal is broken, 2 in queue | p = ', p6)

e_p_denial = p4 + p6
e_avg_queue_length = 1 * (p3 + p5) + 2 * (p4 + p6)
e_avg_processing = p2 + p3 + p4
e_Q = 1 - e_p_denial
e_A = 2 * e_Q
e_avg_waiting_time = e_avg_queue_length / 2
e_avg_processing_time = e_Q / 1
e_avg_total_time = e_avg_processing_time + e_avg_waiting_time
print('Probability of denial: ', e_p_denial)
print('Average queue length: ', e_avg_queue_length)
print('Average processing: ', e_avg_processing)
print('Relative throughput: ', e_Q)
print('Absolute throрugput', e_A)
print('Average processing time: ', e_avg_processing_time)
print('Average waiting time: ', e_avg_waiting_time)
print('Average time in system: ', e_avg_total_time)

EPS = 0.1

assert abs(e_p_denial - t_p_denial)
assert abs(e_Q - t_Q) < EPS
assert abs(e_A - t_A) < EPS
assert abs(e_avg_processing - t_avg_processing) < EPS
assert abs(e_avg_queue_length - t_avg_queue_length) < EPS
assert abs(e_avg_processing_time - t_avg_processing_time) < EPS
assert abs(e_avg_waiting_time - t_avg_waiting_time) < EPS
assert abs(e_avg_total_time - t_avg_total_time) < EPS

print('Model is working correctly')

_, ax = plt.subplots(1, 2)
ax[0].title.set_text(f'Empirical probabilities (n={n}, m={m}, lambda={lambda_}, mu={mu}, nu={nu}, r={r})')
ax[0].hist(list(np.arange(0, len(e_probs), 1)), weights=e_probs)
ax[1].title.set_text(f'Theoretical probabilities (n={n}, m={m}, lambda={lambda_}, mu={mu}, nu={nu}, r={r})')
ax[1].hist(list(np.arange(0, len(t_probs), 1)), weights=t_probs)
plt.show()

model.show_stationary_stats()
