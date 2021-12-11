from model import *

params_set = [
    (2, 3, 2, 3, 1, 1000, 0.05),
    (3, 4, 3, 3, 2, 1000, 0.05),
    (5, 5, 1, 1, 1, 1000, 0.05)
]

for params in params_set:
    model = Model(*params)
    model.run()
    model.show_stats()
