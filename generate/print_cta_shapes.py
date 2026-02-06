from itertools import product

# vals = [2**i for i in range(3, 9)]  # 2^3=8 through 2^8=256
m_vals = [16, 32, 64, 128, 256]           # m starts at 16
n_vals = [8, 16, 32, 64, 128, 256]        # n starts at 8
k_vals = [16, 32, 64, 128, 256]           # k starts at 16

for m, n, k in product(m_vals, n_vals, k_vals):
    print(f"  - {m}x{n}x{k}")