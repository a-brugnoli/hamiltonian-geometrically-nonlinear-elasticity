import sympy as sp

# Define symbolic variables
dt,omega = sp.symbols('dt omega', real=True)
beta, gamma = 0, 0.5
# beta, gamma = sp.symbols('beta gamma', real=True)

# Declare two 3x3 symbolic matrices
H1 = sp.Matrix([
    [1, 0, -dt**2*beta],
    [0, 1, -dt*gamma],
    [omega**2, 0, 1]
])

H0 = sp.Matrix([
    [1, dt, (1/2-beta)*dt**2],
    [0, 1, (1-gamma)*dt],
    [0, 0, 0]
])

# State-space matrix
A = sp.simplify(H1.inv() * H0)
eigs_M = list(A.eigenvals().keys())

inv_H1_22 = 1/H1[2, 2]

# Reduced matrices
H1_red = H1[0:2, 0:2] - H1[0:2, 2]*inv_H1_22*H1[2, 0:2]
H0_red = H0[0:2, 0:2] - H0[0:2, 2]*inv_H1_22*H1[2, 0:2]

# State-space matrix
A_red = sp.simplify(H1_red.inv() * H0_red)
eigs_A_red = list(A_red.eigenvals().keys())


# Compute the reduced matrices analytically

H1_red_analytic = sp.Matrix([
    [1+dt**2*beta*omega**2, 0],
    [+dt*gamma*omega**2, 1],
])

H0_red_analytic = sp.Matrix([
    [1-(1/2-beta)*dt**2*omega**2, dt],
    [-(1-gamma)*dt*omega**2,      1],
])

# sp.pprint(H1_red_analytic-H1_red)
# sp.pprint(sp.simplify(H0_red_analytic-H0_red))

stability = [sp.simplify(sp.Abs(ev) < 1) for ev in eigs_A_red]


for ii in range(2):
    print(f"Reduced eigenvalue {ii}:")
    sp.pprint(eigs_A_red[ii])
    print(f"Is |λ| < 1? -> {stability[ii]}")
