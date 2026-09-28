"""CasADi dynamics shared by the angle and complex-z collocation methods."""


def symbolic_accel(model, cs, ss, ws, u, v):
    """Return [cart acceleration, absolute link angular accelerations]."""
    import casadi as ca

    n, A, D, l = model.n, model.A, model.D, model.l
    Mrows = [[0]*(n+1) for _ in range(n+1)]
    Mrows[0][0] = model.M + model.S_total[0]
    for i in range(1, n+1):
        Mrows[0][i] = Mrows[i][0] = A[i-1]*l[i-1]*cs[i-1]
        Mrows[i][i] = D[i-1]*l[i-1]**2 + model.rod_inertia[i-1]
        for j in range(i+1, n+1):
            rc = cs[i-1]*cs[j-1] + ss[i-1]*ss[j-1]
            Mrows[i][j] = Mrows[j][i] = A[j-1]*l[i-1]*l[j-1]*rc
    Mm = ca.vertcat(*[ca.horzcat(*row) for row in Mrows])

    F = [0]*(n+1)
    F[0] = (u - model.cart_damping*v
            - model.cart_coulomb*ca.tanh(v/model.coulomb_velocity)) + sum(
        A[i]*l[i]*ws[i]**2*ss[i] for i in range(n))
    for i in range(1, n+1):
        Fi = A[i-1]*model.g*l[i-1]*ss[i-1]
        for j in range(i+1, n+1):
            rs = ss[i-1]*cs[j-1] - cs[i-1]*ss[j-1]
            Fi -= A[j-1]*l[i-1]*l[j-1]*ws[j-1]**2*rs
        for j in range(1, i):
            rs = ss[j-1]*cs[i-1] - cs[j-1]*ss[i-1]
            Fi += A[i-1]*l[j-1]*l[i-1]*ws[j-1]**2*rs
        Fi -= model.joint_damping[i-1]*(ws[i-1] - (ws[i-2] if i > 1 else 0))
        if i < n:
            Fi += model.joint_damping[i]*(ws[i] - ws[i-1])
        F[i] = Fi
    return ca.solve(Mm, ca.vertcat(*F))
