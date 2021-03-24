from math import log


def matix(a):
    rows_a = int(a[0])
    cols_a = int(a[1])
    new_a = []
    for n in range(rows_a):
        new_a.append(a[2+n*cols_a: 2+(n+1)*cols_a])
    return new_a, rows_a, cols_a


def matmult (A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    C = [[0.0 for row in range(cols_B)] for col in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C


def dot_product(a, b):
    return [[x*y for x, [y] in zip(a[0], b)]]


def alpha_calculate(state, transition, emission, initial, previous_alpha):
    Bo = [[x[state]] for x in emission]
    if previous_alpha:
        current_state = matmult(previous_alpha, transition)
    else:
        current_state = initial
    alpha = dot_product(current_state, Bo)
    return alpha


def scale_alpha(alpha, t):
    ct = sum(alpha[0])
    ct = 1 / ct
    for i in range(len(alpha[0])):
        alpha[0][i] = ct * alpha[0][i]
    return alpha, ct


def get_alpha(A, B, initial, obs):
    alpha = None
    alpha_list = []
    c = []
    for n in range(len(obs)):
        state = obs[n]
        alpha = alpha_calculate(state, A, B, initial, alpha)
        alpha, ct = scale_alpha(alpha, n)
        alpha_list.append(alpha[0])
        c.append([ct])
    return alpha_list, c


def betaj_mult(future_beta, Bo1, Ai, j):
    res = future_beta[j] * Bo1[j][0] * Ai[j]
    return res


def beta_calculate(obs1, transition, emission, future_beta, c):
    Bo1 = [[x[obs1]] for x in emission]
    beta = [0.0 for n in range(len(transition))]
    # beta_ij = [0.0 for x in range(len(transition[0]))]
    for i in range(len(transition)):
        for j in range(len(transition[0])):
            beta[i] = beta[i] + betaj_mult(future_beta, Bo1, transition[i], j)
        beta[i] = beta[i] * c
    return beta


def get_beta(A, B, obs, c):
    T = len(obs)
    beta_list = [[0.0 for row in range(len(A))]for col in range(len(obs))]
    beta_list[T - 1] = [1.0 * c[T - 1][0]] * len(A)
    for t in range(len(obs) - 2, -1, -1):
        obs1 = obs[t + 1]
        beta_list[t] = beta_calculate(obs1, A, B, beta_list[t+1], c[t][0])
        # print(beta_list[t])
    return beta_list


def di_gamma(alpha_list, A, B, beta_list, cols_A, rows_A, obs):
    gammaij = [[[0.0 for col in range(cols_A)] for row in range(rows_A)] for t in range(len(obs))]
    for t in range(len(obs)-1):
        obs1 = obs[t+1]
        Bo1 = [[x[obs1]] for x in B]
        for i in range(rows_A):
            for j in range(cols_A):
                gammaij[t][i][j] = alpha_list[t][i] * A[i][j] * Bo1[j][0] * beta_list[t+1][j]
    return gammaij


def Gamma(gammaij, obs, rows_A, alpha_list):
    gamma = [[0.0 for row in range(rows_A)] for t in range(len(obs))]
    for t in range(len(obs)-1):
        for i in range(rows_A):
            gamma[t][i] = sum(gammaij[t][i])
    for i in range(rows_A):
        gamma[len(obs)-1][i] = alpha_list[len(obs)-1][i]
    return gamma


def re_estimate(A, B, initial, gamma, gammaij, obs):
    for i in range(len(A[0])):
        initial[0][i] = gamma[0][i]
        denom = 0
        for t in range(len(obs)-1):
            denom = denom + gamma[t][i]
        for j in range(len(A)):
            numer_A = 0
            for t in range(len(obs) - 1):
                numer_A = numer_A +gammaij[t][i][j]
            A[i][j] = numer_A / denom
        for o in range(len(B[0])):
            numer_B = 0
            for t in range(len(obs)):
                if obs[t] == o:
                    numer_B = numer_B +gamma[t][i]
            B[i][o] = numer_B / denom
    return A, B, initial


def compute_log(c, obs):
    logprob = 0
    for t in range(len(obs)):
        logprob = logprob + log(c[t][0])
    logprob = -logprob
    return logprob


def iteration(A, B, initial, obs, cols_A, rows_A):
    iters = 0
    max_iter = 400
    logprob = -100000
    oldlogprob = -10000000
    while logprob - oldlogprob > 1e-4 and iters < max_iter:
        oldlogprob = logprob
        alpha_list, c = get_alpha(A, B, initial, obs)
        beta_list = get_beta(A, B, obs, c)
        gammaij = di_gamma(alpha_list, A, B, beta_list, cols_A, rows_A, obs)
        gamma = Gamma(gammaij, obs, rows_A, alpha_list)
        logprob = compute_log(c, obs)
        A, B, initial = re_estimate(A, B, initial, gamma, gammaij, obs)
        iters += 1
    output(A)
    output(B)
    output(initial)
    print(iters)


def output(A):
    # print(A)
    col = len(A[0])
    row = len(A)
    A = [round(A[n][m], 6) for n in range(row) for m in range(col)]
    A.insert(0, row)
    A.insert(1, col)
    A = " ".join(str(x) for x in A)
    print(A)


def initialize(A, B, initial, obs_input):
    A, _, _ = matix(A)
    B, _, _ = matix(B)
    initial, _, _ = matix(initial)
    obs = [x for x in obs_input[1:]]


if __name__ == '__main__':
    A = list(map(float, input().split()))
    B = list(map(float, input().split()))
    initial = list(map(float, input().split()))
    obs_input = list(map(int, input().split()))
    A, rows_A, cols_A = matix(A)
    B, _, _ = matix(B)
    initial, _, _ = matix(initial)
    obs = [x for x in obs_input[1:]]
    iteration(A, B, initial, obs, cols_A, rows_A)


