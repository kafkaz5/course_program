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


def matix(a):
    rows_a = int(a[0])
    cols_a = int(a[1])
    new_a = []
    for n in range(rows_a):
        new_a.append(a[2+n*cols_a: 2+(n+1)*cols_a])
    return new_a, rows_a, cols_a


def dot_product(a, b):
    return [[x*y for x, [y] in zip(a[0], b)]]


def alpha_calculate(state, transition, emission, initial, previous_alpha):
    Bi = [[x[state]] for x in emission]
    if previous_alpha:
        current_state = matmult(previous_alpha, transition)
    else:
        current_state = initial
    alpha = dot_product(current_state, Bi)
    return alpha


def get_alpha(A, B, initial, obs_input):
    A, _, _ = matix(A)
    B, _, _ = matix(B)
    initial, _, _ = matix(initial)
    obs = [x for x in obs_input[1:]]
    alpha = None
    alpha_list = []
    for n in range(len(obs)):
        state = obs[n]
        alpha = alpha_calculate(state, A, B, initial, alpha)
        alpha_list.append(alpha[0])
    print(alpha_list)
    true_alpha = round(sum(alpha[0]), 6)
    return true_alpha


if __name__ == '__main__':
    A = list(map(float, input().split()))
    B = list(map(float, input().split()))
    initial = list(map(float, input().split()))
    obs_input = list(map(int, input().split()))
    final_alpha = get_alpha(A=A, B=B, initial=initial, obs_input=obs_input)
    print(final_alpha)
