def matix(a):
    rows_a = int(a[0])
    cols_a = int(a[1])
    new_a = []
    for n in range(rows_a):
        new_a.append(a[2+n*cols_a: 2+(n+1)*cols_a])
    return new_a, rows_a, cols_a


def dot_product(a, b):
    res = [[x * y for x, [y] in zip(a[0], b)]]
    res = [[round(value, 6) for value in res[0]]]
    return res


def delta_calculate(state, transition, emission, initial, previous_delta):
    Bi = [[x[state]] for x in emission]
    delta = []
    idex = []
    if previous_delta:
        for n in range(len(transition)):
            Ai = [[x[n]] for x in transition]
            probabilities = dot_product(previous_delta, Ai)
            for index in range(len(probabilities[0])):
                probabilities[0][index] = Bi[n][0] * probabilities[0][index]
                probabilities = [[round(value, 6) for value in probabilities[0]]]
            # print(probabilities[0])
            delta.append(max(probabilities[0]))
            idex.append(probabilities[0].index(max(probabilities[0])))
        delta = [delta]
    else:
        delta = dot_product(initial, Bi)
    # print(previous_delta)
    return delta, idex


def get_delta_state(A, B, initial, obs_input):
    A, _, _ = matix(A)
    B, _, _ = matix(B)
    initial, _, _ = matix(initial)
    obs = [x for x in obs_input[1:]]
    delta = None
    hidden_state = []
    delta_list = []
    delta_idex = []
    for n in range(len(obs)):
        state = obs[n]
        delta, idex = delta_calculate(state, A, B, initial, delta)
        delta_list.append(delta[0])
        delta_idex.append(idex)
        print(delta_idex)
        max_last_delta = max(delta[0])
        last_state = (delta[0]).index(max_last_delta)
        hidden_state.append(last_state)
    return hidden_state


if __name__ == '__main__':
    A = list(map(float, input().split()))
    B = list(map(float, input().split()))
    initial = list(map(float, input().split()))
    obs_input = list(map(int, input().split()))
    hidden_state = get_delta_state(A, B, initial, obs_input)
    output = " ".join(str(x) for x in hidden_state)
    print(output)