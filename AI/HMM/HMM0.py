def matmult(a, b):
    zip_b = zip(*b)
    # uncomment next line if python 3 :
    # zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b))
             for col_b in zip_b] for row_a in a]


def matix(a):
    rows_a = int(a[0])
    cols_a = int(a[1])
    new_a = []
    for n in range(rows_a):
        new_a.append(a[2+n*cols_a: 2+(n+1)*cols_a])
    return new_a


if __name__ == '__main__':
    A = list(map(float, input().split()))
    B = list(map(float, input().split()))
    pi = list(map(float, input().split()))
    A = matix(A)
    B = matix(B)
    pi = matix(pi)
    # print(f"{A}\n{B}\n{pi}")
    states = matmult(pi, A)
    observations = matmult(states, B)
    col = len(observations[0])
    row = len(observations)
    observations = observations[0]
    observations = [round(x, 2) for x in observations]
    observations.insert(0, row)
    observations.insert(1, col)
    # print(observations)
    output = " ".join(str(x) for x in observations)
    print(output)