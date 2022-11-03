"""https://i.pinimg.com/originals/bd/b6/cd/bdb6cd9f52015c66fd48ec56a65f6b7e.png"""
import numpy as np
from matplotlib import pyplot as plt

def get_alphabet():
    A = np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1]])

    B = np.array([[1, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 0]])

    C = np.array([[0, 1, 1, 1],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 1, 1]])

    D = np.array([[1, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 0]])

    E = np.array([[1, 1, 1, 1],
                  [1, 0, 0, 0],
                  [1, 1, 1, 0],
                  [1, 0, 0, 0],
                  [1, 1, 1, 1]])

    F = np.array([[1, 1, 1, 1],
                  [1, 0, 0, 0],
                  [1, 1, 1, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0]])

    G = np.array([[0, 1, 1, 1],
                  [1, 0, 0, 0],
                  [1, 0, 1, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 1]])

    H = np.array([[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1]])

    # I

    J = np.array([[0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]])

    K = np.array([[1, 0, 0, 1],
                  [1, 0, 1, 0],
                  [1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [1, 0, 0, 1]])

    # L

    # M

    N = np.array([[1, 0, 0, 1],
                  [1, 1, 0, 1],
                  [1, 0, 1, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1]])

    O = np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]])

    P = np.array([[1, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0]])
    # 13
    R = np.array([[1, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 0, 0, 1]])

    S = np.array([[0, 1, 1, 1],
                  [1, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 1],
                  [1, 1, 1, 0]])

    #  T

    U = np.array([[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]])


    Z = np.array([[1, 1, 1, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 1, 1, 1]])
    # 17 that fit in a 5x4 grid
    alphabet = [A, B, C, D, E, F, G, H, J, K, N, O, P, R, S, U, Z]


    W = np.array([[1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1],
                  [1, 0, 1, 0, 1],
                  [0, 1,0, 1, 0]])
    return alphabet

# nletters = len(alphabet)
# simmat = np.zeros((nletters, nletters))
# for i in range(nletters):
#     for j in range(nletters):
#         char1 = alphabet[i]
#         char2 = alphabet[j]
#         simmat[i,j] = np.sum(char1 == char2)
# plt.matshow(simmat)
# plt.xticks(range(17),['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'N', 'O', 'P', 'R', 'S', 'U', 'Z'])
# plt.yticks(range(17),['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'N', 'O', 'P', 'R', 'S', 'U', 'Z'])
# plt.savefig('figures/letters/pixel_overlap.png', dpi=300)
