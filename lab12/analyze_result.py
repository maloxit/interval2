#! /bin/python3

import json
import os

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

import numpy as np

if __name__ == "__main__":
    os.makedirs(f'results/total', exist_ok=True)

    with open('results/result.json', 'r') as file:
        data = json.load(file)

    x_array = []
    y_array = []
    z1_array = []
    z2_array = []

    for x, yz in data.items():
        for y, array in yz.items():
            x_array.append(int(x))
            y_array.append(int(y))
            z1_array.append(array[0][2])
            z2_array.append(array[1][2])

    n = len(set(x_array))
    m = len(set(y_array))
    z1 = np.zeros(shape=(n, m), dtype=np.uint32)
    z2 = np.zeros(shape=(n, m), dtype=np.uint32)
    for i in range(len(x_array)):
        z1[x_array[i]  * n // 8][y_array[i] * m // 1024] = z1_array[i]
        z2[x_array[i]  * n // 8][y_array[i] * m // 1024] = z2_array[i]

    fig, ax = plt.subplots()
    ax.hist(z1_array, bins=10)
    ax.set_title("Количество изменённых строк в первом методе")
    plt.savefig(f'results/total/m1_chanded_hist.png')

    fig, ax = plt.subplots()
    ax.hist(z2_array, bins=10)
    ax.set_title("Количество удалённых строк во втором методе")
    plt.savefig(f'results/total/m2_deleted_hist.png')

    fig, ax = plt.subplots(figsize=[50, 5])
    im = ax.imshow(z1)

    ax.set_xticks(range(0, m, 10), labels=[sorted(list(set(y_array)))[i] for i in range(0, m, 10)])
    ax.set_yticks(range(0, n, 3), labels=[0, 3, 7])

    ax.set_title("Количество изменённых строк в первом методе")
    plt.xlabel("Ячейки")
    plt.ylabel("Каналы")
    fig.tight_layout()
    ax=plt.gca()
    for PCM in ax.get_children():
        if isinstance(PCM, mpl.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)
    plt.savefig(f'results/total/m1_chanded_hm.png')

    fig, ax = plt.subplots(figsize=[50, 5])
    im = ax.imshow(z2)

    ax.set_xticks(range(0, m, 10), labels=[sorted(list(set(y_array)))[i] for i in range(0, m, 10)])
    ax.set_yticks(range(0, n, 3), labels=[0, 3, 7])

    ax.set_title("Количество удалённых строк во втором методе")
    plt.xlabel("Ячейки")
    plt.ylabel("Каналы")
    fig.tight_layout()
    ax=plt.gca()
    for PCM in ax.get_children():
        if isinstance(PCM, mpl.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax) 
    plt.savefig(f'results/total/m2_deleted_hm.png')

    N = 1100
    print("First:")
    print(f"\tMean - {z1.mean()}, {z1.mean() / N * 100}%")
    print(f"\tMedian - {np.median(z1)}, {np.median(z1) / N * 100}%")
    print(f"\tMax - {z1.max()}, {z1.max() / N * 100}%")
    print(f"\tMin - {z1.min()}, {z1.min() / N * 100}%")

    N = 44
    print("Second:")
    print(f"\tMean - {z2.mean()}, {z2.mean() / N * 100}%")
    print(f"\tMedian - {np.median(z2)}, {np.median(z2) / N * 100}%")
    print(f"\tMax - {z2.max()}, {z2.max() / N * 100}%")
    print(f"\tMin - {z2.min()}, {z2.min() / N * 100}%")
