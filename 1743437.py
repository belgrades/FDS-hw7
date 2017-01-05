import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from lib1743734 import reduce, alldist, distortion


def plot_distortions(distortions, labels):
    plt.xlabel("distortion")
    plt.ylabel("frequency")

    for i, distort in enumerate(distortions):
        plt.hist(distort, normed=True, bins=100, histtype='step', label=labels[i])

    plt.legend()
    plt.grid()
    plt.show(block=True)


def main():
    np.set_printoptions(precision=2)
    param = sys.argv

    df = pd.read_csv(param[1])

    df = df.pivot(index='userId', columns='movieId', values='rating')

    nan_indexes = np.isnan(df)
    df = np.where(nan_indexes, np.ma.array(df, mask=nan_indexes).mean(axis=0), df)

    row, column = df.shape

    idx = np.random.choice(range(row), size=min(250, row), replace=False)

    dist_df = alldist(df[idx])
    nbytes = df[idx].nbytes

    distortions, labels = [], []
    for (i, d) in enumerate(map(int, param[2:])):
        labels.append(str(d))
        Y = reduce(df, d)[idx]
        distortions.append(distortion(alldist(Y), dist_df))
        print("%0.2f %0.2f %0.2f %0.2f" % (nbytes*1.0/Y.nbytes, distortions[i].min(), distortions[i].mean(), distortions[i].max()))

    plot_distortions(distortions, labels)

if __name__ == '__main__':
    main()