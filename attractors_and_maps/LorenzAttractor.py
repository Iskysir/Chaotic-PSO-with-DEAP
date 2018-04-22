import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def bound_estimator(stepCnt):
    xs_list = []
    ys_list = []
    zs_list = []
    for _ in range(1):
        dt = 0.01
        #stepCnt = 5000

        # Need one more for the initial values
        xs = np.empty((stepCnt + 1,))
        ys = np.empty((stepCnt + 1,))
        zs = np.empty((stepCnt + 1,))

        # Setting initial values
        xs[0], ys[0], zs[0] = (np.random.rand(), np.random.rand(), np.random.rand())

        # Stepping through "time".
        for i in range(stepCnt):
            # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)

        xs_list.extend(xs)
        ys_list.extend(ys)
        zs_list.extend(zs)

    #print(max(xs_list), min(xs_list))
    #print(max(ys_list), min(ys_list))
    return max(zs_list), min(zs_list)


def lorenz(x, y, z):
    s = 10.
    r = 28.
    b = 2.667
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def main():
    xs_list = []
    ys_list = []
    zs_list = []
    for _ in range(1):
        dt = 0.01
        stepCnt = 100

        # Need one more for the initial values
        xs = np.empty((stepCnt + 1,))
        ys = np.empty((stepCnt + 1,))
        zs = np.empty((stepCnt + 1,))

        # Setting initial values
        xs[0], ys[0], zs[0] = (np.random.rand(), np.random.rand(), np.random.rand())

        # Stepping through "time".

        for i in range(stepCnt):
            # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot(xs, ys, zs, lw=0.5)
        # ax.set_xlabel("X Axis")
        # ax.set_ylabel("Y Axis")
        # ax.set_zlabel("Z Axis")
        # ax.set_title("Lorenz Attractor")
        # plt.show()

        xs_list.extend(xs)
        ys_list.extend(ys)
        zs_list.extend(zs)

        # print(max(xs), min(xs))
        # print(max(ys), min(ys))
        # print(max(zs), min(zs))
        # print(len(xs_list))
        # print(len(ys_list))
        # print(len(zs_list))

    print(max(xs_list), min(xs_list))
    print(max(ys_list), min(ys_list))
    print(max(zs_list), min(zs_list))


if __name__ == '__main__':
    main()
