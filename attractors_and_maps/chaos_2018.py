import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math


def rossler_first_return_map(ID, GEN):
    """
    Initial conditions of the simultation depends on the ID to
    ensure to have distincts values for each call of the map
    """
    Y0 = [-0.3 - 0.00001 * ID, 0+ 0.00001 * ID, 0]
    # estimate sim step to improve performance of the algorithm
    sim_step = GEN
    if 750 < GEN:
        sim_step = int(GEN*6)
    elif 200 <= GEN <= 750:
        sim_step = int(GEN*6.5)
    elif 100 <= GEN < 200:
        sim_step = int(GEN*7)
    elif 0 < GEN < 100:
        sim_step = int(GEN * 7.5)

    L = rossler_simulation(sim_step, 0.01, Y0)  # Solve the Rossler system
    L = rotate(L)  # Necessarry to have a clockwise flow
    # Uncomment to save the solution (require to create folder 'data')
    # save_data(L, 'rossler_'+str(ID)+'_rk4')
    appli = list()
    section = list()
    transitoire = 4000
    count = 0
    current = [0, 0, 0]
    sec = 0
    secmin = -4.32
    secmax = -10.487
    for line in L:
        count += 1
        previous = list(current)
        current = line
        if count > transitoire:
            in_section = poincare_section(0, 0, previous[0], current[0], previous[1], current[1])
            in_other = poincare_section(0, 0, previous[0], current[0], previous[2], current[2])
            if in_section[0]:
                section.append([0, in_section[1], in_other[1]])
                tpsec = sec
                sec = (in_section[1]-secmin)/(secmax-secmin)
                # sec = in_section[1]
                if tpsec != 0:
                    appli.append([tpsec, sec, in_section[1]])
    # Uncomment to save data: the section and the first return map values
    # save_data(section, 'rossler_'+str(ID)+'_section')
    # save_data(appli, 'rossler_'+str(ID)+'_appli')
    print(len(appli))  # number of points in the map
    # appli_0 = appli[:, 0]
    first_c = []
    for i in range(GEN):
        first_c.append(appli[i][0])
    return first_c


def rossler_simulation(temps=1000, pas=0.001, Y0=[-0.4, 0, 0], alpha=-0.25):
    a = 0.2+0.09*alpha
    b = 0.2-0.06*alpha
    c = 5.7-1.18*alpha
    f = lambda Y, t: [-Y[1]-Y[2], Y[0]+a*Y[1], b+Y[2]*(Y[0]-c)]
    Lt = np.arange(0, temps, pas)
    L = odeint(f, Y0, Lt)
    return L


def save_data(L, name):
    file = open(name+".dat", 'w')
    for line in L:
        # CORRECT HERE
        for i in range(len(line)-1):
            file.write(str(line[i])+"\t")
        file.write(str(line[len(line)-1])+"\n")
    file.close()


def poincare_section(sens, plan, v0, v1, obs0, obs1):
    if sens == 0:
        if v0 > plan and v1 <= plan:
            res = True
            coef = (obs1-obs0)/(v1-v0)
            obs = obs1-coef*(v1-plan)
        else:
            res = False
            obs = 0
    else:
        if v0 < plan and v1 >= plan:
            res = True
            coef = (obs1-obs0)/(v1-v0)
            obs = obs1-coef*(v1-plan)
        else:
            res = False
            obs = 0
    return [res, obs]


def rotate(L):
    """ rotation for clockwise flow for Rossler system """
    L_rotate = list(L)
    for i in range(0, len(L)):
        L_rotate[i][0] = -L[i][0]
        L_rotate[i][2] = -L[i][2]
    return L_rotate


def lorenz_first_return_map(ID, GEN):
    # initial conditions
    Y0 = [-0.3 - 0.00001 * ID, 0+ 0.00001 * ID, 0]
    # parameters
    r = 28.0
    b = 8.0/3.0
    s = 10.0
    # estimate sim step to improve performance of the algorithm
    sim_step = GEN
    if GEN >= 1000:
        sim_step = int(GEN*0.8)
    elif 500 <= GEN < 1000:
        sim_step = int(GEN*0.825)
    elif 300 <= GEN < 500:
        sim_step = int(GEN*0.9)
    elif GEN < 200:
        sim_step = int(GEN*1.2)
    # resolution of the system with Runge-Kutta
    L = lorenz_simulation(sim_step, 0.01, Y0)
    # Uncomment to save the solution (require to create folder 'data')
    # save_data(L, 'lorenz_'+str(ID)+'_rk4')
    appli = list()
    section = list()
    transitoire = 4000
    count = 0
    current = [0, 0, 0]
    sec = 0
    xplus = math.sqrt(b*(r-1))
    xminus = -xplus
    secmin1 = 10
    secmax1 = 18
    secmin2 = -18
    secmax2 = -10
    for line in L:
        count += 1
        previous = list(current)
        current = line
        if count > transitoire:
            in_section1 = poincare_section(0, xplus, previous[1], current[1], previous[0], current[0])
            in_other1 = poincare_section(0, xplus, previous[1], current[1], previous[2], current[2])
            if in_section1[0]:
                section.append([in_section1[1], xplus, in_other1[1]])
                tpsec = sec
                sec = (in_section1[1]-secmin1)/(secmax1-secmin1)
                if tpsec != 0:
                    appli.append([tpsec, sec, in_section1[1]])

            in_section2 = poincare_section(1, xminus, previous[1], current[1], previous[0], current[0])
            in_other2 = poincare_section(1, xminus, previous[1], current[1], previous[2], current[2])
            if in_section2[0]:
                section.append([in_section2[1], xminus, in_other2[1]])
                tpsec = sec
                sec = 2 - (in_section2[1]-secmin2)/(secmax2-secmin2)
                if tpsec != 0:
                    appli.append([tpsec, sec, in_section2[1]])

    #  self.chaos_Y0_firefly[ID] = self.rotateInitial(section.__getitem__(len(section) - 1)) # Does it need rotation or not ?


    # Uncomment to save data: the section and the first return map values
    # save_data(section, 'lorenz_'+str(ID)+'_section')
    # save_data(appli, 'lorenz_'+str(ID)+'_appli')
    print(len(appli)) # number of points in the map
    first_c = []
    for i in range(GEN):
        first_c.append(appli[i][0])
    return first_c


def lorenz_simulation(temps=1000, pas=0.001, Y0=[-0.4, 0, 0]):
    r = 28.0
    b = 8.0/3.0
    s = 10.0
    f = lambda Y, t: [s*(Y[1]-Y[0]), r*Y[0]-Y[1]-Y[0]*Y[2], -b*Y[2]+Y[0]*Y[1]]
    Lt = np.arange(0, temps, pas)
    L = odeint(f, Y0, Lt)
    return L


def rossler_map_generator(ID, gen, dimension):
    # Generate values for Rossler and Lorenz Maps
    rossler_map = np.ndarray(shape=(gen, dimension), dtype=float, order='F')
    # TODO change the ID value to have different application for each particle for PSO
    for i in range(dimension): # number of dimensions
        appli = rossler_first_return_map(ID, gen)  # appli in [0]
        rossler_map[:, i] = appli
        ID += 1
    return rossler_map


def lorenz_map_generator(ID, gen, dimension):
    # Generate values for Rossler and Lorenz Maps
    lorenz_map = np.ndarray(shape=(gen, dimension), dtype=float, order='F')
    # TODO change the ID value to have different application for each particle for PSO
    for i in range(dimension): # number of dimensions
        appli = lorenz_first_return_map(ID, gen)  # appli in [0]
        lorenz_map[:, i] = appli
        ID += 1
    return lorenz_map, ID


def main():
    # Generate values for Rossler and Lorenz Maps
    ID = 500000
    GEN = 200
    DIM = 100
    rossler_map = np.ndarray(shape=(GEN, DIM), dtype=float, order='F')
    lorenz_map = np.ndarray(shape=(GEN, DIM), dtype=float, order='F')

    for experiment in range(1): # number of particles * experiments
    # TODO change the ID value to have different application for each particle for PSO
        for i in range(DIM): # number of dimensions
            appli = rossler_first_return_map(ID, GEN)  # appli in [0]
            rossler_map[:, i] = appli
            appli = lorenz_first_return_map(ID, GEN)  # appli in [0]
            lorenz_map[:, i] = appli
            ID += 1
        save_data(rossler_map, 'maps/rossler_maps/rossler_'+str(experiment)+'_map')
        save_data(lorenz_map, 'maps/lorenz_maps/lorenz_'+str(experiment)+'_map')


if __name__ == '__main__':
    #main()

    # # Lorenz Plot
    # data = np.genfromtxt("maps/lorenz_15_appli.dat")
    # X = data[:, 0]
    # Y = data[:, 1]
    # plt.plot(X, Y, 'b,')
    # plt.show()
    #
    # Rossler Plot
    data = np.genfromtxt("maps/rossler_maps/rossler_18_appli.dat")
    X = data[:, 0]
    Y = data[:, 1]
    plt.plot(X, Y, 'b,')
    plt.show()

