import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

def rossler_first_return_map(ID):
    """
    Initial conditions of the simultation depends on the ID to
    ensure to have distincts values for each call of the map
    """
    Y0 = [-0.4+0.1*ID, 0, 0]
    L = rossler_simulation(30000000, 0.01, Y0)  # Solve the Rossler system
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
    save_data(section, 'rossler_'+str(ID)+'_section')
    # save_data(appli, 'rossler_'+str(ID)+'_appli')
    print(len(appli))  # number of points in the map
    return appli


def rossler_simulation(temps=1000, pas=0.001, Y0=[-0.4, 0, 0], alpha=-0.25):
    a = 0.2+0.09*alpha
    b = 0.2-0.06*alpha
    c = 5.7-1.18*alpha
    f = lambda Y, t: [-Y[1]-Y[2], Y[0]+a*Y[1], b+Y[2]*(Y[0]-c)]
    Lt = np.arange(0, temps, pas)
    L = odeint(f, Y0, Lt)
    return L


def save_data(L, name):
    fichier = open(name+".dat", 'w')
    for line in L:
        fichier.write(str(line[0])+"\t"+str(line[2])+"\n")
    fichier.close()


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


def lorenz_first_return_map(ID):
    # initial conditions
    Y0 = [-0.4 + 0.1 * ID, 0, 0]
    # parameters
    r = 28.0
    b = 8.0/3.0
    s = 10.0
    # resolution of the system with Runge-Kutta
    L = lorenz_simulation(1000000, 0.01, Y0)
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
    print(len(appli))  # number of points in the map
    return appli


def lorenz_simulation(temps=1000, pas=0.001, Y0=[-0.4, 0, 0]):
    r = 28.0
    b = 8.0/3.0
    s = 10.0
    f = lambda Y, t: [s*(Y[1]-Y[0]), r*Y[0]-Y[1]-Y[0]*Y[2], -b*Y[2]+Y[0]*Y[1]]
    Lt = np.arange(0, temps, pas)
    L = odeint(f, Y0, Lt)
    return L


def main():
    # Generate values for Rossler and Lorenz Maps
    for ID in range(30):
    # TODO change the ID value to have different application for each particle for PSO
        ID2 = ID
        if ID == 4: # 4 causes error
            ID = 30
        appli = rossler_first_return_map(ID) # appli in [0:1]
        save_data(appli, 'maps/rossler_'+str(ID2)+'_appli')
        appli = lorenz_first_return_map(ID) # WARNING appli in [0:2]
        save_data(appli, 'maps/lorenz_'+str(ID2)+'_appli')
    # appli = lorenz_first_return_map(0)  # WARNING appli in [0:2]
    # save_data(appli, 'maps/lorenz_' + str(1) + '_appli')

if __name__ == '__main__':
    main()

    # # Lorenz Plot
    # data = np.genfromtxt("maps/lorenz_15_appli.dat")
    # X = data[:, 0]
    # Y = data[:, 1]
    # plt.plot(X, Y, 'b,')
    # plt.show()
    #
    # # Rossler Plot
    # data = np.genfromtxt("maps/rossler_18_appli.dat")
    # X = data[:, 0]
    # Y = data[:, 1]
    # plt.plot(X, Y, 'b,')
    # plt.show()

