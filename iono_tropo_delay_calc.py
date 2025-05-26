#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NurSılaÖZKAN_2210674038


# In[2]:


import math
import numpy as np


# In[3]:


#parameters
c = 299792458#(m/s)
wE = 7.2921151467e-5  #(rad/s)
a = 6378137.0  #semimajor axis
f = 1 / 298.257223563  #flatteningfactor
e2 = f * (2 - f)  #square of eccentricity


# In[4]:


#ınputs
day_of_year = 91 #april 1 = day 91
trec = 27720 #=2+2+1+0+6+7+4+0+3+8*840
trecw = 27720 #trecw = trec, april 1, 2025 already falls on tuesday, the first day of the week.
C1 = 22672722.279 #Pseudorange G06
rec = np.array([4239146.6414, 2886967.1245, 3778874.4800]) #Approximate receiver position
sp3 = np.array([
    [27000, -454.541220, 18077.669083, 19470.693115, -279.627624e-9],
    [27700, -2137.711296, 16533.867238, 20707.621260, -279.647363e-9],
    [28800, -3994.991451, 15009.822599, 21589.218783, -279.667163e-9],
    [29700, -5997.263777, 13546.471169, 22100.838747, -279.686954e-9],
    [30600, -8108.581134, 12179.606466, 22234.174779, -279.706710e-9],
    [31500, -10287.385102, 10938.608750, 21987.369722, -279.726526e-9],
    [32400, -12487.939058, 9845.434869, 21365.017867, -279.746362e-9],
    [33300, -14661.921916, 8913.904587, 20378.062710, -279.766178e-9],
    [34200, -16760.122641, 8149.307986, 19043.593506, -279.786072e-9],
    [35100, -18734.172457, 7548.346498, 17384.545015, -279.805844e-9]
]) ##matrix containing satellite clock corrections for ten consecutive periods obtained from precise ephemeris(10x5)

alpha = [0.3353e-07, 0.7451e-08, -0.1788e-06, 0.0]
beta = [0.1372e+06, 0.0, -0.3277e+06, 0.2621e+06] #ion alpha/beta password taken from navigation file


# In[5]:


#codes of assignment 3

def lagrange(eph, dat):
    time_tags = dat[:, 0]
    values = dat[:, 1]
    n = len(time_tags)

    result = 0.0
    for i in range(n):
        term = values[i]
        for j in range(n):
            if i != j:
                term *= (eph - time_tags[j]) / (time_tags[i] - time_tags[j])
        result += term

    return result

def emist(trec, pc, clk):
    clk_corr = lagrange(trec, clk)
    tems = trec - (pc / c) - clk_corr
    return tems

#I made some changes according to the other code
def sat_pos(trec, pc, sp3, r_apr):
    tems = emist(trec, pc, sp3[:, [0, 4]])

    sat_x = lagrange(tems, sp3[:, [0, 1]]) * 1000
    sat_y = lagrange(tems, sp3[:, [0, 2]]) * 1000
    sat_z = lagrange(tems, sp3[:, [0, 3]]) * 1000
    r_sat = np.array([sat_x, sat_y, sat_z])

    rho = np.linalg.norm(r_sat - r_apr)
    dt = rho / c
    theta = wE * dt

    R3 = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    fpos = R3 @ r_sat
    return fpos


# In[6]:


#codes of assignment 1
def xyz2plh(cart):
    X, Y, Z = cart
    lon = np.arctan2(Y, X)
    p = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Z, p * (1 - e2))
    for _ in range(100):
        N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        h = p / np.cos(phi) - N
        phi_new = np.arctan2(Z, p * (1 - e2 * (N / (N + h))))
        if abs(phi_new - phi) < 1e-12:
            break
        phi = phi_new
    lat = np.degrees(phi)
    lon = np.degrees(lon)
    return lat, lon, h

def local(ecef_sat, ecef_rec):
    lat, lon, h = xyz2plh(ecef_rec)
    diff = ecef_sat - ecef_rec
    east = -np.sin(np.radians(lon)) * diff[0] + np.cos(np.radians(lon)) * diff[1]
    north = -np.sin(np.radians(lat)) * np.cos(np.radians(lon)) * diff[0] - np.sin(np.radians(lat)) * np.sin(np.radians(lon)) * diff[1] + np.cos(np.radians(lat)) * diff[2]
    up = np.cos(np.radians(lat)) * np.cos(np.radians(lon)) * diff[0] + np.cos(np.radians(lat)) * np.sin(np.radians(lon)) * diff[1] + np.sin(np.radians(lat)) * diff[2]
    azimuth = np.arctan2(east, north)
    zenith = np.arccos(up / np.linalg.norm(diff))
    slant_distance = np.linalg.norm(diff)
    return azimuth, zenith, slant_distance


# In[7]:


#Code in the python files given in the assignment

def trop_SPP(lat, D, H, E):
    # Average meteorological parameters for the tropospheric delay: pressure [P(mbar)], temperature [T(K)],
    # water vapour pressure [e(mbar)], temperature rate [Beta(K/m)] and water vapour rate [Alpha (Dimensionless)]
    Met15 = [1013.25, 299.65, 26.31, 6.30e-3, 2.77]
    Met30 = [1017.25, 294.15, 21.79, 6.05e-3, 3.15]
    Met45 = [1015.75, 283.15, 11.66, 5.58e-3, 2.57]
    Met60 = [1011.75, 272.15, 6.78, 5.39e-3, 1.81]
    Met75 = [1013.00, 263.65, 4.11, 4.53e-3, 1.55]

    # Seasonal variations for the meteorological parameters
    dMet15 = [0.00, 0.00, 0.00, 0.00e-3, 0.00]
    dMet30 = [-3.75, 7.00, 8.85, 0.25e-3, 0.33]
    dMet45 = [-2.25, 11.00, 7.24, 0.32e-3, 0.46]
    dMet60 = [-1.75, 15.00, 5.36, 0.81e-3, 0.74]
    dMet75 = [-0.50, 14.50, 3.39, 0.62e-3, 0.30]

    if lat >= 0:
        Dmin = 28
    else:
        Dmin = 211

    A = np.zeros(5)
    B = np.zeros(5)

    lat = abs(lat)

    if lat <= 15:
        A = np.array(Met15)
        B = np.array(dMet15)
    elif lat > 15 and lat <= 30:
        for i in range(5):
            A[i] = Met15[i] + ((lat - 15) / 15) * (Met30[i] - Met15[i])
            B[i] = dMet15[i] + ((lat - 15) / 15) * (dMet30[i] - dMet15[i])
    elif lat > 30 and lat <= 45:
        for i in range(5):
            A[i] = Met30[i] + ((lat - 30) / 15) * (Met45[i] - Met30[i])
            B[i] = dMet30[i] + ((lat - 30) / 15) * (dMet45[i] - dMet30[i])
    elif lat > 45 and lat <= 60:
        for i in range(5):
            A[i] = Met45[i] + ((lat - 45) / 15) * (Met60[i] - Met45[i])
            B[i] = dMet45[i] + ((lat - 45) / 15) * (dMet60[i] - dMet45[i])
    elif lat > 60 and lat < 75:
        for i in range(5):
            A[i] = Met60[i] + ((lat - 60) / 15) * (Met75[i] - Met60[i])
            B[i] = dMet60[i] + ((lat - 60) / 15) * (dMet75[i] - dMet60[i])
    elif lat >= 75:
        A = np.array(Met75)
        B = np.array(dMet75)

    ME = 1.001 / np.sqrt(0.002001 + (np.sin(E) ** 2))

    k1 = 77.604
    k2 = 382000
    Rd = 287.054
    gm = 9.784
    g = 9.80665

    P = A[0] - B[0] * np.cos((2 * np.pi * (D - Dmin)) / 365.25)
    T = A[1] - B[1] * np.cos((2 * np.pi * (D - Dmin)) / 365.25)
    e = A[2] - B[2] * np.cos((2 * np.pi * (D - Dmin)) / 365.25)
    Beta = A[3] - B[3] * np.cos((2 * np.pi * (D - Dmin)) / 365.25)
    Alpha = A[4] - B[4] * np.cos((2 * np.pi * (D - Dmin)) / 365.25)

    Trz0d = (1e-6 * k1 * Rd * P) / gm
    Trz0w = ((1e-6 * k2 * Rd) / (((Alpha + 1) * gm) - Beta * Rd)) * (e / T)

    Trzd = ((1 - ((Beta * H) / T)) ** (g / (Rd * Beta))) * Trz0d
    Trzw = ((1 - ((Beta * H) / T)) ** ((((Alpha + 1) * g) / (Rd * Beta)) - 1)) * Trz0w

    return Trzd, Trzw, ME


# In[8]:


#Code in the python files given in the assignment
def Ion_Klobuchar(lat, lon, elv, azm, alfa, beta, tgps):
    # velocity of light
    c = 299792458  # m/s

    # calculate the Earth-centred angle
    Re = 6378  # km
    h = 350  # km
    cns = (Re / (Re + h)) * math.cos(elv)
    eca = math.pi / 2 - elv - math.asin(cns)

    # compute the latitude of IPP
    ax = math.sin(lat) * math.cos(eca) + math.cos(lat) * math.sin(eca) * math.cos(azm)
    lat_ipp = math.asin(ax)

    # compute the longitude of IPP
    lon_ipp = lon + (eca * math.sin(azm)) / (math.cos(lat_ipp))

    # Find the geomagnetic latitude of the IPP
    f_pol = math.radians(78.3)
    l_pol = math.radians(291)
    as_ = math.sin(lat_ipp) * math.sin(f_pol) + math.cos(lat_ipp) * math.cos(f_pol) * math.cos(lon_ipp - l_pol)
    geo = math.asin(as_)

    # Find the local time at the IPP
    t = 43200 * (lon_ipp / math.pi) + tgps
    t = t % 86400  # Seconds of day
    if t >= 86400:
        t = t - 86400
    elif t <= 0:
        t = t + 86400

    tsd = geo / math.pi
    AI = alfa[0] + alfa[1] * tsd + alfa[2] * (tsd ** 2) + alfa[3] * (tsd ** 3)  # seconds
    PI = beta[0] + beta[1] * tsd + beta[2] * (tsd ** 2) + beta[3] * (tsd ** 3)  # seconds
    if AI < 0:
        AI = 0
    if PI < 72000:
        PI = 72000

    # Compute the phase of ionospheric delay
    XI = (2 * math.pi * (t - 50400)) / PI  # radian

    # Compute the slant factor (ionospheric mapping function)
    F = (1 - cns ** 2) ** (-1 / 2)

    # Compute the ionospheric time delay
    if abs(XI) < (math.pi / 2):
        I1 = (5 * (10 ** (-9)) + AI * math.cos(XI)) * F
    elif abs(XI) >= (math.pi / 2):
        I1 = (5 * (10 ** (-9))) * F

    dion = I1 * c

    return dion


# In[9]:



def atmos(day_of_year, trec, trecw, C1, rec, sp3, alpha, beta):
    #calculate satellite position
    fpos = sat_pos(trec, C1, sp3, rec)
    az, zen, slantd = local(fpos, rec)
    az_deg = np.degrees(az) % 360
    zen_deg = np.degrees(zen)
    lat, lon, h = xyz2plh(rec)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    elv_rad = math.radians(90 - zen_deg)
    azm_rad = math.radians(az_deg)
    #compute ionospheric delay using the klobuchar model code
    IonD = Ion_Klobuchar(lat_rad, lon_rad, elv_rad, azm_rad, alpha, beta, trecw)
     #compute tropospheric delays -dry and wet components- using the saastamoinen model code
    TrD, TrW, _ = trop_SPP(lat, day_of_year, h, elv_rad)
    return az_deg, zen_deg, slantd / 1000, IonD, TrD, TrW

az, zen, slantd, IonD, TrD, TrW = atmos(day_of_year, trec, trecw, C1, rec, sp3, alpha, beta)

print(f"Azimuth in degrees: {az:.3f}")
print(f"Zenith in degrees: {zen:.3f}")
print(f"Slant distance in kilometers: {slantd:.3f} km")
print(f"Ionospheric delay in meters: {IonD:.4f} m")
print(f"Tropospheric dry delay in meters: {TrD:.4f} m")
print(f"Tropospheric wet delay in meters: {TrW:.4f} m")


# In[10]:


#NurSılaÖZKAN_2210674038


# In[ ]:




