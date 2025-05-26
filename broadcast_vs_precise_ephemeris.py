#!/usr/bin/env python
# coding: utf-8

# In[34]:


#NurSılaÖZKAN_2210674038


# In[35]:


import numpy as np


# In[36]:


eph = 29400  # My epoch in seconds of day


# In[37]:


def cal_brd(eph, brd):
    GM = 3.986005e14  #Gravitational constant
    We = 7.2921151467e-5  #earth s rotation rate
    
#Split the brd array into 16 separate physical parameters
    Crs, Delta_n, M0, Cuc, e, Cus, sqrt_a, Toe, Cic, long_an, Cis, i0, Crc, ap, ra, idot = brd

    a = sqrt_a ** 2 #Semimajor axis
    n0 = np.sqrt(GM / a ** 3)
    n = n0 + Delta_n

    tk = eph - (Toe % 86400) #Time from ephemeris reference epoch("Toe % 86400" for exceeding the week)

    M = M0 + n * tk

    E = M
    for _ in range(10):
        E_prev = E
        E = M + e * np.sin(E) #Kepler's equation
        if np.abs(E - E_prev) < 1e-12:
            break

    nu = np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e) # True anomaly
    phi = nu + ap # Argument of latitude 

    #orbit corrections
    du = Cuc * np.cos(2 * phi) + Cus * np.sin(2 * phi)
    dr = Crc * np.cos(2 * phi) + Crs * np.sin(2 * phi)
    di = Cic * np.cos(2 * phi) + Cis * np.sin(2 * phi)
    
    #orbital parameters
    u = phi + du
    r = a * (1 - e * np.cos(E)) + dr
    i = i0 + idot * tk + di

    x_prime = r * np.cos(u)
    y_prime = r * np.sin(u)

    long_corr = long_an + (ra - We) * tk - We * Toe #change of right ascension with time + effect of earths own rotation

    x = x_prime * np.cos(long_corr) - y_prime * np.cos(i) * np.sin(long_corr)
    y = x_prime * np.sin(long_corr) + y_prime * np.cos(i) * np.cos(long_corr)
    z = y_prime * np.sin(i)

    return np.array([x/1000, y/1000, z/1000])  #convert to km


brd_brd = np.array([
   0.143000000000e+03, -0.128281250000e+03, 0.428589281035e-08, -0.309377111530e+01,
  -0.653043389320e-05, 0.170464231633e-02, 0.476278364658e-05, 0.515363643646e+04,
   0.201600000000e+06, 0.167638063431e-07, 0.283873202360e+01, 0.428408384323e-07,
   0.965876229513e+00, 0.290062500000e+03, -0.235987751519e+01, -0.818462663684e-08,
  -0.237509893240e-09, 0.100000000000e+01, 0.236000000000e+04, 0.000000000000e+00,
  0.200000000000e+01, 0.000000000000e+00, -0.884756400000e-08, 0.399000000000e+03,
  0.194460000000e+06, 0.400000000000e+01, 0.000000000000e+00, 0.000000000000e+00
])

brd = brd_brd[1:17] # 16 parameters from PNR11 for navigation message

brd_p = cal_brd(eph, brd)  #Calculate
print("Broadcast Position(X, Y, Z)]:",brd_p)


# In[38]:


#9th order Lagrange interpolation for my epoch  
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

def cal_sp3(eph, sp3): #Calculates satellite position using precise ephemeris
    t_tag = sp3[:, 0]
    x_dat = sp3[:, [0,1]]
    y_dat = sp3[:, [0,2]]
    z_dat = sp3[:, [0,3]]

    ##performs Lagrange interpolation to find the position in the period
    x_ip = lagrange(eph, x_dat)
    y_ip = lagrange(eph, y_dat)
    z_ip = lagrange(eph, z_dat)

    return np.array([x_ip, y_ip, z_ip])

#satellite coordinates (X, Y, Z) corresponding to five times before and after the relevant time
sp3 = np.array([
   [24300, 11167.721457, 23836.452176,  3811.260369],
   [25200, 10656.056669, 23464.519119,  6591.227419],
   [26100,  9944.006934, 22871.438653,  9258.266755],
   [27000,  9014.019423, 22092.439425, 11766.726819],
   [27900,  7856.225454, 21166.905352, 14073.682258],
   [28800,  6469.011917, 20136.821536, 16139.656302],
   [29700,  4859.279022, 19045.134613, 17929.284799],
   [30600,  3042.374082, 17934.089994, 19411.911546],
   [31500,  1041.703158, 16843.609151, 20562.105624],
   [32400, -1111.965536, 15809.768519, 21360.092624] 
])  #10x4 matrix [time, X, Y, Z]

sp3_p = cal_sp3(eph, sp3)

print("Precise Position(X, Y, Z):",sp3_p)


# In[39]:



# In[40]:


#calculate the Interpolate values
dat = np.array([             
    [24300, 11167.721457],
    [25200, 10656.056669],
    [26100,  9944.006934],
    [27000,  9014.019423],
    [27900,  7856.225454],
     [28800,  6469.011917],
     [29700,  4859.279022],
    [30600,  3042.374082],
     [31500,  1041.703158],
     [32400, -1111.965536] 
    ])   #10x2 matrix [time, value]

value = lagrange(eph, dat)
print("Interpolate value:",value)


# In[41]:


#Additionally, absolute error is also a process to find the difference between precise and broadcast position
D = sp3_p - brd_p
TE = np.linalg.norm(D)
    
print("Difference:",D)
print("Total Error:",TE)


# In[42]:


#NurSılaÖzkan_2210674038

