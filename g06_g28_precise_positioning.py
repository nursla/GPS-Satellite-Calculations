#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NurSılaÖzkan_2210674038_A3_GMT312


# In[2]:


import numpy as np


# In[3]:


c = 299792458 #(m/s)
we = 7.2921151467e-5  #(rad/s)
trec = 31680 # =2+2+1+0+6+7+4+0+3+8*960
pc_G06 = 24848028.368 #Pseudorange G06
pc_G28 = 23205784.623 #Pseudorange G28


# In[4]:


def approx_pos(rinex_file):
    with open(rinex_file, 'r') as file:
        for line in file:
            if "APPROX POSITION XYZ" in line:
                parts = line.split()
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                return np.array([x, y, z])
    return None

rinex_file = "C:\\Users\\nanus\\Desktop\\MERS00TUR_R_20250910000_01D_30S_MO.rnx"
r_apr = approx_pos(rinex_file)

if r_apr is not None:
    print(f"Approximate receiver position (ECEF): {r_apr}")
else:
    print("Approximate receiver position not found in the RINEX file.")
    
#  4239146.6414  2886967.1245  3778874.4800                  APPROX POSITION XYZ


# In[5]:


#Lagrange function in homework 2
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


# In[6]:


def emist(trec, pc, clk):
    t_vals = clk[:, 0]
    clk_vals = clk[:, 1]
    
    #clock correction by Lagrange interpolation
    clk_corr = lagrange(trec, np.column_stack((t_vals, clk_vals)))
    
    #calculation signal emission time
    tems = trec - (pc / c) - clk_corr
    return tems

#Matrices of satellite clock corrections taken from SP3 file for 10 periods(10x2)
clk_G06 = np.array([
    [27000, -279.627624e-9],
    [27700, -279.647363e-9],
    [28800, -279.667163e-9],
    [29700, -279.686954e-9],
    [30600, -279.706710e-9],
    [31500, -279.726526e-9],
    [32400, -279.746362e-9],
    [33300, -279.766178e-9],
    [34200, -279.786072e-9],
    [35100, -279.805844e-9]
])

clk_G28 = np.array([
    [27000, -589.362734e-9],
    [27700, -589.368763e-9],
    [28800, -589.374828e-9],
    [29700, -589.380837e-9],
    [30600, -589.386822e-9],
    [31500, -589.392831e-9],
    [32400, -589.398860e-9],
    [33300, -589.404829e-9],
    [34200, -589.410918e-9],
    [35100, -589.416940e-9]
])

tems_G06 = emist(trec, pc_G06, clk_G06)
tems_G28 = emist(trec, pc_G28, clk_G28)

print("Signal emission time (Satellite G06):", tems_G06)
print("Signal emission time (Satellite G28):", tems_G28)


# In[7]:


def sat_pos(trec, pc, sp3, r_apr):
    #data taken from SP3 file
    t_vals = sp3[:, 0]
    sat_coords = sp3[:, 1:4]
    clk_vals = sp3[:, 4]
    
    #calculation satellite coordinates with Lagrange interpolation
    x_sat = lagrange(trec, np.column_stack((t_vals, sat_coords[:, 0])))
    y_sat = lagrange(trec, np.column_stack((t_vals, sat_coords[:, 1])))
    z_sat = lagrange(trec, np.column_stack((t_vals, sat_coords[:, 2])))
    
    #calculation satellite clock correction 
    clk_corr = lagrange(trec, np.column_stack((t_vals, clk_vals)))
    
    # coordinates and corrections
    r_apr_sat = np.array([x_sat, y_sat, z_sat]) + clk_corr
    
    # calculation geometric distance
    r_dist = np.linalg.norm(r_apr_sat - r_apr)
    
    # time the signal travels
    delta_t = r_dist / c
    
    # calculation arth rotation correction
    R3 = np.array([[np.cos(we * delta_t), np.sin(we * delta_t), 0],
                   [-np.sin(we * delta_t), np.cos(we * delta_t), 0],
                   [0, 0, 1]])
    
    # calculation final satellite coordinates
    fpos = np.dot(R3, r_apr_sat)
    return fpos

#matrix containing satellite clock corrections for ten consecutive periods obtained from precise ephemeris(10x5)
sp3_G06 = np.array([
    [27000, -454.541220, 18077.669083, 19470.693115, -279.627624],
    [27700, -2137.711296, 16533.867238, 20707.621260, -279.647363],
    [28800, -3994.991451, 15009.822599, 21589.218783, -279.667163],
    [29700, -5997.263777, 13546.471169, 22100.838747, -279.686954],
    [30600, -8108.581134, 12179.606466, 22234.174779, -279.706710],
    [31500, -10287.385102, 10938.608750, 21987.369722, -279.726526],
    [32400, -12487.939058, 9845.434869, 21365.017867, -279.746362],
    [33300, -14661.921916, 8913.904587, 20378.062710, -279.766178],
    [34200, -16760.122641, 8149.307986, 19043.593506, -279.786072],
    [35100, -18734.172457, 7548.346498, 17384.545015, -279.805844]
])

sp3_G28 = np.array([
    [27000, 3085.235084, -16704.615725, 20408.565186, -589.362734],
    [27700, 4914.650249, -15174.923990, 21228.911385, -589.368763],
    [28800, 6878.072216, -13698.172736, 21683.707032, -589.374828],
    [29700, 8940.882865, -12308.826892, 21765.151045, -589.380837],
    [30600, 11063.050756, -11035.310522, 21471.871702, -589.386822],
    [31500, 13200.508131, -9899.018163, 20808.948117, -589.392831],
    [32400, 15306.691123, -8913.609169, 19787.820890, -589.398860],
    [33300, 17334.184380, -8084.611662, 18426.093544, -589.404829],
    [34200, 19236.407241, -7409.350653, 16747.228229, -589.410918],
    [35100, 20969.277014, -6877.202412, 14780.140896, -589.416940]
])

#G06
fpos_G06 = sat_pos(trec, pc_G06, sp3_G06, r_apr)
#G28
fpos_G28 = sat_pos(trec, pc_G28, sp3_G28, r_apr)

# results
print("Final satellite coordinates (Satellite G06):", fpos_G06)
print("Final satellite coordinates (Satellite G28):", fpos_G28)


# In[8]:


#NurSılaÖzkan_2210674038_A3_GMT312


# In[ ]:




