#!/usr/bin/env python
# coding: utf-8

# In[10]:


#NurSılaÖZKAN_2210674038_A1_GMT312


# In[11]:


import numpy as np


# In[12]:


a = 6378137.0  # semi major axis(m)
f = 1 / 298.257223563  # flattening Factor
e2 = 2 * f - f**2   # first eccentricity squared


# In[13]:


#To convert 3D cartesian coordinates (X Y Z) to geographic coordinates (latitude longitude altitude)
def xyz2plh(cart):
    
    X, Y, Z = cart
    lon = np.arctan2(Y, X)
    
    p = np.sqrt(X**2 + Y**2) #Calculation of projection distribution 
    phi = np.arctan2(Z, p * (1 - e2)) #initial for latitude

   #With Newton raphson iteration latitude(phi) becomes more precise
    for _ in range(100):  # "while True:" command can also be used
        
        N = a / np.sqrt(1 - e2 * np.sin(phi)**2) #radius of curvature
        h = p / np.cos(phi) - N #height
        phi_new = np.arctan2(Z, p * (1 - e2 * (N / (N + h)))) #precise latitude value

        if abs(phi_new - phi) < 1e-12:
            break
        phi = phi_new

    lat = np.degrees(phi)
    lon = np.degrees(lon)

    return np.array([lat, lon, h])


# In[14]:


#Converts the position difference between a satellite and receiver to the local ENU (EastNorthUp) coordinate system and then gives azimuth and zenith values
def local(rec, sat):
    dX = sat - rec 
    slantd = np.linalg.norm(dX) 
    
    phi, lam, _ = xyz2plh(rec) #Converts receiver ECEF coordinates to latitude,longitude,altitude
    phi, lam = np.radians(phi), np.radians(lam)
    
    # ENU dönüşüm matrisi
    R = np.array([[-np.sin(lam), np.cos(lam), 0],
                  [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
                  [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]])
#Calculates the east,north and up coordinates of the satellite relative to the receiver
    local_vec = R @ dX 
    E, N, U = local_vec
    
    az = np.degrees(np.arctan2(E, N)) % 360  
    zen = np.degrees(np.arccos(U / slantd))  
    
    return az, zen, slantd


# In[15]:


#Extra
# A modular code fragment with more functionality but same functionality as the code above
""" def enu_coordinates(rec, sat):

    rec_lat, rec_lon, _ = xyz2plh(rec)
    lat_rad = np.radians(rec_lat)
    lon_rad = np.radians(rec_lon)

    R = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)]
    ])
    delta = sat - rec
    enu = R @ delta
    return enu
    
def zenith_angle(enu):
    east, north, up = enu
    zen = 90 - np.degrees(np.arctan2(up, np.sqrt(east**2 + north**2))) 90°-zenith transformation
    return zen

def local(rec, sat):   # converts locale to enu
    enu = enu_coordinates(rec, sat)

    east, north, up = enu
    az = np.degrees(np.arctan2(east, north))
    if az < 0:
#If the angle is negative, +360 is added
        az += 360 #%360 command is more accurate and clean

    zen = zenith_angle(enu)
    slantd = np.linalg.norm(enu) #the distance between the satellite and the receiver is calculated with the euclidean norm

    return az, zen, slantd"""


# In[16]:


#Gets receiver and satellite coordinates from user
rec = np.array(list(map(float, input("Alıcı koordinatlarını girin/Enter receiver coordinates (X Y Z): ").split())))
sat = np.array(list(map(float, input("Uydu koordinatlarını girin/Enter satellite coordinates (X Y Z): ").split())))
 
#Calculate azimuth,zenith angle and slant distance
azimuth, zenith, slant_distance = local(rec, sat)

#Print azimuth, zenith angle and slant distance
print(f"Azimut Açısı/Azimuth Angle: {azimuth:.6f} Derece/Degree")
print(f"Zenit Açısı/Zenith Angle: {zenith:.6f} Derece/degree")
print(f"Eğik Mesafe/Slant Distance: {slant_distance:.6f} Metre/Meter")


# In[17]:


#example
rec = np.array([4126591.9053472467, 2638251.8309179237, 4073163.104790425]) #Receiver(Saraycık TOKİ(MyHome)) coordinates (meters)
sat = np.array([4191000.0, 261000.0, 24806000.0])  #Satellite coordinates (meters)

#Calculate azimuth, zenith angle and slant distance
azimuth, zenith, slant_distance = local(rec, sat)

#Print azimuth, zenith angle and slant distance
print(f"Azimut Açısı/Azimuth Angle: {azimuth:.6f}Derece/Degree")
print(f"Zenit Açısı/Zenith Angle: {zenith:.6f} Derece/degree")
print(f"Eğik Mesafe/Slant Distance: {slant_distance:.6f} Metre/Meter")


# In[18]:


#NurSılaÖZKAN_2210674038_A1_GMT312

