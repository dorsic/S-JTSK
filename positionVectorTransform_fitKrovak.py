"""
Based on paper https://www.geoportal.sk/sk/geodeticke-zaklady/geodeticke-systemy-transformacie/
More info on to position vector also here http://desktop.arcgis.com/en/arcmap/latest/map/projections/equation-based-methods.htm
Also note http://www.skgeodesy.sk/files/slovensky/ugkk/legislativa/vyhlaska_300_2009.pdf

To get the data query
    curl -XGET "https://zbgis.skgeodesy.sk/zbgis/rest/services/RGB/MapServer/2/query?f=json&where=1%3D1&returnGeometry=false&spatialRtrue&esriSpatialRelIntersects&outFields=*&orderByFields=OBJECTID%20ASC&outSR=5514&resultOffset=0&resultRecordCount=2000"
"""

import numpy as np
from typing import NamedTuple
import scipy.optimize

class Bessel1841(NamedTuple):
    a = 6377397.155 # je hlavná polos elipsoidu
    f = 1/(299.1528154) # je geometrické sploštenie elipsoidu
    b = a*(1 - f) # je vedľajšia polos elipsoidu
    e = np.sqrt((a**2 - b**2) / a**2) # je prvá excentricita

class Grs80(NamedTuple):
    a = 6378137.000 # je hlavná polos elipsoidu
    f = 1/(298.257222101) # je geometrické sploštenie elipsoidu
    b = a*(1 - f) # je vedľajšia polos elipsoidu
    e = np.sqrt((a**2 - b**2) / a**2) # je prvá excentricita

class Krovak(object):
    a_Bessel1841 = 6377397.155 # dĺžka hlavnej polosi elipsoidu Bessel 1841
    b_Bessel1841 = 6356078.9633 # dĺžka vedľajšej polosi elipsoidu Bessel 1841
    e = np.sqrt((a_Bessel1841**2 - b_Bessel1841**2) / a_Bessel1841**2)
    lamFG = np.radians(17+2/3) # zemepisná dĺžka medzi základnými poludníkmi Ferro a Greenwich na elipsoide Bessel 1841 (Ferro je na západ od Greenwich)
    phi0 = np.radians(49.5) # zemepisná šírka neskreslenej rovnobežky na elipsoide Bessel 1841
    lamKP = np.radians(42.5) # zemepisná dĺžka kartografického pólu na elipsoide Bessel 1841 (definovaná na východ od základného poludníka Ferro)
    alpha = 1.000597498372 # parameter charakterizujúci konformné zobrazenie elipsoidu Bessel 1841 na guľovú plochu (Gaussovu guľu)
    k = 1.003419164 # parameter charakterizujúci konformné zobrazenie elipsoidu Bessel 1841 na guľovú plochu (Gaussovu guľu)
    kc = 0.9999 # koeficient zmenšenia guľovej plochy (Gaussovej gule)
    a = np.radians(30+17/60+17.30311/3600)  # pólová vzdialenosť kartografického pólu na guľovej ploche (Gaussovej guli)
    S0 = np.radians(78.5) # zemepisná šírka základnej kartografickej rovnobežky na guľovej ploche (Gaussovej guli)
    ff = np.radians(45) # pomocna konstanta

    @staticmethod
    def default_values():
        return (Krovak.a_Bessel1841, Krovak.b_Bessel1841, Krovak.alpha, Krovak.k, Krovak.kc)

    def __init__(self, a=None, b=None, alpha=None, k=None, kc=None):
        self.a_Bessel1841 = a if a else self.a_Bessel1841
        self.b_Bessel1841 = b if b else self.b_Bessel1841
        self.alpha = alpha if alpha else self.alpha
        self.k = k if k else self.k
        self.kc = kc if kc else self.kc

    def plh_to_xy(self, plh):
        """
        4.2 Výpočet pravouhlých rovinných súradníc y,x Křovákovho zobrazenia z 
            elipsoidických súradníc phi , lambda vztiahnutých k elipsoidu Bessel 
            poludníkom Greenwich
        """
        phi, lam, h = plh
        dV = self.alpha * (self.lamKP - (lam+self.lamFG))
        U = 2 * (np.arctan(self.k * np.power(np.tan(phi/2+self.ff), self.alpha) * np.power((1+self.e*np.sin(phi))/(1-self.e*np.sin(phi)) ,-self.alpha*self.e/2)) - self.ff)
        S = np.arcsin(np.cos(self.a)*np.sin(U) + np.sin(self.a)*np.cos(U)*np.cos(dV))
        D = np.arcsin(np.cos(U)*np.sin(dV)/np.cos(S))
        R0 = self.kc * (self.a_Bessel1841*np.sqrt(1-self.e**2))/(1 - self.e**2*np.sin(self.phi0)**2) * 1/np.tan(self.S0)
        Dc = D * np.sin(self.S0)
        R = R0 * np.power(np.tan(self.S0/2+self.ff), np.sin(self.S0)) * np.power(1/np.tan(S/2+self.ff), np.sin(self.S0))
        y = R * np.sin(Dc)
        x = R * np.cos(Dc)
        return x, y, 0

    def xyz_to_plh(self, xyz):
        """
        4.3 Výpočet elipsoidickýh súradníc phi, lambda vztiahnutých k elipsoidu Bessel 1841 
        so základným poludníkom Greenwich z pravouhlých rovinných súradníc y,x Křovákovho zobrazenia
        """
        x, y, z = xyz
        R = np.sqrt(x**2 + y**2)
        Dc = np.arctan(y/x)
        R0 = self.kc * (self.a_Bessel1841*np.sqrt(1-self.e**2))/(1 - self.e**2 * np.sin(self.phi0)**2) * 1/np.tan(self.S0)
        S = 2 * (np.arctan(np.power(R0/R, 1/np.sin(self.S0)) * np.tan(self.S0/2+self.ff)) - self.ff)
        D = Dc/np.sin(self.S0)
        U = np.arcsin(np.cos(self.a)*np.sin(S) - np.sin(self.a)*np.cos(S)*np.cos(D))
        dV = np.arcsin(np.cos(S)*np.sin(D)/np.cos(U))
        lam = (self.lamKP - self.lamFG) - dV/self.alpha
        phi = U
        diff = 1
        while diff > 10E-12:
            phi1 = 2 * (np.arctan(np.power(self.k, -1/self.alpha) * np.power(np.tan(U/2+self.ff), 1/self.alpha) * np.power((1+self.e*np.sin(phi))/(1-self.e*np.sin(phi)), self.e/2) ) - self.ff)
            diff = abs(phi - phi1)
            phi = phi1

        return phi, lam, 0

class PositionVector(object):
    
    @staticmethod
    def transform(xyz, params):
        """xyz is array of [x, y, z]
            params is array of [dx, dy, dz, rx, ry, rz, scale] parameters"""
        x, y, z = xyz
        dx, dy, dz, rx, ry, rz, s = params
        rx, ry, rz = np.radians([rx, ry, rz])/3600
        tm = np.matrix([[dx], [dy], [dz]])
        rm = np.matrix([[1, rz, -ry], [-rz, 1, rx], [ry, -rx, 1]])
        xyz = tm + (1+s) * rm @ np.matrix([[x], [y], [z]])
        return xyz[0,0], xyz[1,0], xyz[2,0]

class Sjtsk(object):
    # Transformacne parametre odhadnute UGKK
    ETRF2000_JTSK03 = (-485.014055, -169.473618, -483.842943, 7.78625453, 4.39770887, 4.10248899, 0.0)
    JTSKO3_ETRF2000 = (485.021, 169.465, 483.839, -7.786342, -4.397554, -4.102655, 0.0)

    @staticmethod
    def plh_xyz(plh, elipsoid):
        """ 2.1 Prevod elipsoidických geodetických súradníc phi, lambda, h na pravouhlé karteziánske súradnice XYZ
            Transforms the phi, lambda, height coordinates to X, Y, Z.
            Elipsoid parameter defines the elipsoid with 2 properties - a and f."""
        phi, lam, hei = plh
        N = elipsoid.a/np.sqrt(1 - elipsoid.e**2 * (np.sin(phi)**2))
        X = (N+hei)*np.cos(phi)*np.cos(lam)
        Y = (N+hei)*np.cos(phi)*np.sin(lam)
        Z = (N*(1-elipsoid.e**2)+hei)*np.sin(phi)
        return X, Y, Z

    @staticmethod
    def xyz_plh(xyz, elipsoid):
        """ 2.2 Prevod pravouhlých karteziánskych súradníc XYZ na elipsoidické súradnice phi, lambda, h
            Transforms the X, Y, Z coordinates to phi, lambda, height.
            Elipsoid parameter defines the elipsoid with 2 properties - a and f."""
        X, Y, Z = xyz
        lam = np.arctan(Y/X)
        phi = np.arctan(Z/np.sqrt(X**2+Y**2)*(1/(1-elipsoid.e**2)))
        diff = 1
        while diff > 10E-12:
            N = elipsoid.a / np.sqrt(1 - elipsoid.e**2 * (np.sin(phi)**2))
            h = np.sqrt(X**2 + Y**2)/np.cos(phi) - N
            phi1 = np.arctan(Z/np.sqrt(X**2+Y**2) * ((N+h)/(N+h - elipsoid.e**2*N)))
            diff = abs(phi-phi1)
            phi = phi1

        return phi, lam, h

    @staticmethod
    def etrs_sjtsk(phi, lam, hei, params=ETRF2000_JTSK03, krovak=Krovak()):
        """
        2 MATEMATICKÁ DEFINÍCIA VZŤAHU MEDZI ETRS89 (ETRF2000) A S-JTSK (JTSK03) [EPSG::8365 a EPSG::8367]
        By default returns tbe JTSK03 coordinates.
        """
        etrf_XYZ = Sjtsk.plh_xyz([phi, lam, hei], Grs80())
        bessel_XYZ = PositionVector.transform(etrf_XYZ, params)
        bessel_plh = Sjtsk.xyz_plh(bessel_XYZ, Bessel1841())
        xy = krovak.plh_to_xy(bessel_plh)
        return xy

    @staticmethod
    def sjtsk_etrs(x, y, z, params=JTSKO3_ETRF2000, krovak=Krovak()):
        """
        2 MATEMATICKÁ DEFINÍCIA VZŤAHU MEDZI ETRS89 (ETRF2000) A S-JTSK (JTSK03) [EPSG::8365 a EPSG::8367]
        """
        bessel_plh = krovak.xyz_to_plh([x, y, z])
        bessel_XYZ = Sjtsk.plh_xyz(bessel_plh, Bessel1841())
        etrf_XYZ = PositionVector.transform(bessel_XYZ, params)
        etrf_plh = Sjtsk.xyz_plh(etrf_XYZ, Grs80())
        return etrf_plh

def read_data(csv_file="test.csv"):
    d = np.genfromtxt(csv_file, dtype="f8", delimiter=";", usecols=(9,10,11,12,13,14), names=True)
    return d

def residuals_etrs_sjtsk(params, *args, **kwargs):
    data = args
    res = np.zeros(len(data))
    i = 0
    for i in range(0, len(data)):
        jtsk_xy = Sjtsk.etrs_sjtsk(np.radians(data[i][0]), np.radians(data[i][1]), np.radians(data[i][2]), params)
        res[i] = (jtsk_xy[0]-data[i][4])**2 + (jtsk_xy[1]-data[i][3])**2
    return res

def residuals_sjtsk_etrs(params, *args, **kwargs):
    data = args
    res = np.zeros(len(data))
    i = 0
    for i in range(0, len(data)):
        a, b, alpha, k, kc = params[7:12]
        etrs_plh = Sjtsk.sjtsk_etrs(data[i][4], data[i][3], 0, params[0:7], Krovak(a, b, alpha, k, kc))
        etrs_deg = np.degrees(etrs_plh)
        res[i] = (etrs_deg[0]-data[i][0])**2 + (etrs_deg[1]-data[i][1])**2
    return res

def solve():
    data = read_data()
    # opt = scipy.optimize.least_squares(residuals_sjtsk_etrs, Sjtsk.JTSKO3_ETRF2000, xtol=1e-12, ftol=1e-10, args=data)
    opt = scipy.optimize.least_squares(residuals_sjtsk_etrs, 
        Sjtsk.JTSKO3_ETRF2000 + Krovak.default_values(), 
        gtol = None, xtol=1E-14, 
        args=data)
    print(opt)

def sjtsk_etrs(params, data):
    res = np.zeros((len(data), 2))
    i = 0
    for i in range(0, len(data)):
        a, b, alpha, k, kc = params[7:12]
        etrs_plh = Sjtsk.sjtsk_etrs(data[i][4], data[i][3], 0, params[0:7], Krovak(a, b, alpha, k, kc))
        etrs_deg = np.degrees(etrs_plh)
        #res[i] = etrs_deg[0:2]
        res[i] = (etrs_deg[0], etrs_deg[1])
    return res

def param_estimation():
    #solve() #JTSK->Etrs

    # solve 1 gtol=None (nfev: 20, njev: 7, cost: 5.54e-20, optimality: 4.42e-14)
    # [2.80650058e+01, 2.18220829e+02, 1.44515234e+03, -3.06062649e+00, -3.02553397e+01, -7.47955890e-02,  4.11854889e-04,  6.38009280e+06, 6.35607896e+06,  1.00061076e+00,  1.00341441e+00,  9.99396849e-01]
    # solve 2 gtol=None, xtol=1e-12 (nfev: 26, njev: 19, cost: 5.52e-20, optimality: 5.52e-20)
    # [28.06499195321056, 218.22083546140868, 1445.1523421985046, -3.0592250575099853, -30.254612594859292, -0.07341654882261767, 0.0004037423358959923, 6380092.7952426905, 6356078.9633, 1.0006117622235415, 1.0034130802898595, 0.9993955272861204]
    # solve 3 gtol=None, xtol=1e-14 (nfev: 30, njev: 8, cost: 5.53e-20, optimality: 4.02e-14)
    # [28.06499195321056, 218.22083546140868, 1445.1523421985046, -3.0592250575099853, -30.254612594859292, -0.07341654882261767, 0.0004037423358959923, 6380092.7952426905, 6356078.9633, 1.0006117622235415, 1.0034130802898595, 0.9993955272861204]

    data = read_data()
    params = [28.06499195321056, 218.22083546140868, 1445.1523421985046, -3.0592250575099853, -30.254612594859292, -0.07341654882261767, 0.0004037423358959923, 6380092.7952426905, 6356078.9633, 1.0006117622235415, 1.0034130802898595, 0.9993955272861204]
    proj = sjtsk_etrs(params, data)
    np.savetxt("test_krovak.csv", proj, delimiter=',')



    #solve() #Etrs->JTSK
    # solve 1 'trf' (cost: 12.9078, optimality: 5671, nfev: 120, njev: 109)
    # [-429.488833182965, -146.21611192856062, -493.1184165654901, 7.16243397288447, 5.82596300119802, 4.142615472464401, -0.00015253170138060712]
    # solve 2 'ln' (cost: 13.44501, optimality: 29157, nfev: 133, njev: none)
    # [-453.30730283206134, -173.73119427786793, -479.62374478129004, 8.06764210552825, 4.711735652545145, 3.8062959263157436, -0.0005784955687783325]
    # solve 3 'trf' loss 'soft_l1' (cost:18.19089, optimality: 158828, nfev: 18, njev: 15)
    # [-484.94018892804786, -169.4493713579013, -483.81351285720643, 8.206871127068753, 3.3829925181764406, 4.1669708624211195, -0.001588210986215286]
    # solve 4 'trf', loss 'huber', ftol 1e-8 (cost: 18.736, optimality: 275, nfev: 49, njev: 35)
    # [-4.84118878e+02, -1.69192992e+02, -4.82911860e+02,  8.19587445e+00,\n        3.40950913e+00,  4.16365198e+00, -1.54979742e-03]
    # solve 5 'xtol=ftol=1e-10' (cost:12.90776, optimality: 3026, nfev: 125, njev: 111, xtol)
    # [-4.29488833e+02, -1.46216112e+02, -4.93118417e+02,  7.16243473e+00,\n        5.82596613e+00,  4.14261757e+00, -1.52515703e-04]
    # solve 5 '(xtol=1e-12, ftol=1e-10, optimality: 336, nfev: 130, njev: 112)'
    # [-4.29488833e+02, -1.46216112e+02, -4.93118417e+02,  7.16243473e+00,\n        5.82596613e+00,  4.14261757e+00, -1.52503436e-04]

def samples():
    # JTSK->ETRS podla transformacnej sluzby UGKK https://zbgis.skgeodesy.sk/rts/sk/Transform
        # jstk(y, x) = (559995.88, 1231908.35)
        # bessel_1841(phi, lam) = (48.594652025, 17.227483411)
        # etrs(X, Y, Z) = (4036838.700, 1251627.545, 4760860.054) pre h = 0
        # etrs(phi, lam) = (48.594138919, 17.226123658)
        # jtsk03(y, x) = (559996.697, 1231907.749)

    etrs = [[48.5941389417, 17.226123647222], [48.2704154056, 17.606803761111]]
    jtsk = [[1231908.35, 559995.88], [1270462.82, 535445.82]]
    index = 0
    phi, lam = np.radians(etrs[index]) 
    x, y = jtsk[index]
    h = 331.909

    jtsk03 = Sjtsk.etrs_sjtsk(phi, lam, h)
    etrs_plh = np.degrees(Sjtsk.sjtsk_etrs(x, y, 0))
    print(jtsk03)
    print(etrs_plh)

    diff = np.sqrt((jtsk[index][0] - jtsk03[0])**2 + (jtsk[index][1] - jtsk03[1])**2)
    diff = np.sqrt((etrs_plh[0]-etrs[index][0])**2 + (etrs_plh[1]-etrs[index][1])**2)
    print(diff)

#samples()
param_estimation()