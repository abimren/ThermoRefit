"""
*ThermoRefit v2.2
 author: abdurrahman.imren @ WA,USA Nov 2021
*
*chem.inp (we read only species information) and therm.dat files are read.
*Thermodynamic data are specified in two temperature ranges with seven
*coefficients for each (Tmin,Tmid,Tmax). Standard-state molar heat capacities
*at constant pressure, molar enthalpy, and molar entropy are calculated using
*expressions:
*cp/R=a1+a2T+a3T^2+a4T^3+a5T^4
*H/RT=a1+(a2/2)T+(a3/3)T^2+(a4/4)T^3+(a5/5)T^4+a6/T 
*S/R=a1*lnT+a2T+(a3/2)T^2+(a4/3)T^3+(a5/4)T^4+a7
*
*Two types of discontinuity checks are tested 
*1- if discontinuity exists at specie mid temperature (called Tbreak)
*2- if discontinuity exists at common mid temperature (called Tmid) 
*   when Tbreak is different than Tmid. This test is specifically
*   for OpenFOAM (or other CFD software where only common Tmid is used and
*   specie specific temperature inputs are ignored). Here, we check if
*   extrapolated curves (from Tbreak to Tmid) would create artificial
*   discontinuity. To use this feature: set TmidRefit=True @ line 71-72
*
*If 'refit' is decided: common Tmid value is chosen as the new mid temperature    
*Linear solver is 'dgglse' from SciPy LAPACK interface. Implementation is low
*level. Applying equality constraints and specific index treatments of the
*variables at midpoint temperature values made high level coding be difficult.
*Contributions are welcome.
*Equality Constraints:
* 1. cp(Tmin) 2. cp(Tmax) 3. C0-cp(Tmid) 
* 4. C1-cp(Tmid) 5. C2-cp(Tmid)
*For H(Tmid) and S(Tmid), only C0 continuity is applied (a6 and a7 calculations) 
*
*ThermoRefit was written in Python 3.9 using NumPy 1.21.1, SciPy 1.7.0,
*and Matplotlib 3.4.2 packages. User inputs:
* 1. chem.inp and therm.dat files
* 2. TmidRefit=True or False @ line 71-72
* 3. atol and rtol values can be set @ line 133. Default value is 1e-4
* Output files are newtherm.dat and therm_duplicates.dat.
* Only species listed in chem.inp are written in newtherm.dat
* Plots of refitted species are collected in 'Plots' folder.
*
* Features:
* - while reading chem.inp, duplicates are not allowed (we are using set())
* - can read small letters and UTF-8 characters
* - we read chem.inp in order to pull out only listed species from therm.dat 
* - if specie in chem.inp can't be found in therm.dat, we raise 'ERROR' 
* - while reading therm.dat, duplicates are updated with the first entry
* - if there exists any missing specie temperature data (Tmin,Tmax,Tbreak)
*   we fill it with given common temperature values
* - if specie phase is not given or position changed, we do not fill it with 'G'.
    We raise 'ERROR' to correct its position and manually check given data
    to ensure that neighbor temperature inputs are OK
* - while reading coefficients, we strictly apply 2 temperature-ranges
    If extra coefficient line is found, 'WARNING' is raised and that line is ignored. 
    If there exists a missing line or coefficient or can't read any
    coefficient, we raise 'ERROR'
* - if specie Tbreak >= Tmax, corresponding data is flagged to 'no refit'
* - with TmidRefit=True option, we write in OpenFOAM CK interpreter format 
*   in which no exclamation marks (or any user entry) are accepted after column 80
*   (although this fix may not necessarily provide an exact OpenFOAM CK format,
*    it reduces compatibility issues significantly)
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lapack
from os.path import exists as file_exists

# User option for refit if Tmid common .ne. specie Tmid (Tbreak)
# True is for OpenFOAM simulations where only common Tmid is used
#TmidRefit=False
TmidRefit=True
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False  
# Functions of cp, h and s
def calc_cphs(T,a):
  cp=0
  h=a[5]/T
  s=a[6]+a[0]*np.log(T)
  for i in range(0,5):
    cp=cp+a[i]*T**i
    h=h+a[i]*(T**i)/(i+1)
    if i>0:
      s=s+a[i]*(T**i)/i
  return cp,h*T,s
  
# ***************************************************************************** 
# Check therm.dat exists  
if not file_exists('therm.dat'):
  print("Please ensure that therm.dat exists!")
  quit()
# Check chem.inp exists 
if not file_exists('chem.inp'):
  print('Please ensure that chem.inp exists!')
  quit()
chem = open('chem.inp','r',encoding='utf-8',errors='ignore')
specieList=set()
specieRead=False
for line in chem:
  if (line.lstrip().startswith('END') or line.lstrip().startswith('end')) and specieRead:
    print("Number of species:",len(specieList))
    #print(specieList)
    print("END of reading chem.inp\n")
    break 
  if line.strip() and not line.lstrip().startswith('!'):
    if specieRead:
      for sp in line.split():
        if sp.startswith('!'):
          break
        specieList.add(sp)
    if (line.lstrip().startswith('SPECIES') or \
        line.lstrip().startswith('species')):
      print("SPECIES data is being read")
      specieRead=True
chem.close()   
# Set 'plots' directory
script_dir=os.path.dirname(__file__)
results_dir=os.path.join(script_dir,'plots/')
if not os.path.isdir(results_dir):
  os.makedirs(results_dir) 
# Read therm.dat, open newtherm.dat
th = open('therm.dat','r',encoding='utf-8',errors='ignore')
newth = open('newtherm.dat','w')
#newth = open('newtherm.dat','a')
Tmin=Tmid=Tmax=Tbreak=0.0
al=np.zeros(7); ah=np.zeros(7);
ThermoDict = {}
duplist = []
atol=1e-4;rtol=1e-4
count = 0 
thrm=False
spRead=True
skipSpecie=False
for line in th:
  line=line[0:79]
  if line.lstrip().startswith('END') or line.lstrip().startswith('end'):
    print("END of reading therm.dat\n")
    break 
  if line.strip() and not line.lstrip().startswith('!'):
    if (line.lstrip().startswith('THERMO') or \
        line.lstrip().startswith('thermo')) and not thrm:
      print("THERMO data is being read")
      line=next(th)
      Tmin,Tmid,Tmax = [float(value) for value in line.split()]
      print("Tmin",Tmin,"Tmid",Tmid,"Tmax",Tmax)
      newth.writelines("THERMO ALL\n")
      newth.writelines(line)
      thrm=True
      continue
    if thrm and count == 0:
       coeffs=[line[index:index+15] for index in range(0,len(line),15)]
       coeffs=[item.replace(" ","") for item in coeffs]
       floatSp=[False]*3
       for i in range(0,3):
          floatSp[i] = isfloat(coeffs[i])  
       if all(floatSp):
         print("\n",line)
         print("WARNING - This line is not the one expected with the specie definition. Skipping!\n")
         continue      
    #if line[44] == " " and thrm and count == 0:
    #  specie=line.split()[0]
    #  print(specie,"phase input was not given. Set as G")
    #  line=line[:44]+'G'+line[45:79]        
    if ("G" in line[44] or "L" in line[44] or "S" in line[44] or \
        "g" in line[44] or "l" in line[44] or "s" in line[44]) and thrm and spRead:      
      # first check if any missing coeff line exists
      spRead=False
      print(line)
      #specie=line[0:12].rstrip()
      # read specie name until whitespace 
      specie=line.split()[0]
      count += 1
      if not specie in specieList:
        print("Not present in chem.inp. Skip specie",specie)
        skipSpecie=True
        continue   
      if not specie in ThermoDict:   
        ThermoDict.setdefault(specie,[])
      else:
        print("'",specie,"'-WARNING: Duplicate specie was detected. The first listed will be kept")
        if not specie in duplist:
          duplist.append(specie)
        skipSpecie=True  
        continue  
        #ThermoDict.pop(specie)
        #ThermoDict.setdefault(specie,[])
        #quit()
      #[0]:"true" no need to refit - "false" refit coeffs of corresponding specie
      #[1]: line contains specie and temperature info
      #[2][0:7]: high T coefficients
      #[3][0:7]: low T coefficients 
      #[4][0:7]: keep old high T coefficients
      #[5][0:7]: keep old low T coefficients 
      #[6][0:3]: Tbreak,TminSpecie,TmaxSpecie
      ThermoDict[specie].append("true")
      #ThermoDict[specie].append(line)
      setTstr=False
      if not line[66:75].strip():
        Tbreak=Tmid
        print(specie,"- Tbreak is not given. Set Tbreak as",Tbreak,"K")
        setTstr=True
      else:  
        Tbreak=float(line[66:75])     
      if Tbreak != Tmid and TmidRefit:
        print(specie,"- Tbreak(",Tbreak,"K) is different from Tmid. Check disc at common Tmid:",Tmid,"K!")
        #ThermoDict[specie][0]="false"     
      if not line[48:57].strip():
        TminSpecie=Tmin
        print(specie,"- Tmin is not given. Set Tmin as",Tmin,"K")
        setTstr=True
      else:  
        TminSpecie=float(line[48:57])
      if not line[57:66].strip():
        TmaxSpecie=Tmax
        print(specie,"- Tmax is not given. Set Tmax as",Tmax,"K")
        setTstr=True
      else:  
        TmaxSpecie=float(line[57:66])
      line=line.rstrip()
      line=line.ljust(79)      
      if setTstr:
        sT=str(TminSpecie).ljust(9)+str(TmaxSpecie).ljust(9)+str(Tbreak).ljust(9)
        line=line[:45]+'   '+sT+'    '
      # OpenFoam does not accept any char at line[78]
      if TmidRefit:
        line=line[:78]+" "
      ThermoDict[specie].append(line)
      #if TminSpecie != Tmin or TmaxSpecie != Tmax:
      #  print(specie,"- Tmin,Tmax(",TminSpecie,"-",TmaxSpecie,"K) are diff from common T. Refit to:",Tmin,"-",Tmax,"K")
      #  ThermoDict[specie][0]="false"
    elif thrm and count >= 1:   
      #print("Reading polynomial coefficients - line:",count)  
      # coeffs are stored in 15 fixed columns
      coeffs=[line[index:index+15] for index in range(0,len(line),15)]
      coeffs=[item.replace(" ","") for item in coeffs]
      if count == 1:
        count +=1
        if skipSpecie:
          #print("skip coeff line 1 of",specie)
          continue          
        for i in range(0,5):
          if isfloat(coeffs[i]):
            ah[i] = float(coeffs[i])
          else:
            print(specie,"-ERROR in reading polynomial coefficients. Check if any missing line or typo!")
            print("\tcoeffs are stored in 15 fixed columns")
            quit()
      elif count == 2:
        count +=1
        if skipSpecie:
          #print("skip coeff line 2 of",specie)
          continue
        for i in range(0,2):
          if isfloat(coeffs[i]):
            ah[i+5] = float(coeffs[i])
          else:
            print(specie,"-ERROR in reading polynomial coefficients. Check if any missing line or typo!")
            print("\tcoeffs are stored in 15 fixed columns")
            quit()  
        for i in range(0,3):
          if isfloat(coeffs[i]):
            al[i] = float(coeffs[i+2])
          else:
            print(specie,"-ERROR in reading polynomial coefficients. Check if any missing line or typo!")
            print("\tcoeffs are stored in 15 fixed columns")
            quit()
      elif count == 3:
        # reset count and spRead for the next specie
        count = 0
        spRead=True
        if skipSpecie:
          #print("skip coeff line 3 of",specie)
          skipSpecie=False
          continue      
        for i in range(0,4):
          if isfloat(coeffs[i]): 
            al[i+3] = float(coeffs[i])
          else:
            print(specie,"-ERROR in reading polynomial coefficients. Check if any missing line or typo!")
            print("\tcoeffs must be stored in 15 fixed columns")
            quit()  
        ThermoDict[specie].append([ah[0],ah[1],ah[2],ah[3],ah[4],ah[5],ah[6]])
        ThermoDict[specie].append([al[0],al[1],al[2],al[3],al[4],al[5],al[6]])
        # Keep old coefficients for use in plotting
        ThermoDict[specie].append([ah[0],ah[1],ah[2],ah[3],ah[4],ah[5],ah[6]])
        ThermoDict[specie].append([al[0],al[1],al[2],al[3],al[4],al[5],al[6]])
        # Save TminSpecie,TmaxSpecie,Tbreak
        ThermoDict[specie].append([Tbreak,TminSpecie,TmaxSpecie])
        # Sometimes Tmid and Tmax are given equal. Leave specie data as is
        if Tbreak >= TmaxSpecie-atol or TmaxSpecie <= Tmid:
          print(specie,"- Tbreak(",Tbreak,"K) is greater or equal to Tmax(",TmaxSpecie,"K). No refit!")
          print(specie,"- or TmaxSpecie(",TmaxSpecie,"K) is smaller than Tmid(",Tmid,"K). No refit!")
          ThermoDict[specie][0]="true"
        else:  
          # Flag specie with discontinuity at mid point or specie if Tbreak \= Tmid
          # OpenFoam uses common Tmid (species Tbreak ignored)
          Tb=Tbreak
          if TmidRefit:
            Tb=Tmid  
          cplow,hTlow,slow = calc_cphs(Tb,al) 
          cphigh,hThigh,shigh = calc_cphs(Tb,ah)
          disc_cp=abs(cplow-cphigh);disc_h=abs((hTlow-hThigh)/Tb);disc_s=abs(slow-shigh);
          rel_cp=abs((cplow-cphigh)/cphigh);rel_h=abs((hTlow-hThigh)/hThigh);rel_s=abs((slow-shigh)/shigh);
          atolfit=False
          if disc_cp > atol or disc_h > atol or disc_s > atol:
            print (specie,"- ATOL >",atol,": Discontinuity exists at Tb(",Tb,"K)")
            print ("\tCp discontinuity abs err:",disc_cp)
            print ("\tH discontinuity abs err:",disc_h)
            print ("\tS discontinuity abs err:",disc_s)
            ThermoDict[specie][0]="false"
            atolfit=True
          if (rel_cp > rtol or rel_h > rtol or rel_s > rtol) and not atolfit:
            print (specie,"- RTOL >",rtol,": Discontinuity exists at Tb(",Tb,"K)")
            print ("\tCp discontinuity rel err:",rel_cp)
            print ("\tH discontinuity rel err:",rel_h)
            print ("\tS discontinuity rel err:",rel_s)
            ThermoDict[specie][0]="false"                  
    else:
      if count == 0:
        specie=line.split()[0]
        print(line)
        print("Column45:'",line[44],"'")
        print("'",specie,"'-ERROR: Phase input was not given properly. Set as G,L or S at column 45")
        quit()
      print("There is an undetected reading failure")
      quit()
# Reading is done
th.close()
# Write duplicates if any found
if len(duplist) > 0:
  with open('therm_duplicates.dat','w') as dup:
    for sp in duplist:
      dup.write(sp+'\n')
# Check if all species are available in thermdat
specieSet=set()
for specie in specieList:
  if not specie in ThermoDict:
    print("'",specie,"'-ERROR: Specie is N/A in therm.dat file!")
    specieSet.add(specie)
    if len(specieSet) > 1:
      quit()
# Refit species and write into newtherm.dat
for specie in ThermoDict:
  if ThermoDict[specie][0] == "true":
    print("Good to go with",specie,"- Writing into newtherm.dat")
    newth.writelines(ThermoDict[specie][1]+"1\n")  
    sl=""
    for j in range(0,5):
      sl += f"{ThermoDict[specie][4][j]:15.8E}"
    newth.writelines(sl+"2\n".rjust(6))
    sl=""
    for j in range(5,7):
      sl += f"{ThermoDict[specie][4][j]:15.8E}"
    for j in range(0,3):
      sl += f"{ThermoDict[specie][5][j]:15.8E}"
    newth.writelines(sl+"3\n".rjust(6))
    sl=""
    for j in range(3,7):
      sl += f"{ThermoDict[specie][5][j]:15.8E}"
    newth.writelines(sl+"4\n".rjust(21))  
  else:
    print("Refitting -",specie)
    # Set specie Tmin and Tmax
    Tmin = ThermoDict[specie][6][1]
    Tmax = ThermoDict[specie][6][2] 
    print("Tmin,Tmax",Tmin,Tmax)
    # Define temperature interval; ncoeff is 5+5 for cp high and low coeffs
    dT=25; ncoeff=10;
    m=int((Tmax-Tmin)/dT)+2
    # m contains elements between two temperature-ranges
    mm=m-1
    # number of equality constraints
    eq=5
    ac = np.zeros(m*ncoeff).reshape(m,ncoeff); cc = np.zeros(m)
    bc = np.zeros(eq*ncoeff).reshape(eq,ncoeff); dc = np.zeros(eq)
    # Calculate cp for T range
    cpp = np.zeros(mm); hTp = np.zeros(mm); sp = np.zeros(mm)
    T = Tmin
    Tbreak = ThermoDict[specie][6][0]
    for i in range(0,mm):
      if T <= Tbreak:
        cpp[i],hTp[i],sp[i] = calc_cphs(T,ThermoDict[specie][5])
      else:
        cpp[i],hTp[i],sp[i] = calc_cphs(T,ThermoDict[specie][4])
      T += dT
    # Solve least squares problem
    # min ||a @ x - c||
    # subject to  b @ x = d
    # define a,c and b,d for a1-a5    
    T = Tmin
    i=0
    while T <= Tmid:
      for j in range(0,5):
        ac[i][j]=T**j
      cc[i] = cpp[i]
      i += 1
      T += dT  
    T=Tmid
    while T <= Tmax:
      for j in range(0,5):
        ac[i][j+5]=T**j
      cc[i] = cpp[i-1]
      i += 1
      T += dT    
    # Here are 5 equality constraints
    for j in range(0,5):
      bc[0][j]=Tmin**j
      bc[2][j]=Tmid**j
      bc[3][j]=j*Tmid**(j-1)
      bc[4][j]=(j-1)*j*Tmid**(j-2)
      bc[1][j+5]=Tmax**j
      bc[2][j+5]=-bc[2][j]
      bc[3][j+5]=-bc[3][j]
      bc[4][j+5]=-bc[4][j]
    dc[0]= cpp[0]
    dc[1]= cpp[mm-1]
    # Solve x
    x = lapack.dgglse(ac, bc, cc, dc)[3]
    for j in range(0,5):
      ThermoDict[specie][3][j]=x[j]
      ThermoDict[specie][2][j]=x[j+5]
    # Coefficient a6
    ncoeff=2;eq=1; 
    ac = np.zeros(m*ncoeff).reshape(m,ncoeff)
    cc.fill(0)
    bc = np.zeros(eq*ncoeff).reshape(eq,ncoeff)
    dc = np.zeros(eq)
    # Fill the coefficient matrix    
    T = Tmin
    i=0
    while T <= Tmid:
      T2=T**2/2.0;T3=T**3/3.0;T4=T**4/4.0;T5=T**5/5.0
      ac[i][0]=1.0
      cc[i] = hTp[i]-(x[0]*T+x[1]*T2+x[2]*T3+x[3]*T4+x[4]*T5)
      i += 1
      T += dT  
    T=Tmid
    while T <= Tmax:
      T2=T**2/2.0;T3=T**3/3.0;T4=T**4/4.0;T5=T**5/5.0
      ac[i][1]=1.0
      cc[i] = hTp[i-1]-(x[5]*T+x[6]*T2+x[7]*T3+x[8]*T4+x[9]*T5)
      i += 1
      T += dT     
    # Equality constraints
    bc[0][0]=1.0
    bc[0][1]=-1.0
    dc[0]=0.0
    for j in range(0,5):
      dc[0]= dc[0]+(x[j+5]-x[j])*Tmid**(j+1)/(j+1)
    # Solve y
    y = lapack.dgglse(ac, bc, cc, dc)[3]
    ThermoDict[specie][3][5]=y[0]
    ThermoDict[specie][2][5]=y[1]
    # Coefficient a7
    ncoeff=2;eq=1;
    ac = np.zeros(m*ncoeff).reshape(m,ncoeff)
    cc.fill(0)
    bc = np.zeros(eq*ncoeff).reshape(eq,ncoeff)
    dc = np.zeros(eq)
    # Fill the coefficient matrix    
    T = Tmin
    i=0
    while T <= Tmid:
      T2=T**2/2.0;T3=T**3/3.0;T4=T**4/4.0;
      ac[i][0]=1.0
      cc[i] = sp[i]-(x[0]*np.log(T)+x[1]*T+x[2]*T2+x[3]*T3+x[4]*T4)
      i += 1
      T += dT  
    T=Tmid
    while T <= Tmax:
      T2=T**2/2.0;T3=T**3/3.0;T4=T**4/4.0;
      ac[i][1]=1.0
      cc[i] = sp[i-1]-(x[5]*np.log(T)+x[6]*T+x[7]*T2+x[8]*T3+x[9]*T4)
      i += 1
      T += dT 
    # Equality constraints
    bc[0][0]=1.0
    bc[0][1]=-1.0
    dc[0]=(x[5]-x[0])*np.log(Tmid)
    for j in range(1,5):
      dc[0]= dc[0]+(x[j+5]-x[j])*Tmid**j/j
    # Solve z
    z = lapack.dgglse(ac, bc, cc, dc)[3]
    ThermoDict[specie][3][6]=z[0]
    ThermoDict[specie][2][6]=z[1]
    # Discontinuity check at mid point
    cplow,hTlow,slow = calc_cphs(Tmid,ThermoDict[specie][3]) 
    cphigh,hThigh,shigh = calc_cphs(Tmid,ThermoDict[specie][2])
    disc_cp=abs(cplow-cphigh);disc_h=abs((hTlow-hThigh)/Tmid);disc_s=abs(slow-shigh);
    #print("disc",disc_cp,disc_h,disc_s)
    if disc_cp > atol or disc_h > atol or disc_s > atol:
        print (specie,"- Discontinuity exists at new Tmid > atol:",atol)
        ThermoDict[specie][0]="false"
    ThermoDict[specie][0]="true"
    # Write refitted specie data
    print("Tbreak,Tmid",Tbreak,Tmid)
    line = ThermoDict[specie][1]
    sT=str(Tmin).ljust(9)+str(Tmax).ljust(9)+str(Tmid).ljust(9)
    line = line.replace(line[48:75],sT)
    ThermoDict[specie][1]=line
    if Tbreak != Tmid:
      sTorig=str(ThermoDict[specie][6][0])
      if TmidRefit:
        newth.writelines(ThermoDict[specie][1]+"1\n")
      else:
        newth.writelines(ThermoDict[specie][1]+"1!RF Tmid "+sTorig+"\n")
    else:
      if TmidRefit:
        newth.writelines(ThermoDict[specie][1]+"1\n")
      else:  
        newth.writelines(ThermoDict[specie][1]+"1!RF\n")
    sl=""
    for j in range(0,5):
      sl += f"{ThermoDict[specie][2][j]:15.8E}"
    newth.writelines(sl+"2\n".rjust(6))
    sl=""
    for j in range(5,7):
      sl += f"{ThermoDict[specie][2][j]:15.8E}"
    for j in range(0,3):
      sl += f"{ThermoDict[specie][3][j]:15.8E}"
    newth.writelines(sl+"3\n".rjust(6))
    sl=""
    for j in range(3,7):
      sl += f"{ThermoDict[specie][3][j]:15.8E}"
    newth.writelines(sl+"4\n".rjust(21))
    # Plot refit
    # low and high T curves: calculate with the original coefficients
    T1 = np.arange(Tmin,Tbreak,dT)
    if T1[-1] != Tbreak:
      T1=np.append(T1,Tbreak)
    T2 = np.arange(Tbreak,Tmax,dT)
    if T2[-1] != Tmax:
      T2=np.append(T2,Tmax)
    m=len(T1);n=len(T2)
    cpl = np.zeros(m); hTpl = np.zeros(m); spl = np.zeros(m)
    cph = np.zeros(n);hTph = np.zeros(n); sph = np.zeros(n)
    for i in range(0,m):
      cpl[i],hTpl[i],spl[i] = calc_cphs(T1[i],ThermoDict[specie][5])
    for i in range(0,n):
      cph[i],hTph[i],sph[i] = calc_cphs(T2[i],ThermoDict[specie][4])
    # Calculate with the new ones
    T1n = np.arange(Tmin,Tmid,dT)
    if T1n[-1] != Tmid:
      T1n=np.append(T1n,Tmid)
    T2n = np.arange(Tmid,Tmax,dT)
    if T2n[-1] != Tmax:
      T2n=np.append(T2n,Tmax)
    m=len(T1n);n=len(T2n)
    cpln = np.zeros(m); hTpln = np.zeros(m); spln = np.zeros(m)
    cphn = np.zeros(n);hTphn = np.zeros(n); sphn = np.zeros(n)
    cplo = np.zeros(m); hTplo = np.zeros(m); splo = np.zeros(m)
    cpho = np.zeros(n);hTpho = np.zeros(n); spho = np.zeros(n)
    for i in range(0,m):
      cpln[i],hTpln[i],spln[i] = calc_cphs(T1n[i],ThermoDict[specie][3])
      cplo[i],hTplo[i],splo[i] = calc_cphs(T1n[i],ThermoDict[specie][5])
    for i in range(0,n):
      cphn[i],hTphn[i],sphn[i] = calc_cphs(T2n[i],ThermoDict[specie][2])
      cpho[i],hTpho[i],spho[i] = calc_cphs(T2n[i],ThermoDict[specie][4])
    # calculate RMS errors
    err1=np.sum(((cpln-cplo)/cpln)**2)
    err1=err1+np.sum(((cphn-cpho)/cphn)**2)
    err1=np.sqrt(err1/(m+n))    
    err2=np.sum(((hTpln-hTplo)/hTpln)**2)
    err2=err2+np.sum(((hTphn-hTpho)/hTphn)**2)
    err2=np.sqrt(err2/(m+n))   
    err3=np.sum(((spln-splo)/spln)**2)
    err3=err3+np.sum(((sphn-spho)/sphn)**2)
    err3=np.sqrt(err3/(m+n))
    print(f"{specie} RMS errors of CP: {err1:9.4E},H: {err2:9.4E},S: {err3:9.4E}") 
    # set matplotlib    
    fig, ax = plt.subplots()
    if TmidRefit:
       ax.plot(T1n,cplo,'-k',T2n,cpho,'-b')
    else:        
       ax.plot(T1,cpl,'-k',T2,cph,'-b')
    ax.plot(T1n,cpln,'--m',T2n,cphn,'--r')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True,direction='in') 
    ax.tick_params(which='minor',top=True,right=True,direction='in')
    ax.set(xlabel='Temperature (K)', ylabel='Cp/R')
    ax.grid(linestyle=':')
    ax.legend(['Orig-low T','Orig-high T','Refit-low T','Refit-high T'])
    fig.savefig(results_dir+"CPR_"+specie+".png")
    plt.cla()
    if TmidRefit:
       ax.plot(T1n,hTplo,'-k',T2n,hTpho,'-b')
    else:        
       ax.plot(T1,hTpl,'-k',T2,hTph,'-b')    
    ax.plot(T1n,hTpln,'--m',T2n,hTphn,'--r')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True,direction='in') 
    ax.tick_params(which='minor',top=True,right=True,direction='in')
    ax.set(xlabel='Temperature (K)', ylabel='H/RT')
    ax.grid(linestyle=':')
    ax.legend(['Orig-low T','Orig-high T','Refit-low T','Refit-high T'])
    fig.savefig(results_dir+"HRT_"+specie+".png")
    plt.cla()
    if TmidRefit:
       ax.plot(T1n,splo,'-k',T2n,spho,'-b')
    else:        
       ax.plot(T1,spl,'-k',T2,sph,'-b')   
    ax.plot(T1n,spln,'--m',T2n,sphn,'--r')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True,direction='in') 
    ax.tick_params(which='minor',top=True,right=True,direction='in')
    ax.set(xlabel='Temperature (K)', ylabel='S/R')
    ax.grid(linestyle=':')
    ax.legend(['Orig-low T','Orig-high T','Refit-low T','Refit-high T'])
    fig.savefig(results_dir+"SR_"+specie+".png")
    #plt.show()
    plt.close('all')
  #print(specie,ThermoDict[specie])    
newth.writelines("END\n")
newth.close()
