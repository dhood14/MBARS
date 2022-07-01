import numpy as np
import scipy as sp
from scipy import optimize as spopt
import matplotlib.pyplot as plt
import MBARS

'''
This is for making CFAs from the output from arcGIS.
Works on data tables of the format used in "OutToGIS"
'''

#get the rectangle area from GIS (in square meters)
#AREA = 39740000
#AREA = float(AREA)
MIN_SIZE = 1
MAX_SIZE = 6
num_bins = 20

#fnm = 'PSP_001481_2410_RED16bit_500PX_All_boulderdata'
#ext = '.csv'

ext = '.txt'
#ext = '.csv'

#set wether using the auto or manual runs
Auto = 1
Manual = 0

#EXAMPLE ENTRY
#image = 'filename//'
#The above setting will send the program to BASEPATH//filename//GISFiles// to look for the output data


#image = 'PSP_007718_2350_//'
#image = 'PSP_007718_2350_RED_1000PX//'
#image = 'TRA_000828_2495_RED_16bit//'
#image = 'TRA_000828_2495_RED16bit_500PX//'
#image = 'PSP_001391_2465_RED16bit_1000PX//'
#image = 'PSP_001415_2470_RED16bit_1000PX//'
#image = 'PSP_001481_2410_RED16bit_500PX//'

#VL1 Images
#image = 'PSP_001521_2025_RED_1000PX//'
#image = 'PSP_001719_2025_RED_1000PX//'
#image = 'ESP_046170_2025_RED_1000PX//'

#VL2 Images
#image = 'PSP_001501_2280_RED_1000PX//'
#image = 'PSP_001976_2280_RED_1000PX//'
#image = 'PSP_002055_2280_RED_1000PX//'

#Y01-05
#image = 'PSP_001738_2345_RED_1000PX//'
#image = 'PSP_001785_2330_RED_1000PX//'
#image = 'PSP_007803_2375_RED_1000PX//'
#image = 'ESP_018211_2370_RED_1000PX//'
#image = 'PSP_009086_2360_RED_1000PX//'

#Y06-10
#image = 'PSP_010141_2360_RED_1000PX//'
#image = 'PSP_009604_2310_RED_1000PX//'
#image = 'PSP_001418_2495_RED_1000PX//'
#image = 'ESP_018139_2315_RED_1000PX//'

#Y11-15
#image = 'PSP_001482_2490_RED_1000PX//'
#image = 'PSP_001505_2485_RED_1000PX//'
#image = 'PSP_001559_2485_RED_1000PX//'
#image = 'PSP_001473_2480_RED_1000PX//'
#image = 'PSP_001430_2470_RED_1000PX//'

#Y21-25
#image = 'PSP_001721_2460_RED_1000PX//'
#image = 'PSP_001813_2455_RED_1000PX//'
#image = 'PSP_001681_2455_RED_1000PX//'
#image = 'PSP_001814_2455_RED_1000PX//'
image = 'PSP_001481_2410_RED_1000PX//'


#Specify the percentiles (suffix on the autobound_## file) and test area names
percs = ['10','20','30','40','50','60','70','80','85']
#percs = ['90','95','100']
#percs = ['30','40','50','60','70']
#percs = ['100']
#areas = ['A','B','C','D']
areas = ['A','B','C']
#areas = ['A','B']
#areas = ['1','2','3']

bnm = 'Clean'
#bnm = "All"

#Specfy the column where the widths are! 0 is the first column

#for ArcGIS Files
widcol = 6
fgcol = 10
areacol = 15

#only for error purposes
resolution = .25



PATH = MBARS.BASEPATH

#FULLPATH = '%s%sGISFiles//%s%s%s'%(PATH,image,runfile,fnm,ext)
#Auto Results
if Auto:
    Compile = True
    paramlist = []
    for perc in percs:
        for area in areas:
            runfile = 'autobound_%s//'%(perc)
            fnm = '%s_%s_%s'%(bnm,perc,area)
            fpath = '%s%sGISFiles//%s'%(PATH,image,runfile)
            paramlist+=[(fpath,fnm,widcol,fgcol,areacol,resolution)]
#ManualResults  
if Manual:
    Compile = False
    widcol = 3
    fgcol = 4
    areacol = 5
    #PATH = 'C://Users//Don_Hood//Documents//MBARS//ImagePrep//VL2//'
    #PATH = 'c://Users//Don_Hood//Documents//MBARS//RefData//PSP_007718_2350//'
    PATH = 'c://Users//Don_Hood//Documents//MDAP_2020_Polygons//Image_APRX_Files//Y1_21_25//Manual_Data//'
    #PATH = 'c://Users//Don_Hood//Documents//MDAP_2020_Polygons//Image_APRX_Files//Y1_01_05_ManualData//'
    paramlist = []
    paramlist+=[(PATH,'PSP_001481_2410_Manual_A',widcol,fgcol,areacol,resolution)]
    paramlist+=[(PATH,'PSP_001481_2410_Manual_B',widcol,fgcol,areacol,resolution)]
    paramlist+=[(PATH,'PSP_001481_2410_Manual_C',widcol,fgcol,areacol,resolution)]
    #paramlist+=[(PATH,'TRA_828_B_Manual_DH',widcol,fgcol,areacol,resolution)]
    #paramlist = []
    runfile=''

#Make a CFA
def GolomPSDCFA(D,k):
    ''' The Model curves used in Golombek's 2012 work, similar to those used in Li 2018
    k is fractional area covered by rocks
    this produces a CFA
'''
    q = 1.79 + .152/k
    F = k*np.exp(-q*D)
    return F
def fittoRA(xdat,ydat,RNG = [1.5,2.25]):
    '''Function to fit CFA results to a rock abundance
    Takes in bins as xdat, CFA as ydat, lower and upper bounds as RNG (assumed 1.5-2.25 to match Golombek 2012
    '''
    fit_xdat = []
    fit_ydat = []
    for i in range(len(xdat)):
        if xdat[i] <= RNG[1] and xdat[i] >= RNG[0]:
            fit_xdat+=[xdat[i]]
            fit_ydat+=[ydat[i]]
            #print(fit_ydat)
            #print(fit_xdat)
    try:        
        popt,pcov = spopt.curve_fit(GolomPSDCFA,fit_xdat,fit_ydat,p0=[.1])
    except:
        print('fit failed')
        return [False],0,0,0
        

    #calculate the R2 on the fit maybe??
    ybar = np.average(fit_ydat)
    SStot = np.sum((fit_ydat-ybar)**2)
    predicted = GolomPSDCFA(fit_xdat,popt)
    SSres = np.sum((fit_ydat-predicted)**2)
    R2 = 1-SSres/SStot
    minRA = 100
    maxRA = 0
    for i in range(len(fit_xdat)):
        x_pt = [fit_xdat[i]]
        y_pt = [fit_ydat[i]]
        if y_pt[0] <= 0:
            continue
        print(x_pt)
        print(y_pt)
        opt,cov = spopt.curve_fit(GolomPSDCFA,x_pt,y_pt,p0=[.1])
        RA = opt[0]
        if RA < minRA:
            minRA = RA
        if RA > maxRA:
            maxRA = RA
    
    plt.scatter(fit_xdat,fit_ydat)
    xrng = np.linspace(RNG[0],RNG[1])
    RACurve = GolomPSDCFA(xrng,popt[0])
    HighRACurve = GolomPSDCFA(xrng,maxRA)
    LowRACurve = GolomPSDCFA(xrng,minRA)
    plt.plot(xrng,RACurve)
    plt.plot(xrng,HighRACurve)
    plt.plot(xrng,LowRACurve)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    return popt, R2, minRA, maxRA

    
    
def run(path,fnm,widcol,fgcol,acol,resolution):
    outstring = '%s//%s_CFA.csv'%(path,fnm)
    fullpath = '%s%s%s'%(path,fnm,ext)
    DeadReturn = ('',[],[],None,None)
    with open(outstring,'w') as output:
        try:
            data = open(fullpath,'r')
        except(FileNotFoundError):
            print("%s not found"%(fullpath))
            return DeadReturn
        prefilt = np.loadtxt(data,delimiter=',',skiprows = 1,usecols = (widcol,fgcol,acol),ndmin=2)
        
        if len(prefilt)==0:
            #Input boulder file has no boulders, reject
            print("No Boulders in %s"%(fullpath))
            return DeadReturn
        
        AREA = prefilt[0][2]
        widths = list(prefilt)
        widths = [a[0] for a in widths if a[1] == 1.0]
        widths = list(map(float,widths))
        widths = np.array(widths)
        areas = np.pi*(widths/2.)**2
        area_sigs = (np.pi*resolution*.5*widths)**2
        #approximation of the 1.2 assumption, reality is it should be
        #+.2a and -.16a
        #area_sigs = .2*areas
        #getting uncertainties by assuming axial ratio of <1.2
        up_area = 1.2*areas
        d_area = areas/1.2
        bins = np.linspace(MIN_SIZE,MAX_SIZE,num_bins)
        areas = areas.tolist()
        widths = widths.tolist()
        sigs = area_sigs.tolist()
        a_w = list(map(list,zip(areas,widths,area_sigs)))
        CFAs = []
        up_CFAs = []
        d_CFAs = []
        CNRs = []
        CFAsigs = []
        for i in bins:
            areas_above = [a[0] for a in a_w if a[1]>i and a[1]<MAX_SIZE]
            up_areas = map(lambda x: x+.2*x,areas_above)
            d_areas = map(lambda x: x-.2*x,areas_above)
            sigs_prod = [(a[2]*a[0])**2 for a in a_w if a[1]>i and a[1]<MAX_SIZE]
            t_n = len(areas_above)
            t_area = sum(areas_above)
            f_area = float(t_area)/AREA
    
            f_up_area = float(sum(up_areas))/AREA
            f_d_area = float(sum(d_areas))/AREA
            
            CFA_sig = np.sqrt(sum(sigs_prod))/AREA
            f_n = t_n/AREA
            CFAs+=[f_area]
            up_CFAs+=[f_up_area-f_area]
            d_CFAs+=[f_area-f_d_area]
            CFAsigs+=[CFA_sig]
            CNRs+=[f_n]
        opt,R2,minRA,maxRA = fittoRA(bins,CFAs,[1.5,3])
        output.write('Image Area is %s square meters, Best fit RA:,%s, R2,%s,maxRA,%s,minRA,%s \n'%(AREA,opt[0],R2,maxRA,minRA))
        output.write('Bins (meters),')
        for i in bins:
            output.write('%s,'%(i))
        output.write('\nCFA,')
        for i in CFAs:
            output.write('%s,'%(i))
        output.write('\nCFA Sigma,')
        for i in CFAsigs:
            output.write('%s,'%(i))
        output.write('\nCNR per m^2,')
        for i in CNRs:
            output.write('%s,'%(i))
        output.write('\nUp CFAs,')
        for i in up_CFAs:
            output.write('%s,'%(i))
        output.write('\nDown CFAs,')
        for i in d_CFAs:
            output.write('%s,'%(i))
        results = (fnm,bins,CFAs,opt[0],R2)
        return results
oresults = []
for params in paramlist:
    fpath,fnm,widcol,fgcol,areacol,resolution = params
    oresult = run(fpath,fnm,widcol,fgcol,areacol,resolution)
    oresults+=[oresult]

if Compile:
    CompiledFile = '%s%sGISFiles//CFA_Compiled.csv'%(PATH,image)
    with open(CompiledFile,'w') as output:
        bins = False
        for ores in oresults:
            f,b,c,ra,r = ores
            if not bins:
                output.write('FNM,RA,R2,Diameter,')
                for i in b:
                    output.write('%s,'%(i))
                output.write('\n')
                bins = True
            output.write('%s,%s,%s,,'%(f,ra,r))
            for i in c:
                output.write('%s,'%(i))
            output.write('\n')
            
            # with open(file,'r') as info:
            #     output.write(file+'\n')
            #     count=0
            #     #trimmed the compile a bit to speed things up
            #     while count <4:
            #         line = info.readline()
            #         output.write(line)
            #         count+=1
            #     output.write('\n')
            
