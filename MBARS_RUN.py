import MBARS
import matplotlib.pyplot as plt
import scipy.misc as spm
import numpy as np
import time
import cPickle as pickle
'''
set number of images, this is expecting a series of images with the same root name
and numerals at the end, i.e. "root0.PNG" and "root23.PNG"
'''
##root = 'PSP_007718_2350_'
##MBARS.ID = 'PSP_007718_2350_RED'
##panels=3
##MBARS.NOMAP = True

##root = 'TRA_000828_2495_RED_300PX'
##MBARS.ID = 'TRA_000828_2495_RED'
##panels = 4965

##root = 'PSP_001501_2280_RED_NOMAP_300PX'
##MBARS.ID = 'PSP_001501_2280_RED'
##panels = 8977

##root = 'ESP_028612_1755_RED66_'
##MBARS.ID = 'ESP_028612_1755_RED'
##MBARS.NOMAP = False
##panels = 204
##
#filename = 'PSP_007718_2350_RED300px'
filename = 'TRA_000828_2495_RED500PX'
fullrun =False
multirun = True

root, MBARS.ID, MBARS.NOMAP,panels = MBARS.RunParams(filename)

MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(root)
MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ = MBARS.start()
#MBARS.SUN = [1.,1.]



#controls go here
gam = .6
plot = False
bound = .1
#MBARS.NOMAP = False
#for continuing broken runs, use Startat to specify which panel to begin on for the first run
startat = 0


#multirun parameters
gams = [.6,.6,.6]
bounds = [.09,.10,.11]
#CFA_results = []

if fullrun:
    MBARS.current()
    for i in range(startat,panels):
        MBARS.FNM = '%s%s'%(root, i)
        mod, seg, good, runfile = MBARS.gamfun(gam, plot, bound)
        #print 'step 1 done'
        if good:
            mod = None
            if any(seg.compressed()):
                bads = MBARS.boulderdetect(seg,runfile)
        else:
            print 'image %s has problems\n'%(i)
        if i%100 == 0:
            print '\n===========done %s out of %s==============\n'%(i,panels)
    
if multirun:

    for j in range(len(gams)):
        t1 = time.clock()
        gam = gams[j]
        bound = bounds[j]
        for i in range(startat,panels):
            MBARS.FNM = '%s%s'%(root, i)
            mod, seg, good, runfile = MBARS.gamfun(gam, plot, bound)
            #print 'step 1 done'
            if good:
                mod = None
                if any(seg.compressed()):
                    bads = MBARS.boulderdetect(seg,runfile)

            if i%100 == 0:
                print '\n===========done %s out of %s==============\n'%(i,panels)
            startat = 0
        t2 = time.clock()
        ttime = (t2-t1)/3600.
        print ('total time: '+str(ttime)+'hours')

        #this is to note the last running conditions:
        string = "This image was last run with the parameters gam = %s, plot = %s, bound=%s. \nIt took %s hours"%(gam, plot,bound,ttime)
        record = open('%s%s%s_runinfo.txt'%(MBARS.PATH,runfile,MBARS.FNM),'w')
        for item in string:
            record.write(item)
        record.close()
else:
    MBARS.FNM = root+"52"
    lina, linb, mod, seg = MBARS.gamfun(.5, True, 60, .01)
    lina = None
    linb = None
    mod = None
    print "finding boulders"
    im = imagio.imread(MBARS.PATH+MBARS.FNM+MBARS.FLT)
##    plt.figure(1)
##    plt.imshow(im[0:500], cmap = "binary_r")
##    plt.figure(2)
##    plt.imshow(seg[0:500])
##    plt.figure(3)
##    plt.imshow(mod[0:500], cmap = "rainbow")
##    plt.colorbar()
##    plt.show()
    MBARS.boulderdetect(seg)




#this is some spot code to make an image that I want
##
##load = open(MBARS.PATH+MBARS.FNM+'_shadows.shad', 'rb')
##for line in load:
##    x=1
##eof = load.tell()
##load = open(MBARS.PATH+MBARS.FNM+'_shadows.shad', 'rb')    
##while True:
##    dat = pickle.load(load)
##    dat.pointsplot()
##    if load.tell() == eof:
##        break
##plt.imshow(mod, cmap='binary_r')
##plt.show()

