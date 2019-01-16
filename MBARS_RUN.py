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
#filename = 'ESP_011357_2285_RED300PX'
#filename = 'PSP_001391_2465_RED500PX'
plot = True
#for continuing broken runs, use Startat to specify which panel to begin on for the first run
startat = 0
# parameters
gams = [.6,.6]
bounds = [.10,.11]

def run(filename,gams,bounds,plot,startat):

    root, MBARS.ID, MBARS.NOMAP,panels = MBARS.RunParams(filename)

    MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(root)
    MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ, MBARS.ROTANG = MBARS.start()

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
            if i%500 == 0:
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
    return

run(filename,gams,bounds,plot,startat)




