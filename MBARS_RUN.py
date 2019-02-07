import MBARS
import matplotlib.pyplot as plt
import scipy.misc as spm
import numpy as np
import time
import cPickle as pickle
import threading
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

filenameA = 'TRA_000828_2495_RED500PX'
filenameB = 'ESP_011357_2285_RED300PX'
filenameC = 'PSP_001391_2465_RED500PX'
filenameD = 'PSP_007718_2350_RED300px'
#filename = 'ESP_036437_2290_RED500PX'
filenames  = [filenameA,filenameB,filenameC,filenameD]
plot = False
#for continuing broken runs, use Startat to specify which panel to begin on for the first run
#does not do anything for threaded runs as they are not necessarily linear
startat = 0
# parameters
gams = [.6]
bounds = [.10]

#experimental, 100 seems to work, no limit causes memory errors.
thread_limit = 100

def run(filename,gams,bounds,plot,startat):
    
    mangam, manbound = MBARS.FindIdealParams(filename)

    MBARS.FNM, MBARS.ID, MBARS.NOMAP,panels = MBARS.RunParams(filename)

    MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(MBARS.FNM)
    MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ, MBARS.ROTANG = MBARS.start()

    for j in range(len(gams)):
        t1 = time.clock()
        gam = gams[j]
        bound = bounds[j]
        for i in range(startat,panels):
            #MBARS.FNM = '%s%s'%(root, i)
            mod, seg, good, runfile = MBARS.gamfun(i,mangam, plot, bound,manbound)
            #print 'step 1 done'
            if good:
                mod = None
                if any(seg.compressed()):
                    bads = MBARS.boulderdetect(i,seg,runfile)
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


def core(num,gam,plot,bound,manbound,odr_keycard):
    '''core function of the run file, does gamfun and boulder detect
    '''
    mod, seg, good, runfile = MBARS.gamfun(num,gam, plot, bound,manbound)
    #print 'step 1 done'
    if good:
        mod = None
        if any(seg.compressed()):
            bads = MBARS.boulderdetect_threadsafe(num,seg,runfile,odr_keycard)
    if num%200 == 0:
        print 'Done with image %s'%(num)
    return runfile

def thread_run(filename,gams,bounds,plot,startat):
    
    

    MBARS.FNM, MBARS.ID, MBARS.NOMAP,panels = MBARS.RunParams(filename)

    MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(MBARS.FNM)
    MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ, MBARS.ROTANG = MBARS.start()
    mangam,manbound = MBARS.FindIdealParams(filename,True)

    for j in range(len(gams)):
        t1 = time.clock()
        gam = gams[j]
        bound = bounds[j]
        threads = []
        krange = range(panels)
        #length = len(krange)
        #args = zip(krange,[mangam]*length,[plot]*length,[bound]*length,[manbound]*length)
        #args = [list(a) for a in args]
        #run core(i) function
        print '%s images to run'%(panels)
        odr_keycard = threading.Lock()
        threads = [threading.Thread(target = core, args=(a,mangam,plot,bound,manbound,odr_keycard)) for a in krange]
        for i in threads:
            runfile = i.start()
            while threading.active_count() > thread_limit:
                #print 'Waiting at %s on thread room'%(i.name)
                time.sleep(5)
        startat = 0
        for a in threads:
            a.join()
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
#setup all the parameters before running
for i in filenames:
    mangam, manbound = MBARS.FindIdealParams(i)
    
for i in filenames:
    thread_run(i,gams,bounds,plot,startat)




