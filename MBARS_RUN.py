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
filenames = []

#filename += 'ESP_011357_2285_RED300PX'


#filenameA = 'ESP_036437_2290_RED500PX'
#filenames  = [filenameB,filenameC,filenameD]

#GOlombek Comparison Images
filenames += ['TRA_000828_2495_RED500PX']
#filenames += ['PSP_001391_2465_RED500PX']

#viking 1 lander setup:
##filenameAA = 'PSP_001521_2025_RED100PNL47_500PX'
#filenameBB = 'PSP_001719_2025_RED100PNL52_500PX'
#filenameCC = 'ESP_046170_2025_RED_100PNL52_500PX'
#filenames = [filenameCC]

#viking 2 lander images
#filenameA = 'PSP_001501_2280_RED100PNL47_500PX'
#filenameB = 'PSP_001976_2280_RED100PNL52_500PX'
#filenameC = 'PSP_002055_2280_RED100PNL57_500PX'
#filenames = [filenameA]

#PSP_007718 subset images
#filenames = ['PSP_007718_2350_']
#filenames += ['PSP_007718_2350_RED300px']

#JoesImages
#filenames += ['PSP_007693_2300_RED500PX']

#Proposal Test Images
#filenames += ['PSP_001415_2470_RED500PX']


######SOME CONTROLS###################
#produce intermediate plots, False unless you are debugging something
plot = False

#for continuing broken runs, use Startat to specify which panel to begin on for the first run
#Keep in  mind that threaded runs do not complete the files in order, use with caution
startat = 0

#mostly deprecated, uses manually determined boundaries
### parameters
##gams = [.6]
##bounds = [.10]

#Process is largely processer-limited, so benefit to large number of threads is minimal
#setting no limit causes memory errors.
thread_limit = 2
#make true  if you want to run without threads
NOTHREADS = False

def run(filename,plot,startat):
    '''for running in a non-threaded fashion if desired'''
    #retrieve running parameters
    mangam, manbound = MBARS.FindIdealParams(filename)

    #setup paths and filenames
    MBARS.FNM, MBARS.ID, MBARS.NOMAP,panels = MBARS.RunParams(filename)
    MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(MBARS.FNM)
    MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ, MBARS.ROTANG = MBARS.start()

    t1 = time.clock()
    for i in range(startat,panels):
        runfile = core(i,mangam,plot,manbound,True)
        if i%500 == 0:
            print '\n===========done %s out of %s==============\n'%(i,panels)
        startat = 0
    t2 = time.clock()
    ttime = (t2-t1)/3600.
    print ('total time: '+str(ttime)+'hours')

    #this is to note the last running conditions:
    string = "This image was last run with the parameters mangam = %s, manbound=%s. \nIt took %s hours"%(mangam,manbound,ttime)
    record = open('%s%s%s_runinfo.txt'%(MBARS.PATH,runfile,MBARS.FNM),'w')
    for item in string:
        record.write(item)
    record.close()
    return


def core(num,gam,plot,manbound,odr_keycard):
    '''core function of the run file, does gamfun and boulder detect
    '''
    seg, good, runfile = MBARS.gamfun(num,gam, plot,manbound)
    #print 'step 1 done'
    if good:
        if any(seg.compressed()):
            bads = MBARS.boulderdetect_threadsafe(num,seg,runfile,odr_keycard)
            MBARS.overlapcheck_threadsafe_DBSCAN(num,runfile, odr_keycard, overlap=.05)
    if num%200 == 0:
        print 'Done with image %s'%(num)
    return runfile

def thread_run(filename,plot,startat):

    MBARS.FNM, MBARS.ID, MBARS.NOMAP,panels = MBARS.RunParams(filename)

    MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(MBARS.FNM)
    MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ, MBARS.ROTANG = MBARS.start()
    mangam,manbound = MBARS.FindIdealParams(filename,True)
    t1 = time.clock()
    
    threads = []
    krange = range(startat,panels)
    print '%s images to run'%(panels)
    odr_keycard = threading.Lock()
    threads = [threading.Thread(target = core, args=(a,mangam,plot,manbound,odr_keycard),name='%s'%(a)) for a in krange]
    count=0
    for i in range(len(threads)):
        runfile = threads[i].start()
        while threading.active_count() > thread_limit:
            #print 'Waiting at %s on thread room'%(i.name)
            time.sleep(5)
        # make sure the dragging end doesnt get too far behind
        # This way, when it encounters a big image it wont move on too far
        if i>thread_limit:
            threads[i-thread_limit].join()
            if (i-thread_limit)%200 == 0:
                print 'completed thread %s'%(threads[i-thread_limit].name)
        
    #make sure it does not execute any more code until all threads done
    for i in threads:
        i.join()
    
    startat = 0  
    t2 = time.clock()
    ttime = (t2-t1)/3600.
    print ('total time: '+str(ttime)+'hours')
    #this is to note the last running conditions:
    string = "This image was last run with the parameters mangam = %s, manbound=%s. \nIt took %s hours"%(mangam,manbound,ttime)
    record = open('%s%s%s_runinfo.txt'%(MBARS.PATH,runfile,MBARS.FNM),'w')
    for item in string:
        record.write(item)
    record.close()
    return
#setup all the parameters before running
for i in filenames:
    mangam, manbound = MBARS.FindIdealParams(i)
    
for i in filenames:
    if NOTHREADS:
        run(i,plot,startat)
    else:
        thread_run(i,plot,startat)




