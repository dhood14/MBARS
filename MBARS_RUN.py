import MBARS
import matplotlib.pyplot as plt
import scipy.misc as spm
import numpy as np
import time
#import cPickle as pickle
import threading
'''
set number of images, this is expecting a series of images with the same root name
and numerals at the end, i.e. "root0.PNG" and "root23.PNG"
'''
filenames = []

FRACS = [30,40,50,60,70]

#filename += 'ESP_011357_2285_RED300PX'


#filenameA = 'ESP_036437_2290_RED500PX'
#filenames  = [filenameB,filenameC,filenameD]

#GOlombek Comparison Images
#filenames += ['TRA_000828_2495_RED500PX']
#filenames +=['TRA_000828_2495_RED_16bit']

#filenames += ['PSP_001391_2465_RED500PX']
#filenames += ['PSP_001391_2465_RED16bit500PX']
filenames += ['PSP_001391_2465_RED16bit_1000PX']
filenames+=['TRA_000828_2495_RED16bit_500PX']

#McNaughton Comparison
#filenames+= ['PSP_002387_1985_RED16bit_500PX']
#filenames+=['PSP_002387_1985_RED16bit_Dtop_500PX']

#viking 1 lander setup:

#filenames += ['PSP_001521_2025_RED16bit100PNL44']
#filenames +=['PSP_001719_2025_RED16bit100PNL52_500PX']
#filenames +=['ESP_046170_2025_RED16bit100PNL52_500PX']



#viking 2 lander images
#filenames+=['PSP_001976_2280_RED16bit100PNL52_500PX']
#filenames+=['PSP_002055_2280_RED16bit100PNL57_500PX']
#filenames+=['PSP_001501_2280_RED16bit100PNL47']


#PSP_007718 subset images
#filenames = ['PSP_007718_2350_']
#filenames += ['PSP_007718_2350_RED16bit500PX']
#filenames += ['PSP_007718_2350_RED300px']

#JoesImages
#filenames += ['PSP_007693_2300_RED16bit500PX']

#Proposal Test Images
#filenames += ['PSP_001415_2470_RED500PX']
#filenames+= ['PSP_001415_2470_RED16bit500PX']
filenames+=['PSP_001415_2470_RED16bit_1000PX']
#filenames+=['PSP_001741_2395_RED16bit_500PX']
#filenames+=['PSP_001481_2410_RED16bit_500PX']
#filenames+=['PSP_001473_2480_RED16bit_500PX']
#filenames+=['PSP_001430_2470_RED16bit_500PX']
#filenames+=['PSP_001418_2495_RED16bit_500PX']
#filenames+=['PSP_001482_2490_RED16bit_500PX']
#filenames+=['PSP_001484_2455_RED16bit_500PX']
#filenames+=['PSP_001742_2370_RED16bit']


######SOME CONTROLS###################
#produce intermediate plots, False unless you are debugging something
plot = False

#for continuing broken runs, use Startat to specify which panel to begin on for the first run
#Keep in  mind that threaded runs do not complete the files in order, use with caution
startat = 0

#Process is largely processer-limited, so benefit to large number of threads is minimal
#setting no limit causes memory errors.
thread_limit = 30

def core(num,gam,plot,manbound,bound,odr_keycard):
    '''core function of the run file, does gamfun and boulder detect
    '''
    seg,good,runfile = MBARS.autobound(num,bound)
    #print 'step 1 done'
    if good:
        if any(seg.compressed()):
            bads = MBARS.boulderdetect_threadsafe(num,seg,runfile,odr_keycard)
            #MBARS.overlapcheck_threadsafe_DBSCAN(num,runfile, odr_keycard, overlap=.001)
            MBARS.overlapcheck_shadbased(num,runfile,odr_keycard)
    if num%200 == 0:
        print ('Done with image %s'%(num))
    return runfile

def thread_run(filename,plot,startat, frac):

    MBARS.FNM, MBARS.ID, MBARS.NOMAP,panels = MBARS.RunParams(filename)

    MBARS.PATH = 'D://MBARS//Images//%s//'%(MBARS.FNM)
    MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ, MBARS.ROTANG = MBARS.start()
    #mangam,manbound = MBARS.FindIdealParams(filename,True)
    #set the proportion of the shadow to use here
    bound = MBARS.getimagebound(panels,frac)
    mangam = 0
    manbound = 0
    t1 = time.process_time()
    
    threads = []
    krange = range(startat,panels)
    print ('%s images to run'%(panels))
    odr_keycard = threading.Lock()
    threads = [threading.Thread(target = core, args=(a,mangam,plot,manbound,bound,odr_keycard),name='%s'%(a)) for a in krange]
    count=0
    for i in range(len(threads)):
        runfile = threads[i].start()
        while threading.active_count() > thread_limit:
            #print( 'Waiting at %s on thread room'%(i))
            #print(threading.enumerate())
            time.sleep(5)
        # make sure the dragging end doesnt get too far behind
        # This way, when it encounters a big image it wont move on too far
        if i>thread_limit:
            threads[i-thread_limit].join()
            if (i-thread_limit)%200 == 0:
                print ('completed thread %s'%(threads[i-thread_limit].name))
        
    #make sure it does not execute any more code until all threads done
    for i in threads:
        i.join()
    
    t2 = time.process_time()
    ttime = (t2-t1)/3600.
    print ('total time: '+str(ttime)+'hours')
    #this is to note the last running conditions:
    string = "This image was last run with the parameters mangam = %s, manbound=%s. \nIt took %s hours"%(mangam,manbound,ttime)
    record = open('%s%s_runinfo.txt'%(MBARS.PATH,MBARS.FNM),'w')
    for item in string:
        record.write(item)
    record.close()
    return panels

#setup all the parameters before running
#for i in filenames:
    #mangam, manbound = MBARS.FindIdealParams(i)
    
for i in filenames:
    for j in FRACS:
        PNLS = thread_run(i,plot,startat,j)
        MBARS.OutToGIS('autobound//','autobound_'+str(j)+'//',PNLS)




