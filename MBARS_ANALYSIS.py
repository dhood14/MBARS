import MBARS
import matplotlib.pyplot as plt
import scipy.misc as spm
import numpy as np
import time
import cPickle as pickle
import shutil
import matplotlib.patches as patches
import os
from numpy import ma as npma
import skimage.transform as sktrans
import skimage.util as skutil
import imageio
'''
Catch all code to look at results
'''




#filename = 'ESP_018352_1805_RED'
#filename = 'TRA_000828_2495_RED_300PX'

#filename = 'ESP_028612_1755_RED66_'


#filename = 'ESP_011357_2285_RED300PX'

#filename = 'ESP_036437_2290_RED500PX'

#Sholes Images
#filename = 'PSP_007718_2350_'
#filename = 'PSP_007718_2350_RED300px'
filename = 'PSP_007718_2350_RED16bit500px'
runfile='autobound//'
#runfile='gam600_manbound102//'

#Golombek Comparison Images
#filename = 'TRA_000828_2495_RED500PX'
#filename = 'TRA_000828_2495_RED_16bit'
#runfile = 'autobound//'
#runfile = 'gam600_manbound159//'
#filename = 'PSP_001391_2465_RED500PX'
#filename = 'PSP_001391_2465_RED16bit500PX'
#runfile = 'gam600_manbound135//'

#viking 1 lander images:
#filename = 'PSP_001521_2025_RED100PNL47_500PX'
#filename = 'PSP_001719_2025_RED100PNL52_500PX'
#filename = 'ESP_046170_2025_RED_100PNL52_500PX'

#Viking 2 lander:
#filename = 'PSP_001501_2280_RED100PNL47_500PX'
#filename = 'PSP_001976_2280_RED100PNL52_500PX'
#filename = 'PSP_002055_2280_RED100PNL57_500PX'

#Joes Images
#filename = 'PSP_007693_2300_RED500PX'
#runfile = 'gam600_manbound173//'

#Proposal Test Images
#filename = 'PSP_001415_2470_RED500PX'
#runfile = 'gam600_manbound85//'

#specify which running file to examine, include '//' on the end of the name

#runfile = 'gam600_manbound147//'

MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(filename)
MBARS.FNM = filename
root, MBARS.ID, MBARS.NOMAP, panels = MBARS.RunParams(filename)
MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ, MBARS.ROTANG = MBARS.start()


#######################PICK THE ANALYSES##################
ManualMerge = False
OutToGIS = True
MakeCFAs = False
bigs = False
imageanalysis = True

#Can manually reduce the panels to look at results subsets
#panels = 3000

#manual Merge Controls:
#uses trgetfile as the runfile
#A and B flags mark the boulders to merge
MMnum = 28
MMflags = [243,244]

#controls for making CFAs
#determines diameter past which the program does not count boulders
maxd = 10

#controls for examining images & bigs
#diameter past which boulders are considered "big" and plotted
maxdiam = 2
#when running exmaine images, do you want to see images with no boulders?
showblanks = False


#check for the proper path:
if not os.path.exists('%s%s'%(MBARS.PATH,runfile)):
    print 'runfile does not exist for this image, check runfile'
    exit()
#where the actual work happens
if ManualMerge:
    query = 'Did you mean to do a manual Merge?\n y/n?\n'
    answer = raw_input(query)
    if answer != 'y':
        print 'OK I wont do it'
    else:
        MBARS.ManualMerge(runfile,MMnum,MMflags)
if OutToGIS:
    MBARS.OutToGIS(runfile,panels)

if MakeCFAs:
    record = file('%s%s_record.csv'%(MBARS.PATH,MBARS.FNM),mode='wb')
    record.write('Filename ,Best Fit Rock Abundance,R^2,Upper limit RA, lower limt RA, maxd=%s\n'%(maxd))
    plt.figure(runfile)
    fit_k,upfit_k,downfit_k, fit_r2 = MBARS.bulkCFA(runfile,panels+1,maxd,2.25,root)

    print'Best fit rock abundance for file %s is %s percent, up to %s, or down to %s with an R^2 of %s'%(runfile,fit_k*100,upfit_k*100,downfit_k*100, fit_r2)
    #MBARS.plotCFArefs()
    plt.show()
    record.write('%s,%s,%s,%s,%s\n'%(runfile,fit_k,fit_r2,upfit_k,downfit_k))
        
    record.close()
    
if bigs:
    MBARS.FindExcluded(runfile,panels,maxdiam)
    for i in range(panels+1):
        bigs = MBARS.FindBigs(runfile,i, maxdiam)
        plt.show()
        
if imageanalysis:
    while True:
        query = "Which image do you want to analyze?\n"
        num = raw_input(query)
        try:
            MBARS.ExamineImage(runfile,num, showblanks)
        except:
            print "Failed to examine, image does not exist"
        query = 'Look at another? y/n\n'
        answer = raw_input(query)
        if answer != 'y':
            break


