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
#filename = 'PSP_007718_2350_'
#filename = 'ESP_028612_1755_RED66_'
#filename = 'PSP_007718_2350_RED300px'
#filename = 'TRA_000828_2495_RED500PX'
filename = 'ESP_011357_2285_RED300PX'

MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(filename)
MBARS.FNM = filename
root, MBARS.ID, MBARS.NOMAP, num = MBARS.RunParams(filename)
MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ = MBARS.start()


#######################PICK THE ANALYSES##################
MakeCFAs = False
bigs = False
imageanalysis = True
#DOES NOT WORK
densmap = False


#controls for making CFAs
#determines diameter past which the program does not count boulders
maxd = 2.25

#doing this automatically now, will run on all unless told otherwise:
runfiles = []
allfiles = os.listdir(MBARS.PATH)
runfiles = [f for f in allfiles if 'gam' in f]
runfiles = [f+'//' for f in runfiles]

#force it into a limited list here:
runfiles = ['gam600_bound100//']


#controls for examining images & bigs
trgtfile = 'gam600_bound100//'
maxdiam = 2.25
#when running exmaine images, do you want to see images with no boulders?
showblanks = False



#where the actual work happens
if MakeCFAs:
    record = file('%s%s_record.csv'%(MBARS.PATH,MBARS.FNM),mode='wb')
    record.write('Filename ,Best Fit Rock Abundance,R^2,Upper limit RA, lower limt RA, maxd=%s\n'%(maxd))
    for i in range(len(runfiles)):
        plt.figure(i)
        fit_k,upfit_k,downfit_k, fit_r2 = MBARS.bulkCFA(runfiles[i],num+1,maxd,root)
        if fit_k == None:
            continue
        print'Best fit rock abundance for file %s is %s percent, up to %s, or down to %s with an R^2 of %s'%(runfiles[i],fit_k*100,upfit_k*100,downfit_k*100, fit_r2)
    #MBARS.plotCFArefs()
        plt.show()
        record.write('%s,%s,%s,%s,%s\n'%(runfiles[i],fit_k,fit_r2,upfit_k,downfit_k))
        MBARS.OutToGIS(runfiles[i],num)
    record.close()
    

if bigs:
    MBARS.FindExcluded(trgtfile,num,maxdiam)
    for i in range(num+1):
        bigs = MBARS.FindBigs(trgtfile,i, maxdiam)
        plt.show()
        
if imageanalysis:
    for i in range(num+1):
        print i
        MBARS.ExamineImage(trgtfile,i, showblanks)

if densmap:
    diam = 1.75
    drange = .25
    #binsize in meters
    binsize = 10.
    for i in range(num+1):
        #MBARS.DensMap(trgtfile,i)
        load = MBARS.getshads(trgtfile,i)

        points = []
        if not load:
            continue
        while True:
            try:
                dat = pickle.load(load)
            except(EOFError):
                break
            if dat.measured and dat.bouldwid_m < diam+drange and dat.bouldwid_m > diam-drange:
                points+=[dat.bouldcent]
        res = dat.resolution
        #hist, yedge, xedge = np.histogram2d(ys,xs,bins=20)
        
        #lets build these data into a smoothed image
        seg = np.load('%s%s%s%s_SEG.npy'%(MBARS.PATH, trgtfile, MBARS.FNM,i))
        #copy the mask and shape from seg then fill with zeros
        densmap = npma.copy(seg)
        densmap.fill(0)
        binsize_p = int(binsize/res)
        for j in points:
            y = int(j[0])
            x = int(j[1])
            for k in range(y-binsize_p,y+binsize_p):
                for l in range(x-binsize_p,x+binsize_p):
                    dist = np.sqrt((y-k)**2+(x-l)**2)
                    if dist<binsize_p:
                        try:
                            densmap[k][l]+=1
                        except(IndexError):
                            continue
            #densmap[y][x]+=1
            
        #Make sure that MBARS.NOMAP has been set to this image's parameters
        densmap_UR = npma.copy(densmap)
        #fillval = np.max(densmap_UR.flatten()*2)
        fillval = 0
        densmap_UR = densmap_UR.filled(fillval)
        if MBARS.NOMAP:
            rotang = -90-MBARS.SAZ            
        else:
            rotang = MBARS.SAZ-MBARS.NAZ
        densmap_UR = sktrans.rotate(densmap_UR, rotang, resize=False, preserve_range=True)
        o_image = imageio.imread(MBARS.PATH+root+str(i)+'.PNG')
        cropy = (len(densmap_UR)-len(o_image))/2
        cropx = (len(densmap_UR[0])-len(o_image[0]))/2
        densmap_UR = skutil.crop(densmap_UR,((cropy,cropy),(cropx,cropx)))
        imageio.imsave('%s%s%s%s_densmap.png'%(MBARS.PATH,trgtfile,MBARS.FNM,i),densmap_UR)
        MBARS.GISprep(trgtfile,i,'_densmap')


######Zone of dead code, only here until replcaement is confirmed to work
