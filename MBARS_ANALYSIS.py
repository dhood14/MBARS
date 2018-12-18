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




#root = 'ESP_018352_1805_RED'
#root = 'TRA_000828_2495_RED_300PX'
#root = 'PSP_007718_2350_'
#root = 'ESP_028612_1755_RED66_'
#root = 'PSP_007718_2350_RED300px'
root = 'TRA_000828_2495_RED500PX'

MBARS.PATH = 'C://Users//dhood7//Desktop//MBARS//Images//%s//'%(root)
MBARS.FNM = root

root, MBARS.ID, MBARS.NOMAP, num = MBARS.RunParams(root)
MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ = MBARS.start()

#runfile = 'gam60_bound10//'
#runfiles = ['gam60_bound10//','gam60_bound1//','gam70_bound10//']

MakeCFAs = False
bigs = False
imageanalysis = True
densmap = False


#controls for making CFAs
#determines diameter past which the program does not coutn boulders
maxd = 2.25
runfiles = []
#doing this automatically now, will run on all unless told otherwise:
allfiles = os.listdir(MBARS.PATH)
runfiles = [f for f in allfiles if 'gam' in f]
runfiles = [f+'//' for f in runfiles]
#force it into a limited list here:
runfiles = ['gam600_bound90//']

##MBARS.current()


#controls for examining images & bigs
trgtfile = 'gam600_bound90//'
maxdiam = 2.25
#when running exmaine images, do you want to see images with no boulders?
showblanks = False
#retrieve the number of files from runparams
#runparams = open(MBARS.PATH+'runparams.txt','rb')


##num =0
##files = os.listdir('%s%s'%(MBARS.PATH,runfiles[0]))
##files = [s for s in files if '_SEG' in s]
##files = [s.replace(root,'') for s in files]
##files = [s.replace('_SEG.npy','') for s in files]
##files = [int(x) for x in files]
##num = np.max(files)
print num

#while True:
#    os.path.isfile('%s//%s%s
##nums = [46,47,51,52,56]
##sizes = []
##centers = []
#this is the smoothing filter, its a radial smoother:
def smooth(array,i,j,r):
    #array is the array  you are smoothing, i,j are the 1st and 2nd coords, r is the radius
    pixels = 0.
    avg = 0.
    for k in range(-r,r+1):
        for p in range(-r,r+1):
            try:
                avg+=array[i+k][j+p]
                pixels +=1
            except IndexError:
                continue
    avg = avg/pixels
    return avg


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
##
##        plt.figure(1)
##        plt.imshow(densmap)
##        plt.figure(2)
##        plt.imshow(seg)
##        plt.figure(3)
##        plt.imshow(densmap_UR)
##        plt.figure(4)
##        plt.imshow(o_image)
##        plt.show()

        
        
##        if not load:
##            continue
##        patches = []
##        while True:
##            try:
##                dat = pickle.load(load)
##            except EOFError:
##                break
##            #dat.mirror()
##            #dat.shadowmeasure_m()
##            patches+=dat.patchplot()
##            #MBARS.plotborder(dat.mborder)
##            
##        fig,ax = plt.subplots(2,2,sharex = True, sharey = True)
##        #image = imageio.imread('%s%s%s.PNG'%(MBARS.PATH, MBARS.FNM, i))
##        #image = np.load('%s%s%s%s_rot_masked.npy'%(MBARS.PATH,runfile,MBARS.FNM,i))
##        ax[0][0].imshow(image, cmap='binary_r', interpolation='none')
##        ax[0][1].imshow(image,cmap='binary_r',interpolation='none')
##
##        #fig2 = plt.figure(2)
##        #ax2 = fig2.add_subplot(111)
##        #image2=imageio.imread('%s%s%s_SEG.PNG'%(MBARS.PATH, MBARS.FNM, i))
##        image2 = np.load('%s%s%s%s_SEG.npy'%(MBARS.PATH,runfile,MBARS.FNM,i))
##        ax[1][0].imshow(image2,interpolation='none')
##        ax[1][1].imshow(image2,interpolation='none')
##        for j in patches:
##            ax[0][1].add_patch(j)
##            #ax[1][1].add_patch(j)
##        plt.show()
##    num = i
##    load = MBARS.getshads(num)
##    if not load:
##        continue
##    centers = []
##    xs = []
##    ys = []
##    shadows = []
##    boulders = []
##    fig=plt.figure()
##    ax = fig.add_subplot(111)
##    while True:
##        try:
##            dat = pickle.load(load)
##        except EOFError:
##            #print 'NO FILE'
##            break
##        if dat.measured:
##            sizes += [[dat.bouldwid]]
##            centers+=[[dat.sunpoint]]
##            xs+=[dat.sunpoint[1]]
##            ys+=[dat.sunpoint[0]]
##            #dat.pointsplot()
##            #print dat.fitbeta[4]
##            shadows +=[patches.Ellipse([dat.fitbeta[2],dat.fitbeta[0]], dat.fitbeta[3]*2., dat.fitbeta[1]*2., np.degrees(np.pi-(dat.fitbeta[4]%(2*np.pi))), color = 'w',alpha=0.4)]
##            boulders += [patches.Circle([dat.bouldcent[1],dat.bouldcent[0]], (dat.bouldwid/2.), alpha=0.6)]
##    image = imageio.imread('%s%s%s.PNG'%(MBARS.PATH, MBARS.FNM, num))
##    #image = imageio.imread('%s%s%s_SEG.PNG'%(MBARS.PATH, MBARS.FNM, num))
##    #b = patches.PatchCollection(boulders, alpha=0.4)
##    #s = patches.PatchCollection(shadows, alpha=0.4)
##    print("total boulders=%s"%(len(sizes)))
##    plt.imshow(image, cmap='binary_r')
##    for j in boulders:
##        ax.add_patch(j)
##    ##    for j in shadows:
##    ##       ax.add_patch(j)
##    plt.show()
##    mi, lindat = MBARS.MoransI(i)
##    print mi.I
    
    
    


##    hist, yedge, xedge = np.histogram2d(ys,xs,bins=20)
##    #lets build these data into a smoothed image
##    seg = imageio.imread('%s%s%s_SEG.PNG'%(MBARS.PATH, root, num))
##    #fix the xedge and yedge beginnings and ends
##    yedge[0] = 0
##    xedge[0] = 0
##    yedge[-1] = len(seg)
##    xedge[-1] = len(seg[0])
##    densmap = np.zeros_like(seg)
##    xpos = 0
##    ypos = 0
##    for i in range(len(densmap)):
##        if i >= yedge[ypos]:
##            ypos+=1
##        for j in range(len(densmap[0])):
##            if j>= xedge[xpos]:
##                xpos+=1
##            densmap[i,j] = hist[ypos-1,xpos-1]
##        xpos=0
    #print hist
    #print densmap
    
    #smooth the density map

##    smdensmap = np.copy(densmap)
##    for i in range(len(densmap)):
##        for j in range(len(densmap[0])):
##            smdensmap[i][j] = smooth(densmap,i,j,20)
            
##    #plt.plot(xs,ys)
##    plt.figure(1)
##    plt.imshow(hist)
##    plt.figure(2)
##    plt.imshow(densmap)
##    plt.figure(3)
##    plt.imshow(seg)
##    #plt.figure(4)
##    #plt.imshow(smdensmap)
##    plt.show()
##    #return(xedge, yedge)
##    spm.imsave('%s%s%s_DENS.PNG'%(MBARS.PATH, root, num), densmap)
###make a size-frequency dist
##seg = np.load('%s%s%s_SEG.npy'%(MBARS.PATH, root, 46))
##conv = .59 #expressed in m/pixel
##area = 5*(len(seg)*len(seg[0])*(conv**2)) #calculating the area anlyzed (very ad hoc)
##corrsizes = []
##for i in sizes:
##    corrsizes+=[conv*i]
##sizehist, sizeedges = np.histogram(corrsizes, bins=20)
##bincents = []
##fiveper = []
##fortyper = []
##for i in range(len(sizeedges)):
##    if i != 0:
##        avg = (sizeedges[i]+sizeedges[i-1])/2.
##        bincents +=[avg]
##        fiveper+=[.328*10**(-2.8108*avg)]
##        fortyper+=[.892*10**(-1.3196*avg)]
##sizehist = sizehist/area
##np.savetxt('%s%s_SFDist_x.csv'%(MBARS.PATH,root), bincents, delimiter=",")
##np.savetxt('%s%s_SFDist_y.csv'%(MBARS.PATH,root), sizehist, delimiter=',')
###lets make some model curves
##
##
##plt.plot(bincents, sizehist, 'ko')
###plt.plot(bincents, fiveper , 'b-')
###plt.plot(bincents, fortyper,'g-')
##plt.yscale('log')
##plt.xscale('log')
##plt.show()


######Zone of dead code, only here until replcaement is confirmed to work
