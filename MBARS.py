import sys
#maytbe can take this out?
sys.path.append('C:\\Python27\\Lib\\site-packages')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.misc as spm
from scipy import odr
from scipy.optimize import curve_fit
import time
#use Cpickle if available
#import cPickle as pickle
import pickle
import os
import matplotlib.patches as patches

import numpy.ma as npma
import skimage.transform as sktrans
from math import *
import skimage.feature as skfeat
import skimage.segmentation as skseg
import skimage.morphology as skmorph
import skimage.filters.rank as skrank
import skimage.restoration as skrestore
import skimage.util as skutil
import sklearn.cluster as skcluster
from scipy.ndimage import filters
import scipy.stats as sps
import scipy.signal as spsig
import imageio
import threading

try:
    raw_input = input
except(NameError):
    pass

#This is the MBARS library, it contains all the functions needed to run MBARS

#Global Variables, adjust as needed:
#REFPATH is where important reference files are stored, the key one is
# the HiRISE info file (RDRCUMINDEX.TAB) needs to be in the REFPATH folder
REFPATH = 'C://Users//Don_Hood//Documents//MBARS//RefData//'
#BASEPATH is where MBARS will look for the provided filename
BASEPATH = 'C://Users//Don_Hood//Documents//MBARS//Images//'
#The PATH variable is modified to be the BASEPATH + the filename, where to look
#for the images for a given MBARS run.
PATH = None
FNM = None

#enter product ID
ID = None
NOMAP = None

WIDFACT = 2.
LENFACT = 2.
#assigned in start as i, s, and r
INANGLE = None
NAZ = None
SAZ = None
SUNANGLE = None
RESOLUTION = None
ROTANG = None
#MAX EXPECTED BOULDER SIZE, expressed in pixels
#max diameter (MD) and max height (MH), done after measurements
MD = 30
MH = 30
#maximum shadow area, taken before measrements, based soleley on the shadow area
MA = 3000
#minimum accepted shadow size expressed in pixels
minA = 4
#minimum distance between maxima in the  watershed splitting
mindist = 3


'''this is called at the end to initialize the program'''
def start():
    i,s,r,n, saz,rotang = getangles(ID)
    current()
    return i,s,r,n,saz, rotang
    

def autobound(num,bound):
    ''' An automatic boundary-finder for HiRISE images, relies on input statistics
'''
   
    runfile = 'autobound//'
    if not os.path.exists('%s%s'%(PATH,runfile)):
        try:
            os.makedirs('%s%s'%(PATH,runfile))
        except(WindowsError):
            #this is in case two threads try and make something at the same time
            #is it dirty? yes, does it work? also yes.
            pass
    try:
       image = imageio.imread('%s%s%s.PNG'%(PATH,FNM,num))
    except(ValueError, SyntaxError):
        return None, False, runfile

    if not np.any(image.flatten()):
        return None, False, runfile
    image = sktrans.rotate(image,ROTANG, resize=True, preserve_range=True)
    image = npma.masked_equal(image, 0)
    

    ''' rotation seems to cause some stray data to appear at the edge, this is often
        categorized as shadows because it is very dark but not zero, this code will
        essentially erode the edges a bit, shifting the mask up, down, left and right
        and taking the logical OR of the two masks, masking things that are adjacent to
        the mask in any direction
        '''
    shift = np.roll(image.mask,1,1)
    image.mask = np.logical_or(image.mask, shift)
    shift = np.roll(image.mask,-1,1)
    image.mask = np.logical_or(image.mask, shift)
    shift = np.roll(image.mask,1,0)
    image.mask = np.logical_or(image.mask, shift) 
    shift = np.roll(image.mask,-1,0)
    image.mask = np.logical_or(image.mask, shift)
    
    imageseg = npma.copy(image)
    imageseg = imageseg.astype(float)
    imageseg = imageseg.filled(-1)

    imageseg[imageseg>bound+1] = bound+1
    imageseg = npma.masked_equal(imageseg, -1)    
    imageseg = imageseg.astype(int)
    imageseg.fill_value = 0
    
    imageseg.dump("%s%s%s%s_SEG.npy"%(PATH,runfile,FNM,num))
    
    #guard against images with no shadows
    if np.min(imageseg)>= bound:
        good = False
    else:
        good=True
    
    return(imageseg, good, runfile)

def getimagebound(panels,prop):
    '''TO retrieve the overall image stats and calculate the absolute shadow boundary

    assumes a 2-pixel wide (4 total pixels) shadow to the minimum shadow size'''

    bins = np.linspace(0,1023,1024)
    bins = bins.astype(int)
    hist = None
    cum_hist = None
    #panels = 400
    for i in range(0,panels):
        #print i
        im = imageio.imread('%s%s%s.PNG'%(PATH,FNM,i))
        n,bins = np.histogram(im,bins)
        #cum_n,bins,c=plt.hist(im.flatten(),bins,cumulative=True)
        n[0] = 0
        if i==0:
            hist = n
        else:
            hist+=n

    #we have the histogram for the whole image now.

    mode = np.argmax(hist)

    #make and normalize the cumulative histogram
    cum_hist = np.cumsum(hist)

    ncum_hist = cum_hist/float(max(cum_hist))
    #this is now a map to the actual image distribution
    runs = 100
    #this is important, what boundary should be chosen? I am going with the
    #average of the 100th percentile
    stats = []
    for i in range(runs):
        img = ImageMaker(ncum_hist,bins)
        #plt.imshow(img)
        #plt.show()
        #.77 for the lorentzian taken from
        #Kirk et al 2008, 10.1029/2007JE003000
        img_con = convolve_lorentzPSF(img,mode,.77)

        #shadow size comes in HERE, must be same as in Image Maker function
        shad_con = img_con[25:35,25:35]

        stat = np.percentile(shad_con,prop)
        stats+=[stat]

    bound = np.average(stats)
    print ("Selected boundary at %s"%(bound))
    return bound


def ImageMaker(mapping_hist,mapping_bins,dimx=50,dimy=50):
    ''' returns a value from the HiRISE image population, meant to replicate what I see in MBARS'''
       
    img = np.random.rand(dimy,dimx)
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(mapping_hist)):
                if mapping_hist[k]>=img[i][j]:
                    img[i][j] = float(mapping_bins[k])
    #shadow size comes in here, must be same as above
    img[25:35,25:35] = 1
    return img
    
def convolve_lorentzPSF(image,avg,gam=.77):
    '''convolve an array with a lorentzian HiRISE PSF'''
    kern = lorentz_kern(31,gam)
    #print avg
    newimage=spsig.convolve2d(image,kern,mode='same',boundary='fill',fillvalue=avg)
    return newimage
    
def lorentz(x,xo,gam):
    #this is a lorentzian, a more accurate PSF for HiRISE
    l = (1/(np.pi*gam))*((gam**2)/(((x-xo)**2)+gam**2))
    return l

def lorentz_kern(dim=21,gam=.77):
    #method to make a square lorentzian kernal for deconvolution
    #VERY IMPORTANT, MUST BE ODD numbered
    if dim%2==0:
        print("Must be odd, adding 1 to dim")
        dim+=1
    kern = np.empty((dim,dim))
    mid = dim/2
    for i in range(len(kern)):
        for j in range(len(kern[0])):
            y= float(i-mid)
            x = float(j-mid)
            r = np.sqrt(y**2+x**2)
            kern[i][j] = lorentz(r,0,gam)
    total = sum(kern.flatten())
    kern = kern/total
    #plt.imshow(kern)
    #plt.show()
    return(kern)

#########This is the measuring side of the code#####################################

def boulderdetect_threadsafe(num,image,runfile,odr_keycard):
    #flag must be dtype long, otherwise it will wrap at high numbers and reset the flag to 1
    #longs do not exist in python 3, leaving some compatibility in
    try:
        flag = long(1)
    #NameError should only trip when Python 3 is running, in which case all ints are longs in effect
    except(NameError):
        flag = 1
    coords = [0,0]
    save = open("%s%s%s%s_shadows.shad"%(PATH,runfile, FNM,num), 'wb')
    im_area = len(image)*(len(image[0]))
    #flagged image has the boulders marked
    #print 'Running Watershed %s\n'%(num)
    fimage = watershedmethod(image)
    #have to explicitly pass the mask on from the input image
    fimage = npma.masked_array(fimage)
    fimage.dump('%s%s%s%s_flagged.npy'%(PATH, runfile,FNM,num))
    fimage.mask = image.mask
    fmax = np.max(fimage)
    
    #clear the seg image to save memory
    image = None
    shade=None
    
    for i in range(fmax+1):
        #find all the parts of the image with a shared flag ID, these are theoretically part of a single shadow area
        pixels = np.argwhere(fimage == i)
        #must be converted to list for the neighbor-checking function to work
        pixels = pixels.tolist()
        
        #A shdaow image will only be made if the shadow is of an appropriate size, not too big or too small
        if len(pixels)<MA and len(pixels)>minA:
            #Create a shadow object initialized on the list of pixels
            shade = shadow(i, pixels, im_area)
            
            #broken into 3 steps to narrow the thread-unsafe part into one function
            shade.run_prep()
            with odr_keycard:
                shade.run_fit()
            shade.run_post()
            
            pickle.dump(shade,save)
    save.close()
    
    return
            
def watershedmethod(image):
    #this is the new way of finding the shadows in an image
    #first find the "plateau" value, we will need this for masking
    plat = sps.mode(image.compressed())
    #fill the image with a known value in order to preserve the mask
    temp = image.filled(np.max(image)+1)
    #invert the image so the shadows are peaks not lows
    temp = temp*(-1)+np.max(image)+1
    #find the peaks in the image, return the points as a nx2 array
    #min_distance is a super important argument,changes the minimum distance allowed betwen maxima
    #added absolute threshold so that the background doesnt get selected...
    #Need to fix this whole function for the future warnings I keep getting:
    
    #To avoid this warning, please do not use the indices argument. Please see peak_local_max documentation for more details.
    #  points = skfeat.peak_local_max(temp,min_distance=mindist,threshold_abs = 2, indices=True)
    #C:\Users\Don_Hood\Anaconda3\envs\MBARS\lib\site-packages\skimage\morphology\_deprecated.py:5: skimage_deprecation: Function ``watershed`` is deprecated and will be removed in version 0.19. Use ``skimage.segmentation.watershed`` instead.
    #  def watershed(image, markers=None, connectivity=1, offset=None, mask=None,
    

    #dropped the "indices" argument as requiested by the warning, indices is now always "true" so the argument is not needed
    points = skfeat.peak_local_max(temp,min_distance=mindist,threshold_abs = 2)

    #put in a guard against images with no shadows where the entire image is black
    # and images where there are no minima (likely nothing that isnt masked)
    threshold = len(image.compressed())/2
    if len(points)>threshold or len(points)==0:
        return np.ones_like(image)

    #prepare to convert the points matrix to an image-like array
    #this could perhaps be done with DBSCAN
    #some 0-length arrays are making it to DBSCAN, not sure why...
    
    cores,labels = skcluster.dbscan(points,eps=2,min_samples=2)
    
    view = np.zeros_like(image)
    flag = 2
    excl = []
    for i in range(len(points)):
        if labels[i] == -1:
            view[points[i][0]][points[i][1]] = flag
            flag+=1
        elif labels[i] not in excl:
            view[points[i][0]][points[i][1]] = flag
            excl+=[labels[i]]
            flag+=1

    #this will make the mask on view mask out the originally masked points (outside data)
    # as well as the plateau pixels
    temp = npma.masked_equal(image, plat[0][0])
    view.mask = np.logical_or(temp.mask, view.mask)
        
    '''details on the arguments being handed to watershed:
        in general ,the algorithm doesnt like masked arrays, hence everything is filled
        the masks are added back in the makeshadows code above
        the mask at the ennd prevents it from trying to segment all the no-data areas
        however it takes the mask in the opposite sense of the masked arrays
        in ndarrays, the mask is True where you want it masked, watershed wants 0
    '''
    #changed the skmorph.watershed to skseg.watershed    
    #API suggests none of the arguments have changed, so this should be an easy swap    
    boulds = skseg.watershed(image.filled(np.max(image)+1), view.filled(0),mask=~view.mask)

    return boulds

#Not curently used, consdier removing###
def overlapcheck_threadsafe_DBSCAN(num,runfile,odr_keycard,overlap=.1):
    '''
    Code to get rid of double-counts in returned boulders, inputs:
    num - num of target image
    runfile - current runfile for shad file finding purposes
    overlap - allowed overlap between boulders, expressed in fraction of boulder area
    '''
    shadow_file = getshads(runfile, num,mode='rb')
    if shadow_file == None:
        return
    parameters = []
    while True:
        try:
            data = pickle.load(shadow_file)
        except(EOFError):
            break
        #pull in the flag, the y coordiante, the x coordinate, and the width
        parameters+=[[data.flag,data.bouldcent[0],data.bouldcent[1],data.bouldwid, data.fitgood]]
    #get rid of boudlers with unreliable measurements
    parameters = [a for a in parameters if a[4]]
    if len(parameters) == 0:
        shadow_file.close()
        return

    #compute modified distance matrix
    #make meshgrids of every number in the calculation
    #includes some precomputing to minimize the memory footprint
    ya,yb = np.meshgrid([a[1] for a in parameters],[a[1] for a in parameters])
    ydist = (ya-yb)**2
    ya = None
    yb = None
    xa,xb = np.meshgrid([a[2] for a in parameters],[a[2] for a in parameters])
    xdist = (xa-xb)**2
    xa = None
    xb = None
    tdist = np.sqrt(ydist+xdist)
    ydist = None
    xdist = None
    da,db = np.meshgrid([a[3] for a in parameters],[a[3] for a in parameters])
    dsum = da+db
    da = None
    db = None
    #calculate the distance ratio
    #D = euclidean distance between points/sum of radii, this way boulders that touch have a D of 1 or less
    D = 2*tdist/(dsum)
    dsum = None
    tdist = None
    #this is now the Distance matrix to pass to DBSCAN
    #arguments that go to DBSCAN: D, distance matrix, 1=distance to be considered neighbors, 2 min number to form group, metric: Since we are passing a distance matrix, precomputed
    #DBSCAN is a little generous to help catch more things
    cores,labels = skcluster.dbscan(D,1.2,2,metric='precomputed')
    #cores is a list of 'core' points, less important than labels, which groups the points into unified labels, -1 means they arent in any clusters
    #add these to the parameters list
    for i in range(len(parameters)):
        parameters[i]+=[labels[i]]
    #make our 'web' of flags, a list where every sublist is flags that are interconnected
    webs = []
    for i in range(max(labels)+1):
        subweb=[]
        for j in parameters:
            if j[5] == i:
                subweb+=[j[0]]
        webs+=[subweb]
    
    #now lets re-read in the shadow data
    shadow_file.seek(0)
    og_data = []
    while True:
        try:
            og_data+=[pickle.load(shadow_file)]
        except(EOFError):
            break
    #close and re-open for re-writing
    shadow_file.close()
    shadow_file = getshads(runfile,num,mode='wb')

    
    problem_flags = [a[0] for a in parameters if a[5]!=-1]
    problem_boulders = []
    for i in og_data:
        if i.flag in problem_flags:
            problem_boulders+=[i]
        else:
            pickle.dump(i,shadow_file)
    og_data = None
    #print(webs)
    #OK with just the problem_boulders list, we can do the work
    for subweb in webs:
        #print subweb
        current_web = []
        #take the web list (which is just flags) and make a list of shadow objects
        for bould in problem_boulders:
            if bould.flag in subweb:
                current_web+=[bould]
        
        
        #merge = True
        while True:
            #print 'Current_web length: %s'%(len(current_web))
            #print(subweb)
            #reset to no merge
            merge = False
            #current_web = newboulds
            newboulds = []
            for a in current_web:
                #print'looking for merges'
                if merge:
                    continue
                #A_flag = j.flag
                for b in current_web:
                    if merge:
                        continue
                    #B_flag = k.flag
                    if a.flag != b.flag:
                        #look for overlapping pairs in the web
                        
                        area = checkpos(a,b)
                        #print(area)
                        
                        if area > overlap:
                            #print 'Merging Boulders'
                            #if they overlap too much, make a new shadow that combines their pixels and re-run
                            newbould=shadow(a.flag, a.pixels+b.pixels, a.im_area)
                            newbould.run_prep()
                            with odr_keycard:
                                newbould.run_fit()
                            newbould.run_post()

                            #compare the fiterr
                            #the ones too small to fir are assigned stupid high fiterr values
                            if a.fiterr:
                                fita = a.fiterr
                            else:
                                fita = 100
                            if b.fiterr:
                                fitb = b.fiterr
                            else:
                                fitb=100
                            fitavg = (fita+fitb)/2.
                            #trying to not filter until the end to avoid killing intermediate boulders
                            #boulders that are smaller than the min and are improvements on the originals are allowed to pass
                            if newbould.bouldwid<MD and newbould.fiterr<fitavg:
                            #if True
                                #you found two that needed to merge and it made a decent boulder
                                A_flag = a.flag
                                B_flag = b.flag 
                                #print 'Merged %s and %s'%(A_flag,B_flag)
                                merge = True
                                #if the new boulder does not look good, Merge is not set to True and the algorithm never merges them
                            
                                
            if merge:
                #since you found two that merged, dont add them to the next web and put the new one in
                #print 'I found a merge!'
                for bould in current_web:
                    if bould.flag != A_flag and bould.flag != B_flag:
                        newboulds+=[bould]
                        #print bould.flag
                newboulds+=[newbould]
                current_web = newboulds
            else:
                break
                
                
            #if no merges are found in the inner loops, the code will continue to the next web
        #now save all these new ones you just made
        for boulder in current_web:
            #print'Dumping shadows'
            #print 'Saving Boulders'
            #print(boulder.flag)
            #if boulder.fitgood:
            pickle.dump(boulder,shadow_file)
    shadow_file.close()
    return

##This one is actually Used##
def overlapcheck_shadbased(num,runfile,odr_keycard):
    '''
    Another take on the overlap merging function. This one is NOT based on the interpreted boulders, rather the adjacency of shadows
    if shadows touch each other, the function considers whether combining it with other shadows makes it a "better" shadow
    '''
    shadow_file = getshads(runfile, num,mode='rb')
    #print(shadow_file)
    if shadow_file == None:
        return
    parameters = []
    while True:
        try:
            data = pickle.load(shadow_file)
        except(EOFError):
            break
        #pull in the flag, y center of the shadow, the x center of the shadow, the border
        parameters+=[[data.flag,data.center[0],data.center[1],data.border,data.pixels]]
    
    if len(parameters) == 0:
        shadow_file.close()
        return
    
    #each element in parameter is a boulder, first things first is to get the "adjacency" list
    #each item in "adjacency" has three items, flag a, flag b, adjacency score (number of adjacent pixels in borders)
    adjacency = []
    for i in range(len(parameters)):
        aflag = parameters[i][0]
        ay = parameters[i][1]
        ax = parameters[i][2]
        abord = parameters[i][3]
        apix = parameters[i][4]
        if len(abord)==0:
            abord = apix
        #dont double-count pairs, this should speed things up
        for j in range(i+1,len(parameters)):
            #quicker check if they are super far apart
            bflag = parameters[j][0]
            by = parameters[j][1]
            bx = parameters[j][2]
            bbord = parameters[j][3]
            bpix = parameters[j][4]
            if len(bbord) ==0:
                bbord=bpix

            dist = np.sqrt((ay-by)**2+(ax-bx)**2)
            if dist > sum([len(apix),len(bpix)]):
                adjacency+=[[aflag,bflag,0]]
                continue
            #they are not super far away, so we can check adjacency
            D = 0
            for k in abord:
                for l in bbord:
                    dist = np.sqrt((k[0]-l[0])**2+(k[1]-l[1])**2)
                    #print(k,l,dist)
                    if dist<=1:
                        D+=1
            adjacency+=[[aflag,bflag,D]]
    #print [a for a in adjacency if a[2]>0]
    #OK, we have the adjacency, now we need to determine the clusters.
    #these are organized by the A flag, so lets make some webs
    clusters = []
    for adj in adjacency:
        #the two flagged shadows dont touch
        if adj[2] == 0:
            continue
        #they do touch
        else:
            #see if webs exist that contain these flags
            if len(clusters) == 0:
                clusters+=[[adj[0],adj[1]]]
                continue
            #hits will mark the clusters in which the flags are found
            hits = []
            for i in range(len(clusters)):
                if adj[0] in clusters[i] or adj[1] in clusters[i]:
                    hits+=[i]
            if len(hits) == 0:
                clusters+=[[adj[0],adj[1]]]
            else:
                newclust = []
                hits.sort(reverse=True)
                for i in hits:
                    newclust+= clusters.pop(i)
                #need to amke sure new flags get added, repeats will be there, but it doesnt actually matter
                newclust+=[adj[0],adj[1]]
                clusters+=[newclust]
                
    #clusters is now a list of clusters, adjacency is still available to reference the adjacency value for pairs
    #return to the start of the shadows and start the merging process
    #now lets re-read in the shadow data
    shadow_file.seek(0)
    og_data = []
    while True:
        try:
            og_data+=[pickle.load(shadow_file)]
        except(EOFError):
            break
    #close and re-open for re-writing
    shadow_file.close()
    shadow_file = getshads(runfile,num,mode='wb')

    prob_flags = [a for clust in clusters for a in clust]
    problem_boulders = []
    for i in og_data:
        if i.flag in prob_flags:
            problem_boulders+=[i]
        else:
            #pass
            pickle.dump(i,shadow_file)
    og_data = None

    #for testing purposes
    report = []
    for clust in clusters:
        #lets collect the shadow objects for the relevant clusters
        
        boulds = []
        avg_fiterr = 0.
        all_pixels = []
        for rock in problem_boulders:
            if rock.flag in clust:
                boulds+=[rock]
                #if the rock is one of the big ones, anything is an improvement
                if rock.bouldwid > MD:
                    avg_fiterr+=1000
                    
                elif rock.fiterr:
                    avg_fiterr+=rock.fiterr
                else:
                    avg_fiterr+=1000
                all_pixels+=rock.pixels
        base_flag = boulds[0].flag
        #identify areas that are way too big, likely shadow-casting topography
        if len(all_pixels) > 5000:
            #print base_flag
            #print (len(all_pixels))
            continue
            
        avg_fiterr = avg_fiterr
        #now, merge the pixels and see if it is better.
        newbould=shadow(base_flag, all_pixels, boulds[0].im_area)
        newbould.run_prep()
        with odr_keycard:
            newbould.run_fit()
        newbould.run_post()
        #consider which is better?
        #new one exists, is better, and is smaller than max
        #print 'considering cluster %s'%(clust)
        #print 'New fiterr = %s, avg_fiterr = %s'%(newbould.fiterr,avg_fiterr)
        if newbould.fiterr and newbould.fiterr < avg_fiterr and newbould.bouldwid < MD:
            pickle.dump(newbould,shadow_file)
            #report+=['New boulder was better']
     
        #this is the method where we filter based on connectivity scores, 
##        else:
##            #report+=['old boulders were better']
##            #OK, lets attempt to calculate "connectivity", the sum of adjacency scores:
##            connectivity = []
##            for rock in boulds:
##                connect = sum([a[2] for a in adjacency if a[0] == rock.flag or a[1] == rock.flag])
##                connectivity+=[connect]
##            #zip connectivity into the boulds list
##            boulds = [list(a) for a in zip(boulds,connectivity)]
##            #print'Entering Recursive Portion'
##            #print (zip([a[0].flag for a in boulds],connectivity))
##            exclusive_shadowmerge(boulds,2,shadow_file,odr_keycard)
        #Lets try a k-means based method
        else:
            #print newbould.bouldwid
            kmeans_shadowmerge(boulds,shadow_file,odr_keycard,avg_fiterr)
            
            
    return
    #return clusters,adjacency,report
                
                
def exclusive_shadowmerge(boulds,mincon,shadowfile,odr_keycard):
    ''' Calls the shadowmerge method, recursively tries to make new boulders from adjacent shadows
    '''
    all_pixels = []
    avg_fiterr = 0
    inboulds = [rock for rock in boulds if rock[1]>=mincon]
    outboulds = [rock for rock in boulds if rock[1]<mincon]
    if len(inboulds)>0:
        for rock in inboulds:
            if rock[0].bouldwid > MD:
                avg_fiterr+=100
            elif rock[0].fiterr:
                avg_fiterr+=rock[0].fiterr
            else:
                avg_fiterr+=100
            all_pixels+=rock[0].pixels
        base_flag = inboulds[0][0].flag
        avg_fiterr = avg_fiterr/(float(len(inboulds)))
        #now, merge the pixels and see if it is better.
        newbould=shadow(base_flag, all_pixels, inboulds[0][0].im_area)
        newbould.run_prep()
        with odr_keycard:
            newbould.run_fit()
        newbould.run_post()
        if newbould.fiterr and newbould.fiterr < avg_fiterr and newbould.bouldwid < MD:
            pickle.dump(newbould,shadowfile)
            if len(outboulds)>0:
                exclusive_shadowmerge(outboulds,1,shadowfile,odr_keycard)
        else:
            exclusive_shadowmerge(boulds,mincon+1,shadowfile,odr_keycard)
    else:
        for rock in boulds:
            pickle.dump(rock[0],shadowfile)

#Used in the overlap check_Shadbased
def kmeans_shadowmerge(boulds,shadowfile,odr_keycard,avg_fiterr):
    #boudlds is a list of shadow objects, shadowfile is the targeted shadow file, should be in "write" mode
    #odr-keycard is the thread lock object to prevent multiple access to ODR, maxboulds is the highest k-means will go
    #avg_fiterr is the average fit error on the original boulders, we have to be better
    all_pixels = [b for a in boulds for b in a.pixels]
    #this is a list of all flags, maxbolds is limited by the length of this list
    all_flags = [a.flag for a in boulds]
    maxboulds = len(all_flags)
    newerrs_list = []
    newboulds_list = []
    #print all_flags
    for i in range(2,maxboulds+1):
        #lets try manually seeding to limit splits along the sun-line
        #by seeding the k-means as an equal x-spread, this should strongly favor lateral boulders rather than vertical
        minx = min([a[1] for a in all_pixels])
        maxx = max([a[1] for a in all_pixels])
        avgy = np.average([a[0] for a in all_pixels])
        #make a linspace that goes from min to max, then remove the ends.
        seeds  = list(np.linspace(minx,maxx,i+2))
        seeds.pop(0)
        seeds.pop(-1)
        seeds = map(lambda x: (x,avgy),seeds)
        seeds = [list(a) for a in seeds]
        seeds = np.array(seeds)
        #kmeans = skcluster.KMeans(n_clusters=i,init = seeds).fit(all_pixels)
        kmeans = skcluster.KMeans(n_clusters=i).fit(all_pixels)
        labelpix = [list(a) for a in zip(kmeans.labels_,all_pixels)]
        #cents = kmeans.cluster_centers_
        #print(kmeans.labels_)
        #for display purposes
        #print(kmeans.cluster_centers_)
##        cols = [a[0] for a in labelpix]
##        ys = [a[1][0] for a in labelpix]
##        xs = [a[1][1] for a in labelpix]
##        plt.scatter(xs,ys,c=cols,cmap='rainbow')
##        plt.show()

        #check if this is a good fit!
        newboulds = []
        new_fiterr = 0
        for j in range(i):
            pix = []
            for a in labelpix:
                if a[0] == j:
                    pix+=[a[1]]
            newbould = shadow(all_flags[j],pix,boulds[0].im_area)
            newbould.run_prep()
            with odr_keycard:
                newbould.run_fit()
            newbould.run_post()
            newboulds+=[newbould]
            if newbould.bouldwid > MD:
                new_fiterr+=1000
            elif newbould.fiterr:
                new_fiterr+=newbould.fiterr
            else:
                new_fiterr+=1000
        new_fiterr = new_fiterr
        #add the results to these list
        newerrs_list+=[new_fiterr]
        newboulds_list+=[newboulds]
    #pick the best solution:
    #print avg_fiterr
    #print min(newerrs_list)
    if min(newerrs_list)<= avg_fiterr:
        #we found a better solution
        #print "%s is best fit"%(np.argmin(newerrs_list)+2)
        final_boulders = newboulds_list[np.argmin(newerrs_list)]
        for i in final_boulders:
            pickle.dump(i,shadowfile)
    else:
        #no better solution was found
        #print "no better fit was found"
        for i in boulds:
            pickle.dump(i,shadowfile)
    return()
            
#called in an unused dunction, consider removing
def checkpos(shadow1,shadow2):
    '''expects two shadow objects , checks how much they overlap
    returns a fraction of the area of the smaller boulder overlapped by the larger
    '''
    bouldwid1 = shadow1.bouldwid
    bouldwid2 = shadow2.bouldwid

    pos1 = shadow1.bouldcent
    pos2 = shadow2.bouldcent
    #if one is unmeasured or has no dimensions, forget it
    if pos1[0] == None or pos2[0] == None:
        return 0.
    #distance between the two circles
    d = np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
    r1 = bouldwid1/2.
    r2 = bouldwid2/2.
    if r1>=r2:
        ra = r1
        rb = r2
    else:
        ra = r2
        rb = r1
    #lets throw out some simple scenarios
    #the circles exactly reach or do not overlap
    if d>= ra+rb:
        return 0.
    #if the smaller is fully in the larger:
    if d+rb <=ra:
        return 1.
    #if the center of one is within the other, but not entirely
    if ra>d and d+rb>ra:
        #returning 1 beccause I always want these to merge
        return 1.
    else:
        #note: all this work is done in pixels not meters
        x = (ra**2 - rb**2 + d**2)/(2*d)
        y = np.sqrt(ra**2-x**2)
        #note: this returns in radians
        thetaA = abs(np.arccos(x/ra))
        thetaB = abs(np.arccos((d-x)/rb))
        #relevant EQ, area of circle slice: piR^2*(2theta/2pi), triangle .5*b*h
        area = (ra**2)*thetaA - x*y + (rb**2)*thetaB - (d-x)*y
        #print('ra = %s, rb = %s, d = %s'%(ra,rb,d))
        area = area/(np.pi*rb**2)
        return area
    
######Shadow Object Definition###########
class shadow(object):

    def __init__(self, flag, pixels, im_area):
        #marker for this boulder, unique
        self.flag = flag
        #immediate pixels in this shadow
        self.pixels = pixels
        #pixels in the mirrored shadow
        self.mpixels = None
        #area of the shadow, based on how many pixels are in it
        self.area = len(pixels)
        #area for the mirrored shadow, simply 2*area for now, may change
        self.marea = 2*self.area
        #incedence angle of the sun, fixed by target image
        self.inangle = INANGLE
        #angle of the sun, fixed as coming from top (-y), this is holdover till
        #code sentive to angle is entirely gone
        self.beta = np.radians(270)
        #image resolution, tied to each shadow image for safekeeping
        self.resolution = RESOLUTION
        #image area (pixels^2) tied to each shadow for safekeeping
        self.im_area = im_area
        #find the center of the shadow
        tempx = 0.
        tempy = 0.
        for i in range(len(pixels)):
            tempy +=pixels[i][0]
            tempx +=pixels[i][1]
        self.center = [tempy/float(len(pixels)),tempx/float(len(pixels))]
        #border of the shadow
        self.border = []
        #initially, the border of the shadow without the -y(upper) side,
        #finally the rim of the mirrored shadow
        self.mborder = []
        #the flipping axis of the shadow
        self.flipaxis = []
        #initial conditions for the non Area-preserving and AP fits,
        #this gets overwritten in the run_prep section, leaving here for now
        self.fitinit = [self.center[0], 2.0,self.center[1], 2.0, 0.]
        self.AP_fitinit = [self.center[0],2.0,self.center[1], 0.]
        #Empty variable, will be the return from the ODR fit
        self.fitbeta = None
        #Is the fit good assumed False, see ODRFit functions for conditions
        self.fitgood = False
        #Measurement of how well the fit actually fits, used to determine if merges improve the result
        self.fiterr = None
        #records stoppping condition of the ODR, <4 is good, >4 is bad, 4 is OK
        self.fitinfo = None

        
        #has the boulder been assessed?
        self.measured = None

        #boulder morphometry
        #boulder width in pixels and meters
        self.bouldwid = None
        self.bouldwid_m = None
        #height of boulder in pixels and meters
        self.bouldheight = None
        self.bouldheight_m = None
        self.bouldheight_m_actual = None
        #center of fit ellipse
        self.bouldcent = [None, None]
        #semi-axis along the sun direction in pixels and meters
        self.shadlen = None
        self.shadlen_m = None
        #placeholder for the patch object
        self.ellipsepatch = None

    def run_prep(self):
        #main function that does most things we want it to do
        
        #self.findborder()
        self.findborder_cents()
        if len(self.border) != 0:
            flipval = self.mirror()
            self.fitinit = [flipval, 2.0,self.center[1], 2.0, 0.]
        
    def run_fit(self):
        #to change the kind of border fit used, alter this line
        if len(self.mborder)!=0:
            self.odrfit_m()

    def run_post(self):
        #turned off this guard to see if large ones are being tossed
        #if self.fitgood:
        self.shadowmeasure_m()
        
    def findborder(self):
        #we will look at each point and see if it has neighbors, if not, there is a border
        #this cannot be the most efficient way to do this....
    
        for i in self.pixels:
            top=False
            other = False
            if [i[0]-1,i[1]] not in self.pixels:
                self.border+= [[i[0]-.5, i[1]]]
                self.border+= [[i[0]-.5, i[1]+.5]]
                self.flipaxis+= [[i[0]-.5, i[1]]]
                self.flipaxis+= [[i[0]-.5, i[1]+.5]]
                top = True
                
            if [i[0]+1,i[1]] not in self.pixels:
                self.border+= [[i[0]+.5, i[1]]]
                self.border+= [[i[0]+.5, i[1]-.5]]
                self.mborder+= [[i[0]+.5, i[1]]]
                self.mborder+= [[i[0]+.5, i[1]-.5]]
                

            if [i[0],i[1]+1] not in self.pixels:
                self.border+= [[i[0], i[1]+.5]]
                self.border+= [[i[0]+.5, i[1]+.5]]
                if not top:
                    self.mborder+= [[i[0], i[1]+.5]]
                    self.mborder+= [[i[0]+.5, i[1]+.5]]
                

            if [i[0],i[1]-1] not in self.pixels:
                self.border+= [[i[0], i[1]-.5]]
                self.border+= [[i[0]-.5, i[1]-.5]]
                if not top:
                    self.mborder+= [[i[0], i[1]-.5]]
                    self.mborder+= [[i[0]-.5, i[1]-.5]]
    def findborder_cents(self):
        '''
        Alternate method for finding the shadow border, uses pixel cetners rather than pixel edges
        This may generally struggle with shadows that are linear
        must set self.border, self.mborder, self.flipaxis in otder to swap in for findborder
        '''

        #quick check for linear shadows and small shadows, these need to be tossed
        xs = [i[1] for i in self.pixels]
        ys = [i[0] for i in self.pixels]
        if min(xs) == max(xs) or min(ys) == max(ys):
            #this is a linear shadow, it will bomb the ODR fitting
            return
        if len(self.pixels) <=minA:
            #technicaly should have been filtered out earlier, but better safe than sorry
            return
        for i in self.pixels:
            top=False
            other = False
            bord = False
            mirror = False
            if [i[0]-1,i[1]] not in self.pixels:
                self.border+= [i]
                self.flipaxis+= [[i[0]-.5,i[1]]]
                top = True
                bord = True
                
            if [i[0]+1,i[1]] not in self.pixels:
                self.mborder+= [i]
                mirror = True
                if not bord:
                    self.border+= [i]
                    
            if [i[0],i[1]+1] not in self.pixels:
                if not bord:
                    self.border+= [i]
                if not top and not mirror:
                    self.mborder+= [i]
                    
            if [i[0],i[1]-1] not in self.pixels:
                if not bord:
                    self.border+= [i]
                if not top and not mirror:
                    self.mborder+= [i]
        return
        
    def mirror(self):
        ''' this takes the boulder shadow and border and flips it over the
            sun_perpendicular vector, which is the x axis in all cases.
            By  doing this we can fit an ellipse to what is actually an ellipse,
            theoretically giving better fits. the points flip along the MIN (most sunward)
            pixel value, 
            '''
        #step 1, find the flip axis
        yvals = np.array(self.flipaxis)
        yvals = yvals[range(len(yvals)),[0]]
        #this is the flip value
        #flipval = np.average(yvals)
        flipval = np.min(yvals)
        self.mborder = [[float(k[0]),float(k[1])] for k in self.mborder]
        #make a copy of the mborder, which lacks the -y boundary
        temp = np.copy(self.mborder)
        dist = range(len(temp))
        #invert that about the flip value
        temp[dist,[0]]*=-1
        temp[dist,[0]]+=2*flipval
        #mborder is stored as a list, lets keep it that way
        temp = temp.tolist()
        #append it and you are done
        self.mborder+=temp
        
        return flipval 

    def odrfit_m(self):
        input_y = list(map(lambda f:f[0], self.mborder))
        input_x = list(map(lambda f:f[1], self.mborder))
        input_dat = [input_y, input_x]
        fit_data = odr.Data(input_dat, y=1)
        fit_model = odr.Model(self.ellipse, implicit=True)
        fit_odr = odr.ODR(fit_data, fit_model, self.fitinit)
        #print 'doing ODR'
        
        fit_out = fit_odr.run()
        
        #print 'ODR done'
        self.fitinfo = str(fit_out.info)

        self.fitbeta = fit_out.beta
        self.fiterr = fit_out.sum_square

        area = abs(np.pi*self.fitbeta[1]*self.fitbeta[3])
        if area> MA or self.fitbeta[3]*2 > MD:
            self.fitgood = False
        elif self.fitinfo == "2" or self.fitinfo == "3" or self.fitinfo == "1" or self.fitinfo == '4':
            self.fitgood = True
        return
        #fit_out.pprint()
        
    def shadowmeasure_m(self):
        '''shadow measuring now that we are doubling the shadow, very straightforward'''
        if len(self.mborder) == 0:
            self.bouldwid = mindist*2
            self.shadlen = 0
            self.bouldcent = self.center
            self.measured = True
            self.fitgood = True
            self.fitbeta = [self.center[0],mindist,self.center[1],mindist,0]
            self.bouldheight = self.shadlen/np.tan(np.radians(self.inangle))
            self.bouldwid_m = self.bouldwid*self.resolution
            self.shadlen_m = self.shadlen*self.resolution
            self.bouldheight_m = self.bouldheight*self.resolution
            return
        #despite not being constrained, the fit ellipses are pretty much either veritcal or horizonal
        # so np.cos(alpha) is essentially either 0,1, or -1, or at least close to it. With this
        #non-zero results (~1 or -1) will be negative, others will be positive
        
        test = .5 - abs(np.cos(self.fitbeta[4]))

        #build in exception here for small boulders
        
        #with a minimum of four pixels, no problems were had
        
        if test <= 0:
            factor = np.cos(self.fitbeta[4])
            #factor = 1
            self.bouldwid = 2*abs(factor*self.fitbeta[3])
            self.shadlen = abs(factor*self.fitbeta[1])
            self.bouldcent = [self.fitbeta[0],self.fitbeta[2]]
            self.bouldheight = self.shadlen/np.tan(np.radians(self.inangle))
            self.measured = True
        if test > 0:
            factor = np.sin(self.fitbeta[4])
            #factor = 1
            self.bouldwid = 2*abs(factor*self.fitbeta[1])
            self.shadlen = abs(factor*self.fitbeta[3])
            self.bouldcent = [self.fitbeta[0],self.fitbeta[2]]
            self.bouldheight = self.shadlen/np.tan(np.radians(self.inangle))
            self.measured = True
        self.bouldwid_m = self.bouldwid*self.resolution
        self.shadlen_m = self.shadlen*self.resolution
        self.bouldheight_m = self.bouldheight*self.resolution
        if self.bouldwid > MD or self.bouldheight > MH:
            self.measured = False
        return
        
    def ellipse (self, beta, coords):
        '''note alpha (beta[4]) is recorded in radians'''
        y = coords[0]
        x = coords[1]

        yc = beta[0]
        xc = beta[2]
        ay = beta[1]
        ax = beta[3]
        alpha = beta[4]
        #alpha is the clockwise angle of rotation
        #alpha = np.arctan(self.sunangle[1]/self.sunangle[0])
    
        val = (((y-yc)*np.cos(alpha)+(x-xc)*np.sin(alpha))/ay)**2 + (((x-xc)*np.cos(alpha)-(y-yc)*np.sin(alpha))/ax)**2 - 1
        
        #print beta
        return val

    def patchplot(self,filt):
        #this will be the new ellipse plotting function that uses the matplotlib patches function
        #need to figure out how to make sure they are plotting correctly though...
        #filter controls whether big ones will be plotted
       
        color = 'g'
        alpha = .2
        if not filt:
            angle = np.degrees(np.pi-(self.fitbeta[4]%(2*np.pi)))
            shadow = patches.Ellipse([self.fitbeta[2],self.fitbeta[0]], self.fitbeta[3]*2., self.fitbeta[1]*2., angle, color = color, alpha = alpha)
            boulder = patches.Circle([self.bouldcent[1],self.bouldcent[0]], (self.bouldwid/2.), alpha=alpha)
            return [shadow, boulder]
        if self.measured:
            angle = np.degrees(np.pi-(self.fitbeta[4]%(2*np.pi)))
            shadow = patches.Ellipse([self.fitbeta[2],self.fitbeta[0]], self.fitbeta[3]*2., self.fitbeta[1]*2., angle, color = color, alpha = alpha)
            boulder = patches.Circle([self.bouldcent[1],self.bouldcent[0]], (self.bouldwid/2.), alpha=alpha)
            return [shadow, boulder]
        else:
            return []

            
##############Some tools for analysis##########
def gauss(x,sig,mu):
    y = (1./(sig*np.sqrt(2.*np.pi)))*np.exp((-1./2.)*(((x-mu)/sig)**2.))
    return y
def gauss_unnorm(x,sig,mu):
    y = np.exp((-1/2)*((x-mu)/sig)**2)
    return y

def current():
    print ('Current Path is: %s'%(PATH))
    print ('Current Filename is: %s'%(FNM))
    print ('current Product ID is: %s'%(ID))
    return
def getshads(runfile, num, silenced = True, mode='rb'):
    #open and prep a shadow file, returns open file object and endpoint
    #current()
    try:
        load = open('%s%s%s%s_shadows.shad'%(PATH,runfile,FNM,num), mode)
    except IOError:
        if not silenced:
            print ("No shadow file exists")
        return None
    except:
        if not silenced:
            print('Likely broken file')
        return None
    return load
def bulkCFA(runfile,maxnum,maxd,fitmaxd,root):
    ''' runs the CFA protocol on a bunch of files and gives an average
'''
    #set value for maximum on plots (in meters)
    plotmax = 3
    while True:
        new = raw_input('make new CDFs? y/n\n')
        if new == 'y' or new == 'n':
            break
    allCFAs = []
    if new == 'n':
        for i in range(maxnum):
            try:
                data = open('%s%s%s%s_CFA.csv'%(PATH,runfile,FNM,i),'r')
                dat = np.loadtxt(data,delimiter=',')
                
            except(IOError):
                continue

            allCFAs+=[dat]
        if allCFAs == []:
            print ("No CDFs Present")
            new = 'y'
    if new == 'y':
        for i in range(maxnum):
            #acuire all the CFA data, this will come in the form:
            #CFA[0] = bins
            #CFA[1] = lower error bound CFA
            #[2] = actual CFA data
            #[3] = Upper bound error CFA
            dat = CFA(runfile,i, maxd)
            #Keep an eye on this, would break quickly if CFA is changed
            if len(dat) == 3:
                #plt.plot(dat[0],dat[2],'b-',alpha=.05)
                allCFAs+=[dat]
                #print len(dat)

##    if not any(allCFAs):
##        return None, None
    #extract bins and the three CFAs (upCFAs = upper end of CFA uncertainty, downCFAs lower end of uncertainty)
    #above method is being revised, shifting to a sigma calculation
    bins = allCFAs[0][0]
    allCFAs = np.array(allCFAs)
    #CFAs = allCFAs[:,2]
    #upCFAs = allCFAs[:,1]
    #downCFAs = allCFAs[:,3]
    CFAs = allCFAs[:,1]
    CFAsigmas = allCFAs[:,2]
    #tranpose the files to go from (image,bins) to (bins,image) format
    T_CFAs = np.transpose(CFAs)
    T_CFAsigmas = np.transpose(CFAsigmas)
    #T_upCFAs = np.transpose(upCFAs)
    #T_downCFAs = np.transpose(downCFAs)
    #average accross bins, results in 1-D array of length 'bins'
    #also fetch a 25% and 75% CFA curve
    avgCFAs = map(np.average,T_CFAs)
    topqCFAs = map(lambda x: np.percentile(x,75),T_CFAs)
    botqCFAs = map(lambda x: np.percentile(x,25),T_CFAs)
    avgCFAsigmas = []
    
    for i in T_CFAsigmas:
        sigma = (1/float(len(i)))*(np.sqrt(sum(map(lambda x: x**2,i))))
        avgCFAsigmas+=[sigma]
    avgupCFAs = []
    avgdownCFAs = []
    for i in range(len(avgCFAs)):
        avgupCFAs+=[avgCFAs[i]+avgCFAsigmas[i]]
        avgdownCFAs+=[avgCFAs[i]-avgCFAsigmas[i]]
    #avgupCFAs = map(np.average,T_upCFAs)
    #avgdownCFAs = map(np.average,T_downCFAs)
##    print bins
##    print CFAs
##    print upCFAs
##    print downCFAs
    #fit to each end of the spectrum
    fit_k, fit_r2 = fittoRA(bins,avgCFAs, [1.5,fitmaxd])
    #upfit_k, upfit_r2 = fittoRA(bins,avgupCFAs, [1.5,maxd])
    #downfit_k, downfit_r2 = fittoRA(bins,avgdownCFAs, [1.5,maxd])
    upfit_k, upfit_r2 = fittoRA(bins,avgupCFAs, [1.5,fitmaxd])
    downfit_k, downfit_r2 = fittoRA(bins,avgdownCFAs, [1.5,fitmaxd])
    topqfit_k, topqft_r2 = fittoRA(bins,topqCFAs,[1.5,fitmaxd])
    botqfit_k, botqft_r2 = fittoRA(bins,botqCFAs,[1.5,fitmaxd])
    
    #calculate RA for each side of uncertainty
    fit_bins = np.linspace(min(bins),max(bins),100)
    fitRA = GolomPSDCFA(fit_bins,fit_k)
    upfitRA = GolomPSDCFA(fit_bins,upfit_k)
    downfitRA = GolomPSDCFA(fit_bins,downfit_k)
    topqfitRA = GolomPSDCFA(fit_bins,topqfit_k)
    botqfitRA = GolomPSDCFA(fit_bins,botqfit_k)

    #plot them all!
    #errors in format of array [2,bins] all must be positive, lower errors first, see matplotlib API
    errors = []
    errors+=[fitRA-downfitRA]
    errors+=[upfitRA-fitRA]
    #plots stack in reverse, so plot what you want on top first (I think...)
    plt.plot(bins,avgCFAs,label = root,zorder=3, c = 'g', marker='|')
    
    plt.errorbar(fit_bins,fitRA,zorder=2,label = 'RA Envelope',yerr = errors,ecolor = 'k',c = 'k',marker='|',alpha=.5)
    plt.plot(fit_bins,topqfitRA,zorder=3,label = '75th Percentile RA')
    plt.plot(fit_bins,botqfitRA,zorder=3,label = '25th Percentile RA')
    plotCFArefs(plotmax)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin= 1, xmax = plotmax)
    plt.xlabel('Boulder Diameter (m)')
    plt.ylabel('Cumulative Fractional Area')
    plt.ylim(ymin=10**(-5))
    plt.legend(loc=3)
    plt.title('CFA for image %s at parameters %s'%(root,runfile))
    plt.savefig('%s%sCFAPlot.png'%(PATH,runfile))

    #save the CFA data:
    savecfa = open('%s%sCFAdata_maxd_%s_fitmaxd_%s.csv'%(PATH,runfile,maxd,fitmaxd),'w')
    savecfa.write('Bins,')
    for i in bins:
        savecfa.write('%s,'%i)
    savecfa.write('\n')
    savecfa.write('avgCFA,')
    for i in avgCFAs:
        savecfa.write('%s,'%i)
    savecfa.write('\n')
    savecfa.write('sigma,')
    for i in avgCFAsigmas:
        savecfa.write('%s,'%i)
    savecfa.write('\n')
    savecfa.write('topq,')
    for i in topqCFAs:
        savecfa.write('%s,'%i)
    savecfa.write('\n')
    savecfa.write('botq,')
    for i in botqCFAs:
        savecfa.write('%s,'%i)
    savecfa.write('\n')
    savecfa.close()
    #plt.show()
    return fit_k,topqfit_k,botqfit_k, fit_r2
    
def CFA(runfile,num,maxd):
    #Produces data for Cumulative Fractional Area, saves and produces plot
    #plt.show must be called after to plot all the data
    load = getshads(runfile,num)
    sizes = []
    if not load:
        return [None]
    while True:
        try:
            dat = pickle.load(load)
        except EOFError:
            break
        if dat.measured:
            sizes+=[dat.bouldwid_m]
    
            res = dat.resolution
            #may need to re-run some boulder detection to get this into the shad file
            im_area = dat.im_area
    if not any(sizes):
        return [None]
    #im = imageio.imread('%s%s%s.PNG'%(PATH,FNM,num))
    #im = npma.array(im)
    #get the image area in meters
    #im_area = float(len(im.compressed()))*res*res
    #print area
    #create the two ends of the uncertainy spectrum, assumes 1 pixel uncertainty
    err = 1
    sigma = err*res
    sizes = np.asarray(sizes)
    
    sizes = [x for x in sizes if x < maxd]
    
    sizes = np.asarray(sizes)
    
    bins = np.linspace(0,maxd,20*maxd+1)
    SFD, binsconf = np.histogram(sizes, bins=bins)
    #SFD = np.append(SFD,0)
    SFD = SFD/im_area
    sizes.sort()

    #Total  Area, sum of all boudlers
    #TA = sum(areas)
    #CFA will be a list of y points, with the x points in sizes
    CFA = np.zeros_like(bins)
    CFAsigma = np.zeros_like(bins)
    #upCFA = np.copy(CFA)
    #downCFA = np.copy(CFA)
    for i in range(len(bins)):
        CFA[i] = float(sum(map(lambda x: np.pi*((x/2.)**2),sizes[sizes>bins[i]])))/im_area
        CFAsigma[i] = sigma*np.sqrt((np.pi/im_area)*CFA[i])
        #upCFA[i] = float(sum(map(lambda x: np.pi*((x/2.)**2),upsizes[upsizes>bins[i]])))/im_area
        #downCFA[i] = float(sum(map(lambda x: np.pi*((x/2.)**2),downsizes[downsizes>bins[i]])))/im_area
    #save1 = [bins,upCFA.tolist(),CFA.tolist(),downCFA.tolist()]
    save1 = [bins,CFA.tolist(), CFAsigma.tolist()]
    save2 = [binsconf[:-1].tolist(),SFD.tolist()]
    cfasavefile = open('%s%s%s%s_CFA.csv'%(PATH,runfile,FNM,num),'w')
    np.savetxt(cfasavefile, save1,delimiter=',')
    sfdsavefile = open('%s%s%s%s_SFD.csv'%(PATH,runfile,FNM,num),'w')
    np.savetxt(sfdsavefile, save2,delimiter=',')  
##    CFA[0] = TFA-(areas[0]/area)
##    for i in range(1,len(sizes)):
##        CFA[i] = CFA[i-1] - (areas[i]/area)

    #plt.plot(bins, CFA, 'b-', alpha=.01)
    #plt.xscale('log')
    #plt.yscale('log')
    #return [bins,upCFA.tolist(),CFA.tolist(),downCFA.tolist()]
    return [bins, CFA.tolist(), CFAsigma.tolist()]

def plotCFArefs(xmax):
    ''' plots data from golombek 2008 for comparison, user controlled which one
'''
    query = 'Plot: 1= just reference \n 2= just TRA_000828_2495 \n 3=Sholes PSP_007718_2350 Hand Counts \n 5=all\n'
    option = int(raw_input(query))
    if option == 1 or option == 5:
        #dat = np.loadtxt('%sGolomRefCFACurves.csv'%(REFPATH),delimiter=',')
        xs = np.linspace(.1,xmax)
        y20 = GolomPSDCFA(xs,.2)
        y30 = GolomPSDCFA(xs,.3)
        y40 = GolomPSDCFA(xs,.4)
        y50 = GolomPSDCFA(xs,.5)
        x = np.tile(xs,4)
        y = np.concatenate((y20,y30,y40,y50))
        plt.plot(x,y,'m*',alpha=.2,label = '20,30,40,50% RA',zorder=1)
    if option == 2 or option == 5:
        print ('cannot do until data is provided')
        #dat = np.loadtxt('%sGolomCFADat_TRA_000828_2495.csv'%(REFPATH),delimiter=',')
        #plt.plot(dat[0],dat[1],'k*',alpha=.7)
    if option == 3 or option == 5:
        dat = np.loadtxt('%sPSP_007718_2350Ref.csv'%(REFPATH),delimiter=',')
        plt.plot(dat[0],dat[1],'b*',label='Manual Results')
    return
    
def PSD(k):
    ''' the particle size distribution for the martian surface, according to Charalambous 2014 (dissertation
'''
    #These values based on Charambolous
    t = 3
    p = .79
    return sp.special.binom(t+k-1,k)*(p**k)*(1-p)**t

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
    popt,pcov = sp.optimize.curve_fit(GolomPSDCFA,fit_xdat,fit_ydat,p0=[.1])
    #calculate the R2 on the fit maybe??
    ybar = np.average(fit_ydat)
    SStot = np.sum((fit_ydat-ybar)**2)
    predicted = GolomPSDCFA(fit_xdat,popt)
    SSres = np.sum((fit_ydat-predicted)**2)
    R2 = 1-SSres/SStot
    return popt, R2

def checkbads(runfile,num):
    #this is for when you want to get all the shadows that went awry and plot them
    load = getshads(runfile,num)
    bads = []
    while True:
        try:
            dat = pickle.load(load)
        except EOFError:
            break
        if not dat.measured:
            bads+=[dat]
    patches = []
    for j in bads:
        j.shadowmeasure()
        patches+=j.patchplot()
    fig=plt.figure(1)
    ax = fig.add_subplot(111)
    #image = np.load('%s%s%s%s_rot_masked.npy'%(PATH, runfile,FNM, num))
    
    plt.imshow(image, cmap='binary_r')
    for j in patches:
        ax.add_patch(j)
    plt.show()
    return bads

    
    
def ExamineImage(runfile,num, showblanks,filt = True):
    hasshads = True

    if hasshads:
        load = getshads(runfile,num)
        #need two so you dont reuse same artist, silly but necessary
        patches1 = []
        patches2 = []
        while True:
            try:
                dat = pickle.load(load)
            except EOFError:
                break
            patches1 += dat.patchplot(filt)
            patches2 +=dat.patchplot(filt)
        #image = np.load('%s%s%s%s_rot_masked.npy'%(PATH,runfile,FNM,num))
        image = imageio.imread('%s%s//%s%s.PNG'%(PATH,FNM,FNM,num))
        segimage = np.load('%s%s//%s%s%s_SEG.npy'%(PATH,FNM,runfile,FNM,num),allow_pickle=True)
        image = sktrans.rotate(image,ROTANG, resize=True, preserve_range=True)
        #segimage = sktrans.rotate(segimage,ROTANG, resize=True, preserve_range=True)
        image = npma.masked_equal(image, 0)
        filtimage = np.load('%s%s//%s%s%s_flagged.npy'%(PATH,FNM,runfile,FNM,num),allow_pickle=True)
        
        fig,ax = plt.subplots(2,2,sharex = True, sharey = True, figsize=(30,30))
        ax[0][0].imshow(image, cmap='binary_r', interpolation='none')
        ax[0][1].imshow(image,cmap='binary_r',interpolation='none')
        ax[1][0].imshow(segimage,interpolation='none')
        ax[1][1].imshow(filtimage,interpolation='none')
        for j in patches1:
            ax[0][1].add_patch(j)
        for j in patches2:
            ax[1][1].add_patch(j)
        plt.show()
##        plt.figure(1)
##        plt.imshow(image)
##        plt.figure(2)
##        plt.imshow(sktrans.rotate(image, -ROTANG))
##        plt.show()
    elif showblanks:
        #image = np.load('%s%s%s%s_rot_masked.npy'%(PATH,runfile,FNM,num))
        image = imageio.imread('%s%s//%s%s.PNG'%(PATH,FNM,FNM,num))
        image = sktrans.rotate(image,ROTANG, resize=True, preserve_range=True)
        image = npma.masked_equal(image, 0)
        filtimage = np.load('%s%s//%s%s%s_flagged.npy'%(PATH,FNM,runfile,FNM,num),allow_pickle=True)
        fig,ax = plt.subplots(2,2,sharex = True, sharey = True)
        ax[0][0].imshow(image, cmap='binary_r', interpolation='none')
        ax[0][1].imshow(image,cmap='binary_r',interpolation='none')
        ax[1][0].imshow(filtimage,interpolation='none')
        ax[1][1].imshow(filtimage,interpolation='none')
        plt.show()
    return
        
        
def FindBigs(runfile,num,diam = 3):
    '''code to find and identify large boulders thay may be causing issues in the CFA'''
    shads = []
    load = getshads(runfile,num)
    if load == None:
        return None
    while True:
        try:
            dat = pickle.load(load)
            shads+=[dat]
        except(EOFError):
            break
    bigs = []
    patches = []
    total = 0
    for i in shads:
        if i.bouldwid_m>diam:
            bigs+=[i]
            patches+= i.patchplot()
        if i.measured:
            total+=1
    if np.any(bigs) == False:
        return None
    tossout = float(len(bigs)/total)
    #image = np.load('%s%s%s%s_rot_masked.npy'%(PATH,runfile,FNM,num))
    #replacing this with the original image to save on drive space.
    image = imageio.imread('%s%s%s.PNG'%(PATH,FNM,num))
    image = sktrans.rotate(image,ROTANG, resize=True, preserve_range=True)
    image = npma.masked_equal(image, 0) 
    fig,ax = plt.subplots(1,2, sharex = True, sharey = True)
    ax[0].imshow(image,cmap='binary_r',interpolation='none')
    ax[1].imshow(image,cmap='binary_r',interpolation='none')
    for j in patches:
        ax[1].add_patch(j)
    return bigs
def FindExcluded(runfile,maxnum,maxdiam):
    '''finds how many boulders were ignored due to exclusion of large boulders'''
    total = long(0)
    used = long(0)
    for i in range(maxnum):
        shads = []
        load = getshads(runfile,i)
        if load == None:
            continue
        while True:
            try:
                dat = pickle.load(load)
                shads+=[dat]
            except(EOFError):
                break
        for j in shads:
            if j.measured:
                total+=1
                if j.bouldwid_m<maxdiam:
                    used+=1
    exclpercent = 100.*(float(total-used)/float(total))
    print ('%s boulders found in image, %s percent ignored due to diameter'%(total,exclpercent))
    return
    
def ManualMerge(runfile,num,flags):
    '''Code to manually merge two boulders, only to be used in exception circumstances
        runfile and num specify the image, boulders listed in flags will be merged into one with lowest flag value.
        '''
    shads = getshads(runfile,num,mode = 'rb')
    boulds=[]
    mergeboulds = []
    while True:
        try:
            dat = pickle.load(shads)
        except(EOFError):
            break
        if dat.flag in flags:
            mergeboulds += [dat]
        else:
            boulds+=[dat]
    print (len(mergeboulds))
    shads.close()
    shads = getshads(runfile,num,mode='wb')
    for obj in boulds:
        pickle.dump(obj,shads)
    if len(mergeboulds) == len(flags):
        finalflag = mergeboulds[0].flag
        finalarea = mergeboulds[0].im_area
        finalpixels = []
        for i in mergeboulds:
            finalpixels+=i.pixels
        
        newbould=shadow(finalflag, finalpixels, finalarea)
        newbould.run_prep()                  
        newbould.run_fit()
        newbould.run_post()
        pickle.dump(newbould,shads)
        shads.close
        print ('succesfully merged input boulders with flags %s'%(flags))
        return

    else:
        print('Did not find %s boulders, aborting merge'%(len(flags)))
        for obj in mergeboulds:
            pickle.dump(obj,shads)
        shads.close()
        return
        
    
    
##Very important##
#Looks like dlow and dhigh are not used??
def OutToGIS(runfile,writefile,maxnum,dlow = 1.0, dhigh = 5,extension='.PGw'):
    '''this code will take an entire run and export the boulder data to a GIS-interpretable format, assume pngs for the moment'''
    ''' A key part of this is interpreting the PGW files, which follow this convention:
        6 values on 6 lines:
        A
        D
        B
        E
        C
        F
        these are inputs to two equations for xmap and ymap, the coordinates of the pixel in the map, based on x and y, the pixel coordinates of the image:
        xmap = Ax + By +C
        ymap = Dx + Ey +F
    '''
    if not os.path.exists('%sGISFiles//%s'%(PATH,writefile)):
        os.makedirs('%sGISFiles//%s'%(PATH,writefile))
    datafile = open("%sGISFiles//%s%s_All_boulderdata.csv"%(PATH,writefile,FNM),'w')
    datafile2 = open("%sGISFiles//%s%s_Clean_boulderdata.csv"%(PATH,writefile,FNM),'w')
    #put in the headers, we will start small with the boulders:
    headers = 'image,flag,xloc,yloc,bouldwid_m,bouldheight_m,shadlen,measured,fitgood,fiterr,angle\n'
    datafile.write(headers)
    datafile2.write(headers)
    #bring in the original rotation information
    rotang_r = np.radians(ROTANG)
    for i in range(maxnum+1):
        #bring in the image each time, wouldnt have to do this if they were all identical
        #but that cant be gauranteed
        try:
            seg = np.load('%s%s%s%s_flagged.npy'%(PATH,runfile,FNM,i),allow_pickle = True)
        except(IOError):
            continue
        shads = getshads(runfile,i)
        if not shads:
            print('no shads for %s\n'%(i))
            continue
        lycent = len(seg)/2.
        lxcent = len(seg[0])/2.
        seg = None
        o_image = imageio.imread('%s%s%s.PNG'%(PATH,FNM,i))
        o_lycent = len(o_image)/2.
        o_lxcent = len(o_image[0])/2.
        o_image = None
        
        worldfile = open('%s%s%s%s'%(PATH,FNM,i,extension),'r')
        constants = []
        for line in worldfile:
            val = float(line.rstrip())
            constants+=[val]
        boulds = []
        marks = []
        while True:
            try:
                dat  = pickle.load(shads)
            except:
                break
            #taking out the filter for now to see if good ones are getting tossed
            try:
                flag = dat.flag
                bouldwid_m = dat.bouldwid_m
                bouldheight_m = dat.bouldheight_m
                shadlen = dat.shadlen
                AR = float(bouldheight_m/bouldwid_m)
                angle = dat.fitbeta[4]
                try:
                    fiterr = dat.fiterr
                except:
                    fiterr = None
                #GIS doesnt like mixing data types in csv
                measured = int(dat.measured)
                fitgood = int(dat.fitgood)
                #weve got to do something a little tricky here, we need the pixel location of the boulders in the un-rotated images
                #first we make a pointmap with flags to keep track
                #boulds+=[dat]
               
                xpos = dat.bouldcent[1]
                ypos = dat.bouldcent[0]
            except:
                print ("failed to retrieve parameters")
                continue
                #change xpos and ypos to origin on the image center
            xpos_c = xpos-lxcent
            ypos_c = ypos-lycent

            #rotate them, must give the negative rotation
            xpos_rot_c = xpos_c*np.cos(rotang_r) - ypos_c*np.sin(rotang_r)
            ypos_rot_c = xpos_c*np.sin(rotang_r) + ypos_c*np.cos(rotang_r)

            #re-reference to the corner of the image
            xpos_rot = xpos_rot_c + o_lxcent
            ypos_rot = ypos_rot_c + o_lycent
            xmap = constants[0]*xpos_rot + constants[2]*ypos_rot + constants[4]
            ymap = constants[1]*xpos_rot + constants[3]*ypos_rot + constants[5]
            #write it all down
            info = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%(i,flag,xmap,ymap,bouldwid_m,bouldheight_m,shadlen,measured,fitgood,fiterr,angle)
            datafile.write(info)
            #putting the aspect ratio filter in here, going to use numbers based on Demidov and Basilevsky 2014
            #average = ~.5
            #stdev = ~.28
            #low boundary = .22
            #high boundary = .78
            if shadlen!=0 and bouldwid_m<20 and fitgood == 1:
                datafile2.write(info)
                
                
        shads.close()           
    datafile.close()
    datafile2.close()
    return
            
        
        
            
    
def LROCAdapter():
    '''this code is  intended as a quick fix to looking at LROC images, longer term
better infrastructure should be put in place to make this smoother'''
    #the only real problem is reaching for the metadata, we have to shortcut this
    #these are called when MBARS initializes and with MBARS.start, so we can call this instead and it shoudl work
    
    INANGLE = None
    NAZ = None
    SAZ = None
    SUNANGLE = None
    RESOLUTION = None

    return


def plotborder(array):
    #this is to speed up plotting up borders
    #array is in form [[y,x],[y,x]....]
    ydat = []
    xdat = []
    l = len(array)
    array = np.copy(array)
    ydat = array[range(l),[0]*l]
    xdat = array[range(l),[1]*l]
    plt.plot(xdat, ydat, "o")
    return

##Also Very important
def getangles(ID, path = REFPATH):
    '''
    input is the product ID of the HiRISE image, returns key observation values:
    Incedence angle (sun's angle below peak
    sun direction (emission angle of the sun, clockwise from right of image)
    Resolution(m/px)
    North Azimuth (clockwise north direction from right hand side of image
    Sun Azimuth (
    '''
    cumindex = open(path+'RDRCUMINDEX.TAB','r')
    for line in cumindex:
        try:
            dat = line
        except EOFError:
            print ('No such HiRISE image, check product ID or update CUMINDEX file')
            return None, None, None
        dat = dat.split(',')
        pid = dat[5]
        pid = pid.replace(" ","")
        pid = pid.replace('"','')
        if pid == ID:
            break
    """ key factors we are looking for in the TAB file are as follows:
        Attribute                INDEX
        Product ID:             5
        SubSpacecraft lat/long: 29/30
        SubSolar lat/long:      27/28
        Incedence Angle:        20
        North Azimuth:          25
        Sun Azimuth:            26
        Resolution (m/px):             39
        Projection_type:        41     
    """
    
    naz = float(dat[25])
    saz = float(dat[26])
    inangle = float(dat[20])
    glat = float(dat[29])
    glong = float(dat[30])
    slat = float(dat[27])
    slong = float(dat[28])
    res = float(dat[39])
    #the issue is most apparent when the slon and glon are >180 apart. I think this can be fixed (see groundaz)
    sunangle = groundaz(glat,glong,slat,slong)

    if NOMAP:
        rotang = 90+saz
    else:
        projection = dat[41]
        projection = projection.replace("\"","")
        projection = projection.rstrip()
        if projection == 'EQUIRECTANGULAR':
            #print 'Map Projection is listed as %s, make sure to project to Equirectangualr projection in GIS before Paneling'%(dat[40])
            rotang = sunangle
        elif projection == 'POLAR STEREOGRAPHIC':
            rotang = -glong+sunangle
        else:
            print ('Projection listed as %s, I am error'%(projection))
            return (None)

    ''' Inangle is returned as measured from azimuth, sunangle as clockwise from North'''
    
    # #HiRISE_INFO has the resolution, PID is on index 11, resolution on the last one
    # info = open(path+'HiRISE_INFO.txt','r')
    # for line in info:
    #     try: dat=line
    #     except EOFError:
    #         print ('No such HiRISE image, check product ID or update CUMINDEX file')
    #         return None, None, None
    #     dat = dat.split(',')
    #     if ID == dat[11]:
    #         break
    # res = dat[-1].strip()
    # res = float(res)
    
    
        
    return inangle, sunangle, res, naz, saz, rotang

  #Very important do not touch      
def groundaz(glat, glon, slat, slon):
    """
    Translated directly from the ISIS GroundAzimuth C function
    inputs and outputs in degrees
    first pair is ground lat/lon, second is either subsun or subspacecraft
    Originally Authored by Caleb Fassett
    """

    if (glat >= 0):
        a=radians(90-slat)
        b=radians(90-glat)

    else:
        a=radians(90+slat)
        b=radians(90+glat)

    cslon=slon
    cglon=glon
    if cslon>cglon:
        if ((cslon-cglon)>180):
            while ((cslon-cglon)>180):
                cslon=cslon-360

    if cglon>cslon:
        if ((cglon-cslon)>180):
            while ((cglon-cslon)>180):
                cglon=cglon-360
    #I think if this was  changed to evaluate on cglon and cslon instead of glat and glon, it would work.
    #Did the above change, Hopefully it fixes it
    if (slat>=glat):
        if (cslon>=cglon):
            quad=1

        else:
            quad=2

    else:
        if (cslon>=cglon):
            quad=4

        else:
            quad=3

    dlon=radians(glon-slon)
    if dlon<0:  dlon=-dlon
    
    c=acos(cos(a)*cos(b)+sin(a)*sin(b)*cos(dlon))
    az=0

    if (((sin(b)==0)|(sin(c)==0))|((sin(b)*sin(c))==0)):
        
        return az

    else:

        try:
            bigA=degrees(acos((cos(a)-cos(b)*cos(c))/(sin(b)*sin(c))))
        except:
            return az

        ### This is a kludge -- domain errors (perhaps near 180 lon)

        if (glat>=0):
            if ((quad==1)|(quad==4)):
                az=bigA

            else:
                az=360-bigA

        else:
            if ((quad==1)|(quad==4)):
                az=180-bigA

            else:
                az=180+bigA

        return az

def RunParams(filename):
    '''a long term solution to running multiple images, this will
create text files in each image file so that various parameters dont need to be
reentered, files read:
root
ID
NOMAP
Panels
'''
    parampath = '%s%s//runparams.txt'%(BASEPATH,filename)
    if not os.path.isdir('%s%s'%(BASEPATH,filename)):
        print ('no such directory\n')
        return None,None,None,None
    if os.path.isfile(parampath):
        print ('Running Parameters found\n')
        #get these parameters
        paramfile = open(parampath,'r')
        info = paramfile.readline()
        info = info.rstrip()
        #print(info)
        root,mbarsid,mbarsnomap,panels = info.split(',')
##        root = paramfile.readline()
##        root = root.rstrip
        
##        mbarsid = paramfile.readline
##        mbarsid = mbarsid.rstrip()
        
##        mbarsnomap = paramfile.readline()
        if 'True' in mbarsnomap:
            mbarsnomap = True
        else:
            mbarsnomap = False

##        panels = paramfile.readline()
##        panels = panels.rstrip()
        panels = int(panels)

        
    else:
        print ("no parameters found, making new file\n")
        
        params = open(parampath,'w')
        q1 = 'enter MBARS ID\n'
        mbarsid = raw_input(q1)
        
        q2 = 'Is the image projected (i.e. north is up?) y/n \n'
        while True:
            answer = raw_input(q2)
            if answer == 'y':
                mbarsnomap = False
                
                break
            elif answer == 'n':
                mbarsnomap = True
                break
            else:
                print ('y/n \n')
        #retrieve number of panels
        files = os.listdir('%s%s'%(BASEPATH,filename))
        files = [s for s in files if '.PNG' in s]
        files = [s.replace(filename,'') for s in files]
        #print(files)
        files = [filter(lambda s: s in '0123456789',j) for j in files]
        #These had to be added for python 3 compaitbility with changes to filter()
        #Should be python 2 stable, but untested
        files = [list(j) for j in files]
        files = [''.join(j) for j in files]
        #print(files)
        files = [int(s) for s in files]
        panels = np.max(files) +1
        print('%s panels found'%panels)

        root = filename

        #write it all in:
        params.write('%s,'%root)
        params.write('%s,'%mbarsid)
        params.write('%s,'%mbarsnomap)
        params.write('%s'%panels)
        params.close()

    return root, mbarsid, mbarsnomap, panels
        

########INITIALIZATION##########
INANGLE,SUNANGLE,RESOLUTION,NAZ, SAZ, ROTANG = start()


    
