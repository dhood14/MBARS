import MBARS
import matplotlib.pyplot as plt
import os

'''
Catch all code to look at results,

Because the MBARS_RUN.py file now does everything out to exporting to GIS,
this code is potentially defunct. May still have some utility for troubleshooting though.

'''



#fnm = 'ESP_073077_2155_RED_1000PX'
fnm = 'PSP_001556_2460_RED_1000PX'
#Joes Images
#filename = 'PSP_007693_2300_RED16bit500PX'
#runfile = 'gam600_manbound173//'

#Proposal Test Images
#filename = 'PSP_001415_2470_RED500PX'
#runfile = 'gam600_manbound85//'
#filename = 'PSP_001415_2470_RED16bit500PX'

runfile='autobound//'

#filename='PSP_001481_2410_RED16bit_500PX'
#filename='PSP_001473_2480_RED16bit_500PX'
#filename='PSP_001430_2470_RED16bit_500PX'
#filename='PSP_001418_2495_RED16bit_500PX'
#filename='PSP_001482_2490_RED16bit_500PX'
#filename='PSP_001484_2455_RED16bit_500PX'
#filename='PSP_001742_2370_RED16bit'
#filename='PSP_001741_2395_RED16bit_500PX'

#specify which running file to examine, include '//' on the end of the name

#runfile = 'gam600_manbound147//'


MBARS.FNM = fnm
MBARS.PATH = MBARS.BASEPATH+fnm+'//'
root, MBARS.ID, MBARS.NOMAP, panels = MBARS.RunParams(fnm)
MBARS.INANGLE, MBARS.SUNANGLE, MBARS.RESOLUTION, MBARS.NAZ, MBARS.SAZ, MBARS.ROTANG = MBARS.start()


#######################PICK THE ANALYSES##################
ManualMerge = False
OutToGIS = False
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


# #check for the proper path:
if not os.path.exists('%s%s'%(MBARS.PATH,runfile)):
    print ('runfile does not exist for this image, check runfile')
    #exit()
#where the actual work happens

#Do not know if this works
if ManualMerge:
    query = 'Did you mean to do a manual Merge?\n y/n?\n'
    answer = input(query)
    if answer != 'y':
        print ('OK I wont do it')
    else:
        MBARS.ManualMerge(runfile,MMnum,MMflags)
        
#Might work, but also done at the end of each run
if OutToGIS:
    MBARS.OutToGIS(runfile,'autobound_test//',panels)

if MakeCFAs:
    record = open('%s%s_record.csv'%(MBARS.PATH,MBARS.FNM),mode='wb')
    record.write('Filename ,Best Fit Rock Abundance,R^2,Upper limit RA, lower limt RA, maxd=%s\n'%(maxd))
    plt.figure(runfile)
    fit_k,upfit_k,downfit_k, fit_r2 = MBARS.bulkCFA(runfile,panels+1,maxd,2.25,root)

    print('Best fit rock abundance for file %s is %s percent, up to %s, or down to %s with an R^2 of %s'%(runfile,fit_k*100,upfit_k*100,downfit_k*100, fit_r2))
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
        num = input(query)
        try:
            MBARS.ExamineImage(runfile,num, showblanks)
        except(IOError):
            print ("Failed to examine, image does not exist")
        query = 'Look at another? y/n\n'
        answer = input(query)
        if answer != 'y':
            break


