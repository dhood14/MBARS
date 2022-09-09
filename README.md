# MBARS
A body of code developed in order to locate, identify, and measure boulders on the Martian Surface.


# Environment
MBARS is designed to be compatible with Python 2.7 and 3+, however, compatibility with Python 2.7 is not assured for all future versions. Python depends on the following libraries
 - numpy
 - scipy
 - matplotlib
 - pickle
 - sci-kit image (skimage)
 - sci-kit learn (sklearn)
 - imageio
 - threading

Many of these will be pre-loaded with a standard python distrubtion.

# Setup
MBARS is designed to be flexible, but relies on several file structures in order to operate. MBARS will also build files in a predictable manner during use.
At the top of the MBARS.py file which describs the core functions of MBARS, two paths are declared: the REFPATH and the BASEPATH. The REFPATH should provide the complete path to the folder where the
RDRCUMINDEX file, which records various metadata on HiRISE images, is stored (https://hirise-pds.lpl.arizona.edu/PDS/INDEX/). The BASEPATH is the assumed path to the folder where most work will be done. Changing these paths within the MBARS.py file will change them permenantly, and these can be changed on a temporary basis as needed.

The MBARS.py file is the library of MBARS, on its own it does not do anything. MBARS_Run is designed to access MBARS and apply it to images. MBARS_RUN outputs files in many locations, which it will create when run.
 - MBARS.PATH = *, The MBARS PATH is set for each run in MBARS_RUN and points to a folder within MBARS.BASEPATH that contains the specific image files MBARS will operate on.
 - */autobound, This is where MBARS will output segmented images and records of boulder locations.
 - */GISfiles/autobound_##, This is where MBARS will output the final boulder list, ## reflects the chosen shadow boundary percentile.

The csv files located within the /autobound_## folders can be loaded into a GIS using the same coordinate system as the original image.

# Workflow

A workflow to apply MBARS to HiRISE images is described in the Hood et al., 2022 paper and restated here in a more direct, how-to fashion.
1. Install MBARS, and download the RDRCUMINDEX File (https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB) from the NASA PDS
2. Check the BASEPATH and REFPATH as set in the MBARS.py file point to the base folder and the location of the RDRCUIMINDEX.TAB file
3. Download the HiRISE Image to be analyzed (henceforth, "the image")
3a. MBARS relies on projection information output by the GIS, so whatever projeciton you use will be inherited
4. Load the image into a GIS Software, partition the image into manageable pieces, export to PNGs
4a. This can be achieved via the Split Raster tool in ArcGIS
4b. Make sure that the output files are organized as "FILENAME0.png, FILENAME1.png..." and are placed in BASEPATH/FILENAME, such that the path to any given image is BASEPATH/FILENAME/FILENAME#.png
4c. Order among the partitions is not considered in MBARS, so the nature of the partitioning scheme (starting point, enumerated in rows/columns) does not matter
4d. For HiRISE images, partiions sizes of 500-1000 pixels worked well
5. MBARS_Run accepts a list of filenames (formatting details are in the MBARS_Run.py), and a list of FRAC values, which are the boundary parameters. Images will be run in order, and on each provided FRAC value (A_10, A_20...A_90, B_10, B_20...)
5a. Default FRAC values are 10, 20, 30, 40, 50, 60, 70, 80, 85, 90. Values should be between 0 and 100, with higher values, leading to larger shadows and longer runtimes.
6. The results will be output to BASEPATH/FILENAME/GISFiles/autobound_##, one result for each setting of the shadow boundary
7. Each .csv file in the autobound_## folders contains the unified list of all boulders in the image, these can be imported into GIS
7a. For ArcGIS, importing via the "Add XY Events" tool works well.

Each resultant list of boulder objects is an represetnation of the boulders in the image based on an assumed shadow boundary. The best solution for any given HiRISE image, or even a particualr area within a HiRISE Image may vary due
to factors such as soil albedo, lighting conditions, photometric properties of the surface, etc. How to decide among these outputs and choose the best solution for any specific application bcomes rapidly divergent, and is therefore not
discussed here. Details on one approach to calibrate MBARS for application to the martian northern lowlands and be found in Hood et al 2022 (To Be Submitted)

# Publications
 - The Martian Boulder Automatic Recognition System, MBARS (https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022EA002410, 2022)
 - The Martian Boulder Automatic Recognition System: Comparison to Old and New Techniques (https://www.hou.usra.edu/meetings/lpsc2022/pdf/1483.pdf, LPSC 2022)
 - Large-Scale Assessment of Polygon-Edge Boulder Clustering in the Martian Northern Lowlands (https://www.hou.usra.edu/meetings/lpsc2020/pdf/2620.pdf, LPSC 2020)
 - Verification of Automatically Measured Boulder Populations in HiRISE Images (https://www.hou.usra.edu/meetings/lpsc2019/pdf/1893.pdf, LPSC 2019)
 - Automated Boulder Detection and Measuring in HiRISE images (https://www.hou.usra.edu/meetings/lpsc2019/pdf/1893.pdf, LPSC 2018)
 - Semi-Automated Measurement of Boulder Clustering in the Martian Northern Plains (Abstract: https://www.hou.usra.edu/meetings/lpsc2017/pdf/2640.pdf, Eposter: https://www.hou.usra.edu/meetings/lpsc2017/eposter/2640.pdf, LPSC 2017)
 
# Authors
 - Don R. Hood, Primary Developer and Researcher
 - Suniti Karunatillake, Science and Development Advisor
 - Caleb I. Fassett, Science and Development Advisor
 - Stephen F. Sholes, Collaborator and Data provider
 - Ryan C. Ewing, Science Advisor
 - Peter James, Science Advisor
 - James P. Brothers, User Testing and Boulder measurement
 - Aviv Cohen-Zada, Alpha Tester
 
# Acknowledgements
Testing, development, and application of this algorithm have been funded at various stages by both the Louisiana Space Consortium (LaSPACE) and the NASA Mars Data Analysis Program
(Grant #80NSSC21K1093D). Drs. Karunatillake and Fassett are credited for developing many of the original ideas and applications of what would ultimately become MBARS, before passing
along those ideas to Dr. Hood as a part of his Ph.D. Dissertation work.

# License
This work is published under the MIT license, see LICENSE.md for details
