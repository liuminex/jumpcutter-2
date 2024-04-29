'''
this is jumpcutter2.py, an optimized version of original jumpcutter.py at github repo:
	
	https://github.com/carykh/jumpcutter

jumpcutter.py needed 60GB and more of free space to jump-cut a video of 1 hour long. My lectures are
1.5-2.5 hours long so my computer's free space wasn't enough and i needed to optimize it so i made
some changes and renamed it to jumpcutter2.py

There are still a lot of changes that can be done to save space and time. Maybe you can do it at
jumpcutter3.py!

Always open source and free!

Read more to find out:
	1) how to use
	2) changes compared to jumpcutter.py
	3) performace stats
	4) about splits
	5) performance for long videos
	6) notes about the code for easier understanding while you create jumpcutter3!

how to use:

IMPORTANT: the last part where i merge the splitted videos didn't work all the times so i skip it for
	now. To complete the jump-cut you must merge them manually. (suggested program for this:
	Avidemux. It only takes seconds)

you only need jumpcutter.py or jumpcutter2.py to jump-cut videos

first install the packages with:
	
	pip3 install <package name>
	
	packages:
		
		Pillow
		audiotsm
		scipy
		numpy
		pytube
for splitting
	
	install mkvmerge
		
		$ sudo sh -c 'echo "deb https://mkvtoolnix.download/ubuntu/ $(lsb_release -sc) main" >> /etc/apt/sources.list.d/bunkus.org.list'
		$ wget -q -O - https://mkvtoolnix.download/gpg-pub-moritzbunkus.txt | sudo apt-key add -
		$ sudo apt-get update
		$ sudo apt-get install mkvtoolnix mkvtoolnix-gui

to merge final videos: /*currently not working*/
	
	pip install moviepy
	
for last splitted video
	
	pip install opencv-python

install MKVToolNix: (used for splitting video)
	
	sudo sh -c 'echo "deb https://mkvtoolnix.download/ubuntu/ $(lsb_release -sc) main" >> /etc/apt/sources.list.d/bunkus.org.list'
	
	wget -q -O - https://mkvtoolnix.download/gpg-pub-moritzbunkus.txt | sudo apt-key add -
	
	sudo apt-get update
	
	sudo apt-get install mkvtoolnix mkvtoolnix-gui
	
( to uninstall it: sudo apt-get remove --autoremove mkvtoolnix mkvtoolnix-gui )
	
	
you can now use
	
	python3 jumpcutter2.py -h
	
	to find commands

jumpcutter2 works only for .mp4 videos because:
	1) input_file adds ".mp4" as extension
	2) maybe something else too, idk

run this to cut silence:
	example: for blabla.mp4 -> [filename] = blabla
	
	python3 jumpcutter2.py --input [filename]
	
	if resolution is very bad (and lower than original video's) DECREASE frame_quality with
	(argument):
	
		--frame_quality x
		
		(where 1 <= x <= 31)


code changes compared to jumpcutter.py original:

usage:
url arg do download from youtube was removed, also removed if...else for this case and renamed
input_file to input for easier use, set default arg variables to best for university class lectures.
I also added filesIdentifier for two reasons:
 1) simplicity, so i don't need to specify output name
 2) so that i can jump-cut the same file multimple times by executing the same command

technical:
in order to reduce storage used during processing, i set default frame_quality to 20, for lectures
with already low quality you should decrease it probably at 5 (to increase quality). For the same
reason i set default frame_rate to 10 (some videos were at 30 fps, my records were at 15), this
didn't reduce stroage used, so i first convert the video to my desired fps, and made some adjustments
in order to keep the audio from the original video (higher quality audio). This fps-thing reduces
storage used because instead of splitting the video in 30 fps it does it in 10 so 3 times less
storage - the frames extracted from the original video is the biggest issue because for a 40-second
video it used a total of 150 MB of frames when it broke it down, so this plays a big role here - if
after all the storage is still full you can reduce it even more with

	--frame_rate x

and you wiil only lose quality of video frames (video, no audio). Another huge change was the
adjustments in the algorithm of cutting silence. Here the original program copied every final frame
resulting in having almost every frame two times, which means double space. To fix this, insted of
copying them, i deleted the ones i didn't want. But to do this i removed the option to play silent
frames at higher speed - instead i delete them.
The most effective change is the SPLITS feature. See description below (after "next day performance")

performance:
after reducing frame rate (only):
	In a video with duration=40 seconds, space used BY EXTRACTED FRAMES (at frame_quality=5
	because it was	already low, no copies, no audio) was reduced from 104.2 MB to 34.9 !
	(it was 30 fps now it is 10 - math checks).
	Execution time remained about the same.
after changing deleting-copying algorithm (final):
	Execution time remained about the same. For the 40 second video, storage used was 85 MB using
	the old code (duplicate frames), now is uses 50 MB.
all were measured using frame_quality 5, by using frame_quality 20 you save a lot more space

PERFORMANCE MEAURED THE NEXT DAY:
using --frame_quality 20, --frame_rate 10:

for the 34-second video (original: 15 fps, high quality image):
	executed in  20  seconds
	peak space used: 30.48 MB
for the 40 second video (original: 30 fps, low quality image):
	executed in  10  seconds.
	peak space used: 33.73 MB



S P L I T S
Then, i added a very imporatnt feature:
	
	--splits x

this splits input video into n parts of x duration (in seconds) and creates n output videos and then
merges them so maximum storage used is n times smaller! Simple but very effective. Default is
x = 300 (5 mins) (because the videos i edit are 1.5-2.5 hours duration. You can change this with
--splits 999 for example.
like that (example to split video every 3000 seconds):
	python3 jumpcutter2.py --input myfile --splits 3000
	
minimum recommended splits (seconds) is 10 (i put a limit on this in code, but you can remove it)

splits performance (using --splits 10):
	
	merging takes a lot of time...
	
	for the 40-second video:
		executed in 22 sec.
		peak space used: 11.8 MB.
	for the 34-second video:
		executed in 35 sec.
		peak space used: 17.3 MB.

conclusion: for these short videos the difference is not that big, it now uses 11-17 MB instead of
	30-33 MB, but this is the minimum is will ever use (depends on quality also, but will not
	differ much) this means that for a 100 hour long video the maximum space used will be
	video'sSize + 17 MB!
	
	PROBLEM SOLVED!
	
note that the lower --splits value is, the slower the program executes. This didn't make much
difference for these short videos, but it may do in a longer video. When i test it i will write the
results here. 



LONG VIDEOS PERFORMANCE

an estimation i make is that for 10 seconds of video it uses less than 25 MB of space so for my long
videos i will use --splits 400 estimating that i will use a maximum of 1 GB of my space (+ the size
of the input video, so if input video is 1 GB, actual max space used will be 2 GB).

	test video 1: 15 fps, 1 hour 30 minutes duration:
		using 	--splits 400 (default)
			--frame_quality 20 (default)
			and all default settings
		
		executed in 32 mins
		peak storage used 412 MB
	
	test video 2: 30 fps, 2 hours 10 minutes duration:
		using 	--splits 600
			--frame_quality 5
			and all default settings
		
		executed in 64 mins
		peak storage used 1001.4 MB


code notes

what chunks are:
#chunk = [index of previous chunks frame, index of this chunks frame, 1/0 include/not include]
#for example chunk[3,12,0] means from frame 3 to 12 the frames should not be included (0)
#chunks is another form - more compact - to express shouldIncludeFrame


example:

['shouldIncludeFrame'] :
[0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1.]
 
['chunks'] :
[[0, 3, 0.0], [3, 37, 1.0], [37, 40, 0.0], [40, 110, 1.0], [110, 112, 0.0], [112, 185, 1.0],
[185, 190, 0.0], [190, 220, 1.0], [220, 223, 0.0], [223, 272, 1.0], [272, 281, 0.0], [281, 370, 1.0],
[370, 373, 0.0], [373, 393, 1.0], [393, 402, 0.0], [402, 406, 1.0]]

about frame margin:
	example: if frame_margin=2 then if there are <=4 consecutive frames with low sound (silent)
	they will not be removed, if there are 5, there will be included only 1, if there are x
	silent, there will be included the x-4 central frames. That happens for average cases. For
	corner cases where the silent ane in the beggining or the end of the video, there will be
	included x-2 frames. This happens in part 2/3 (see code)
'''

#original imports
from contextlib import closing
from PIL import Image
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
import argparse

#for merging videos
from moviepy.editor import VideoFileClip, concatenate_videoclips

#for last splitted video, get info
import cv2
import datetime

#for filesIndetifier
from datetime import datetime
from datetime import date

#for speed diagnostics
import time

#for diagnostics using def p(var)
import inspect

#for estimations confirm
import sys

#for filesIdentifier
now=datetime.now()
today = date.today()

#for debugging

#print a variable
def p(var):
	callers_local_vars = inspect.currentframe().f_back.f_locals.items()
	name = [var_name for var_name, var_val in callers_local_vars if var_val is var]
	print(name,":")
	print(var)

#for diagnostics

#measure space, return in MB, time in seconds
def space(fol="."):
	tstart=time.time()
	
	start_path = fol
	sz = 0
	for dirpath, dirnames, filenames in os.walk(start_path):
		for f in filenames:
			fp = os.path.join(dirpath, f)
			if not os.path.islink(fp):
				sz += os.path.getsize(fp)
	#from bytes to MB
	sz = sz/(1024*1024)
	tend=time.time()
	return [sz,tend-tstart]
	
def spaceFormat(size):
	post = "MB"
	if size > 1023:
		size = size/1024
		post = "GB"
	return str(float("{0:.1f}".format(size))) + " " + post

#for program

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def deleteFrame(index,tempFolder):
	f = tempFolder+"/frame{:06d}".format(index+1)+".jpg"
	if os.path.exists(f):
		os.remove(f)
	else:
		print("The file ",f," does not exist")

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createPath(s):
    try:  
        os.mkdir(s)
    except OSError:  
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

def deletePath(s): # Dangerous! Watch out!
    try:  
        rmtree(s,ignore_errors=False)
    except OSError:  
        print ("Deletion of the directory %s failed" % s)
        print(OSError)

def deleteFile(f):
	if os.path.exists(f):
		os.remove(f)
	else:
		print("The file "+str(f)+" does not exist")

parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
parser.add_argument('--input', type=str,  help='the video file you want modified (only name, no extension like this: --input videoname, only mp4, change this from code)')
parser.add_argument('--silent_threshold', type=float, default=0.1, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")
parser.add_argument('--sounded_speed', type=float, default=1.1, help="the speed that sounded (spoken) frames should be played at. Typically 1.")
#frame margin (see details at intro)
parser.add_argument('--frame_margin', type=float, default=1, help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")
#sample rate below 34100 resulted in longer output videos (less silent frames were cut)
parser.add_argument('--sample_rate', type=float, default=34100, help="audio sample rate of the input and output videos")
#CURRENTLY YOU MUST ADD FRAME RATE >= of actual OF VIDEO ELSE IT WILL FREEZE FROM A POINT AND AFTER
parser.add_argument('--frame_rate', type=float, default=10, help="video frame rate of the input and output videos. (not-optional anymore - but u can see fps and add it from data in start of program)")
parser.add_argument('--frame_quality', type=int, default=3, help="quality of frames to be extracted from input video, use less memory on disk (highest=1,lowest=31)")
parser.add_argument('--splits', type=int, default=9000, help="maximum storage space to use. Consider 16 MB per 10 splits. Attention: splits value can't be too low for long videos. Minimum splits = 10. Default is 600 which translates to 1 GB. See description in code for more information.")

args = parser.parse_args()


#general
inputName = args.input
INPUT_FILE = inputName+".mp4"

INPUT_FILE_REDUCED_FRAMES = "newInputVideoReducedFpsNoAudio.mp4"
	
filesIdentifier=today.strftime("%Y%m%d")+now.strftime("%H%M%S")
PARENT_TEMP_FOLDER = "TEMP"+"_"+inputName+filesIdentifier
createPath(PARENT_TEMP_FOLDER)

frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [999999.0, args.sounded_speed]

FRAME_QUALITY = args.frame_quality

assert inputName != None , "error: no input file"

OUTPUT_FILE = inputName+"_jumped_"+filesIdentifier+".mp4"

SPLITS = args.splits
if SPLITS < 10:
	SPLITS = 10


#estimations confirm
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")



#jumpcutter
#returns PEAK space used in MB, execution time in seconds, time spent measuring space
def jumpcut(INPUT_FILE,TEMP_FOLDER,splitNo):
	
	timeStart = time.time()
	
	TEMP_FOLDER = PARENT_TEMP_FOLDER+"/"+TEMP_FOLDER
	createPath(TEMP_FOLDER)
	
	SPLIT_OUTPUT_FILE = "split"+str(splitNo)+"_jumped.mp4"

	#re-encode video at different fps
	command = "ffmpeg -i "+INPUT_FILE+" -r 10 -y "+TEMP_FOLDER+"/"+INPUT_FILE_REDUCED_FRAMES
	subprocess.call(command, shell=True)

	#idk what this is, doesn't matter for my goal
	AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)
	    
	#extract frames from video
	command = "ffmpeg -i "+TEMP_FOLDER+"/"+INPUT_FILE_REDUCED_FRAMES+" -qscale:v "+str(FRAME_QUALITY)+" "+TEMP_FOLDER+"/frame%06d.jpg -hide_banner"
	subprocess.call(command, shell=True)

	#create audio file from input video
	command = "ffmpeg -i "+INPUT_FILE+" -ab 160k -ac 2 -ar "+str(SAMPLE_RATE)+" -vn "+TEMP_FOLDER+"/audio.wav"
	subprocess.call(command, shell=True)

	#we dont need this anymore so let's free some space
	deleteFile(TEMP_FOLDER+"/"+INPUT_FILE_REDUCED_FRAMES)

	#sample rate is default 34100, audio data is array of audio data
	sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")

	#number of audio samples the audio has
	audioSampleCount = audioData.shape[0]

	maxAudioVolume = getMaxVolume(audioData)

	#audio samples in one video frame
	samplesPerFrame = sampleRate/frameRate

	#total video frames = duration*frameRate
	audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))

	hasLoudAudio = np.zeros(audioFrameCount)

	#part 1/3: mark wanted video frames based on loundness only

	#for i in video frames
	for i in range(audioFrameCount):
		#index of first audio sample in this video frame
		start = int(i*samplesPerFrame)
		#end last audio sample in this video frame
		end = min(int((i+1)*samplesPerFrame),audioSampleCount)
		#put the audio samples of this frame in array // call them CHUNKS
		audiochunks = audioData[start:end]
		#max chunk volume
		maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume
		if maxchunksVolume >= SILENT_THRESHOLD:
			#if chunk volume level is greater than user's defined
			#mark this video frame as keeper
			hasLoudAudio[i] = 1


	#part 2/3: filter with frame_margin, add result in chunks

	#for chunks see code notes in description on top of document
	chunks = [[0,0,0]]
	shouldIncludeFrame = np.zeros((audioFrameCount))
	#for i in video frames
	for i in range(audioFrameCount):
		#for some reason FRAME_MARGIN := FRAME_SPREADAGE
		#index of first frame
		start = int(max(0,i-FRAME_SPREADAGE))
		#index of last frame
		end = int(min(audioFrameCount,i+1+FRAME_SPREADAGE))
		#hasLoudAudio is either 0 or 1 from previous loop
		#start:end are the frames inside the frame margin
		shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
		#this is how to get the parts of consecutive non-silent frames:
		if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]): # Did we flip?
			chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]])


	#add last chunk
	chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])
	#remove first item from chunks[]
	chunks = chunks[1:]

	#part 3/3

	outputAudioData = np.zeros((0,audioData.shape[1]))

	outputPointer = 0

	lastExistingFrame = None

	#measure space for performance stats
	space1 = space(PARENT_TEMP_FOLDER)
	s1 = space1[0]
	c1 = space1[1]


	#for every part (chunk) of the chunks:
	keeperFrames = np.zeros(audioFrameCount)
	for chunk in chunks:
		#from here until "marker 1" is probaably only about audio, so i didn't study it
		
		#array of audiodata inside ecery chunk
		audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)]
	    
		sFile = TEMP_FOLDER+"/tempStart.wav"
		eFile = TEMP_FOLDER+"/tempEnd.wav"
		wavfile.write(sFile,SAMPLE_RATE,audioChunk)
		with WavReader(sFile) as reader:
			with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
				tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
				tsm.run(reader, writer)
		_, alteredAudioData = wavfile.read(eFile)
		leng = alteredAudioData.shape[0]
		endPointer = outputPointer+leng
		outputAudioData = np.concatenate((outputAudioData,alteredAudioData/maxAudioVolume))
		
		# smooth out transitiion's audio by quickly fading in/out
		if leng < AUDIO_FADE_ENVELOPE_SIZE:
			outputAudioData[outputPointer:endPointer] = 0 # audio is less than 0.01 sec, let's just remove it.
		else:
			premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE
			mask = np.repeat(premask[:, np.newaxis],2,axis=1) # make the fade-envelope mask stereo
			outputAudioData[outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask
			outputAudioData[endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask

		#changed code here
		startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
		endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
		for outputFrame in range(startOutputFrame, endOutputFrame):
			inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))
			keeperFrames[inputFrame] = 1

		outputPointer = endPointer

	#remove last frame
	keeperFrames[len(keeperFrames)-1] = 0
	#delete unwanted frames
	for ind in range(len(keeperFrames)):
		if keeperFrames[ind] == 0:
			deleteFrame(ind,TEMP_FOLDER)

	#measure space for performance stats
	space2 = space(PARENT_TEMP_FOLDER)
	s2 = space2[0]
	c2 = space2[1]

	#rename remaining frames
	jpg_files = []
	for x in os.listdir(TEMP_FOLDER):
		name, ext = x.split('.')
		if ext == 'jpg':
			jpg_files.append(x)
	jpg_files.sort()
	count = 1
	for i in jpg_files:
		dst = TEMP_FOLDER+"/frame{:06d}".format(count)+".jpg"
		count = count + 1
		os.rename(TEMP_FOLDER+"/"+i,dst)

	#create new audio file
	wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData)

	#create final video
	command = "ffmpeg -framerate "+str(frameRate)+" -i "+TEMP_FOLDER+"/frame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -strict -2 "+PARENT_TEMP_FOLDER+"/"+SPLIT_OUTPUT_FILE
	subprocess.call(command, shell=True)
	
	#measure space for performance stats
	space3 = space(PARENT_TEMP_FOLDER)
	s3 = space3[0]
	c3 = space3[1]

	#end

	#DELETE PATH WHEN FINISHED
	deletePath(TEMP_FOLDER)

	return [max(s1,s2,s3),time.time()-timeStart,c1+c2+c3]


#program run
ans = query_yes_no('estimated peak storage is '+spaceFormat(SPLITS*2)+', continue?');
if ans == False: quit();







t0=time.time()

#split original video in duration/SPLITS parts
command = "mkvmerge -o "+PARENT_TEMP_FOLDER+"/split --split "+str(SPLITS)+"s "+INPUT_FILE
subprocess.call(command, shell=True)

#find numberOfSplits
count = 1
for x in os.listdir(PARENT_TEMP_FOLDER):
	count = count + 1

numberOfSplits = count - 1

if numberOfSplits <= 0:
	print("\033[91msplits value is too big (it has to be smaller than the videos duration)\033[0m")
	deletePath(PARENT_TEMP_FOLDER)
	quit()

#jump-cut each splitted video
jumps=[]
spaces = []
mspaces = []

for splitIndex in range(1,numberOfSplits+1):
	
	INPUT_FILE = PARENT_TEMP_FOLDER+"/split-{:03d}".format(splitIndex)

	frames = cv2.VideoCapture(INPUT_FILE).get(cv2.CAP_PROP_FRAME_COUNT)
	if frames < 3:
		print("started deleting")
		deleteFile(INPUT_FILE)
		print("it was deleted")
		numberOfSplits = numberOfSplits - 1
		splitIndex = splitIndex - 1
	else:
		jump = jumpcut(INPUT_FILE,"SPLIT_"+str(splitIndex)+"_TEMP",splitIndex)
		spaces.append(jump[0])
		mspaces.append(jump[2])
		deleteFile(INPUT_FILE)
		jumps.append(jump)

#merge the outputted splitted videos
#MERGING BRICKS MY COMPUTER (sometimes), SO I MERGE THEM MANUALLY (it is also faster this way)
'''
videosToMerge = []
for i in range(1,numberOfSplits+1):
	videoFile = PARENT_TEMP_FOLDER + "/split" + str(i) + "_jumped.mp4"
	vid = VideoFileClip(videoFile)
	videosToMerge.append(vid)

final_video = concatenate_videoclips(videosToMerge)
final_video.write_videofile(OUTPUT_FILE)
'''
#DELETE PATH WHEN FINISHED
#deletePath(PARENT_TEMP_FOLDER)

#print end message
print("\n\nfinished jump-cutting, performance:\n")

for splitIndex in range(1,len(jumps)+1):
	jump = jumps[splitIndex-1]
	print("split",splitIndex,"executed in",float("{0:.1f}".format(jump[1]))," sec, peak space was",spaceFormat(jump[0]))

peak = max(spaces)
mspace = sum(mspaces)
etime = int((time.time()-t0)/60)
print("\ntotal:")
print("executed in",etime,"min.")
print("peak space used:",spaceFormat(peak)+".")
#this is only for diagnostics
#print("time spent measuring space:",float("{0:.3f}".format(mspace)),"sec.")

