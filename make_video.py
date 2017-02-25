# You need to install MoviePy to create the movie. Otherwise, it'll only create the PNG frames.
# See http://zulko.github.io/moviepy/install.html for MoviePy installation instructions.

from PIL import Image, ImageDraw
import os
import sys
import math

CLEANUP = True


# Get command-line argument
assert (len(sys.argv) == 2), "Please provide the data file as a single command-line argument."
infile = os.path.realpath(sys.argv[1])

# Make output folder if it doesn't exist
outfolder = os.path.realpath(infile[:-4])
if not os.path.exists(outfolder):
  os.mkdir(outfolder)

print "Reading from data file {0}".format(infile)
print "Output will be in folder {0}".format(outfolder)

# Create PNGs in <outfolder> by reading the input file
f = open(infile, 'r')

# Read header line
line = f.readline().split(" ")
num_particles = int(line[0])
width = float(line[1])
height = float(line[2])
total_time = float(line[3])
num_time_steps = int(line[4])
time_steps_per_frame = int(line[5])
num_frames = int(math.ceil(float(num_time_steps) / time_steps_per_frame))

print "Reading data file from simulation with the following parameters:"
print "\tNumber of particles: {0}".format(num_particles)
print "\tBox width: {0}".format(width)
print "\tBox height: {0}".format(height)
print "\tTotal simulation time: {0}".format(total_time)
print "\tNumber of time steps: {0}".format(num_time_steps)
print "\tNumber of time steps per frame: {0}".format(time_steps_per_frame)
print "\tNumber of frames: {0}".format(num_frames)

pngs = [[]] * num_frames

density = num_particles / (width * height)
if density > 1/16.0:
  transparency = int(256 * (1 / density / 16))
else:
  transparency = 255

for frames_processed in range(num_frames):
  frame_num = int(f.readline())
  outfile = os.path.realpath(os.path.join(outfolder, "frame{0}.png".format(frame_num)))

  im = Image.new("RGBA", (int(width), int(height)), "white")

  draw = ImageDraw.Draw(im)

  for particle in range(num_particles):
    line = f.readline().split(" ")
    if len(line) != 3:
      print "WARNING: Expected a line of particle data, but got \"{0}\"".format(" ".join(line))

    x = float(line[0])
    y = float(line[1])
    mass = float(line[2])

    rad = max(1.5, math.sqrt(mass/math.pi)) - 0.5

    min_x = x - rad
    max_x = x + rad
    min_y = y - rad
    max_y = y + rad

    min_x = min(max(min_x, 0), int(width - 1))
    max_x = min(max(max_x, 0), int(width - 1))
    min_y = min(max(min_y, 0), int(height - 1))
    max_y = min(max(max_y, 0), int(height - 1))

    #dot = Image.new("RGBA", (int(width), int(height)), (255,255,255,0))
    #draw = ImageDraw.Draw(dot)
    color = (0, 0, 0, transparency)
    draw.ellipse([(min_x, min_y), (max_x, max_y)], fill = color, outline = color)

    #im = Image.alpha_composite(im, dot)


  im.save(outfile)
  pngs[frame_num] = outfile

  if frame_num % 100 == 0:
    print "Saved frame {0} in file: {1}".format(frame_num, outfile)

f.close()

import moviepy.editor as mpy

fps = num_frames / total_time * 10
mp4_outfile = os.path.realpath(os.path.join(outfolder, "movie.mp4"))
gif_outfile = os.path.realpath(os.path.join(outfolder, "movie.gif"))

video = mpy.ImageSequenceClip(pngs, fps = fps)

video.write_videofile(mp4_outfile, fps = fps, write_logfile = True)
print "Video output to: {0}".format(mp4_outfile)

#video.write_gif(gif_outfile, fps = fps)
#print "GIF output to: {0}".format(gif_outfile)

if CLEANUP:
  [os.remove(png) for png in pngs]