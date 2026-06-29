import moten
import matplotlib.pyplot as plt

# Stream and convert the RGB video into a sequence of luminance images
video_file = r"C:\Users\Sipe_Lab\Desktop\2P\Experiment Types\Mesofield-experiment\mp4\mov8.mp4"
luminance_images = moten.io.video2luminance(video_file)
# plt.plot(luminance_images[2])  # Plot the first frame of the luminance images   
# plt.show()  # Show the plot of the first frame

# Create a pyramid of spatio-temporal gabor filters
nimages, vdim, hdim = luminance_images.shape
# pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)
pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(vdim, hdim), stimulus_fps=24)
print(pyramid)  # Should print the pyramid structure and filter details

# Compute motion energy features
# moten_features = pyramid.project_stimulus(luminance_images)

# print(moten_features.shape)  # Should print (nimages, nfilters)

features = pyramid.project_stimulus(luminance_images)
print(features.shape)
# fig, ax = plt.subplots(figsize=(12, 12))
# ax.matshow(features, aspect='auto')
# plt.show()

animation = pyramid.show_filter(600)
plt.show()
# from turbustat.statistics import PowerSpectrum
# from astropy.io import fits

