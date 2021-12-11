import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Plot the class value counts as a barplot 
def barplot_classes(X, LCC):
  classes,class_counts = np.unique(X[:,-1], return_counts=True)
  fig,ax = plt.subplots()
  x = np.linspace(1,len(class_counts),len(class_counts))
  plot1 = ax.bar(x, class_counts)
  ax.set_xticks(x)
  ax.set_xticklabels([LCC[i] for i in classes], rotation=90)
  plt.subplots_adjust(bottom=0.19)
  plt.rc('font', size=18)
  plt.rc('xtick', labelsize=18)
  plt.rc('axes', labelsize=18) 
  plt.show()
  return

# Plot the distribution of class values as a barplot
def plot_class_distr(X,id=None):
  # Convert to numpy array
  X = np.array(X)
  # Number of instance
  n = len(X)
  # Number of features including labels
  m = len(X[0])
  # Get classes and class counts from the class/label column
  classes,class_counts = np.unique(X[:,-1], return_counts=True)
  num_classes= len(classes)
  # Storage for the indexes of dataset columns of each class
  class_idxs = [[] for i in range(num_classes)]
  # Get the indexes of pixels in each class
  for col in range(n):
    class_idx = np.where(classes==X[col][-1])[0][0]
    class_idxs[class_idx].append(col)
  # Convert to numpy arrays
  for i in range(num_classes):
    class_idxs[i] = np.array(class_idxs[i])
  # Split the data by class
  data_by_class = [X[class_idxs[i]] for i in range(num_classes)]
  # Gather band data across all classes
  bands = list()
  for i in range(m-1):
    band_data = list()
    for j in range(num_classes):
      band_data_by_class = list()
      for k in range(len(data_by_class[j])):
        band_data_by_class.append(data_by_class[j][k][i])
      band_data.append(band_data_by_class)
    bands.append(band_data)
  # The bands array is now of shape (num_bands, num_classes, pixels) and contains the data for each class in each band
  for j in range(len(bands)):
    for i in range(len(bands[j])):
      vals,counts = np.unique(bands[j][i], return_counts=True)
      plt.plot(vals,counts, label=i)
    plt.legend()
    if id!=None:
      fname = "assets/"+id+"_band" + str(j+1) + "_class_distr.png"
    else:
      fname = "assets/band" + str(j+1) + "_class_distr.png"
    plt.savefig(fname)
    plt.clf()
  return

# Plot the frequencies of values contained in each band as a barplot
def plot_band_counts(X,id=None):
  n = len(X)
  m = len(X[0])
  # Storage for band unique vals and counts
  band_vals = [[[],[]] for i in range(m)]
  # count and plot unique values in bands
  for i in range(m-1):
    lbl = "band "+str(i+1)
    band_vals[i][0], band_vals[i][1] = np.unique(X[:,i], return_counts=True)
    plt.plot(band_vals[i][0], band_vals[i][1], label=lbl)
  plt.legend()
  # Specific ID for filename
  if id!=None:
    fname = "assets/"+id+"_band_counts.png"
  else:
    fname = "assets/band_counts.png"
  plt.savefig(fname)
  plt.clf()
  return

# Plot the frequencies of each class as a bar plot
def plot_class_counts(X,id=None):
  # Get classes and class counts from the class/label column
  classes,class_counts = np.unique(X[:,-1], return_counts=True)
  for i in range(len(classes)):
    lbl = "class "+str(classes[i])
    plt.bar(i, class_counts[i], label=lbl)
  plt.ylabel("Frequency")
  # Remove x ticks
  plt.tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False)
  plt.legend()
  if id!=None:
    plt.savefig("assets/"+id+"_class_counts.png")
  else:
    plt.savefig("assets/class_counts.png")
  plt.clf()
  return

# Provide colors to roughly match the original NLCD 2019 color scheme 
def LCC_colorMap():
  cmap = {
    0:"crimson",
    11:"cornflowerblue",
    12:"white",
    21:"rosybrown",
    22:"lightcoral",
    23:"red",
    24:"maroon",
    31:"grey",
    41:"lightgreen",
    42:"forestgreen",
    43:"palegreen",
    51:"goldenrod",
    52:"tan",
    71:"sandybrown",
    72:"yellowgreen",
    73:"olivedrab",
    74:"lightsteelblue",
    81:"gold",
    82:"peru",
    90:"aquamarine",
    95:"lightseagreen",
  }
  return cmap

# Show a predicted land-cover map as an image
def show_as_img(predictions,dimensions,pixel_idxs,labels, show=False, fname=None, save_txt=False):
  print("Plotting Prediction Image")
  print("  Dimensions:",dimensions)
  classes = np.unique(predictions)
  print("  NumClasses:",len(classes))
  print("  Classes   :",classes)
  
  # Get dimensions
  n = int(dimensions[0]*dimensions[1])
  # Storage for image values
  image = np.zeros(n)
  # Assign predictions to image to be displayed by the indexes of the predicted pixels
  for idx in range(len(predictions)):
    image[ int(pixel_idxs[idx]) ] = predictions[idx]
  
  # Save text file of predictions
  if save_txt:
    file = open("assets/pred_values.txt","w")
    for i in range(len(image)):
      file.write(str(image[i])+"\n")
    file.close()

  # Fill in color map with the class colors
  cmap = LCC_colorMap()
  ColMap = ListedColormap([str(cmap[_class]) for _class in classes])
  patches = [mpatches.Patch(color=cmap[i],label=labels[i]) for i in classes]
  
  # Plot the predictions
  plt.subplots_adjust(right=0.69)
  plt.legend(loc='upper center', bbox_to_anchor=(1.25,1.0), handles=patches)
  plt.imshow(image.reshape(dimensions), cmap=ColMap)
  
  # Show or save
  if show:
    plt.show()
  if fname!=None:
    plt.imsave(fname, image.reshape(dimensions))
  return

# Show original labeling of data
def show_NLCD(X,labels,dimensions,fname=None):
  cmap = LCC_colorMap()
  classes = np.unique(X)
  # Fill in color map with the class colors
  ColMap = ListedColormap([str(cmap[_class]) for _class in classes])
  patches = [mpatches.Patch(color=cmap[i],label=labels[i]) for i in classes]
  # Plot and show (and save)
  plt.subplots_adjust(right=0.69)
  plt.legend(loc='upper center', bbox_to_anchor=(1.25,1.0), handles=patches)
  plt.imshow(X.reshape(dimensions), cmap=ColMap)
  plt.show()
  if fname != None:
    plt.imsave(fname, X.reshape(dimensions), cmap=ColMap)