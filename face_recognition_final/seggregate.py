#!/usr/bin/env python
# coding: utf-8

# In[45]:


import random 
import shutil
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from argparse import ArgumentParser


# In[43]:


class SeggregateData:

    def __init__(self, base=None, folders=None):
        if all(v is not None for v in [base, folders]): 
            self.__base=base  # base path
            self.__folders=folders # tuple of source and destination folders
            self.__dir_list = shutil.os.listdir(base+folders[0]) # read list of sub directories from source folders

    def getSubDirs(self, base):
        subdirs = shutil.os.listdir(base) # extract directory list
        base_path = [base]*len(subdirs) # repeat base directory path according to dir count
        paths = list( filter(shutil.os.path.isdir, 
                                     map(shutil.os.path.join,base_path,subdirs) # join base_path and subdirs
                                    ) # check whether path is directory
                           ) #  convert to list
        return paths    
    
    def getFiles(self, base):
        files = shutil.os.listdir(base) # extract directory list
        base_path = [base]*len(files) # repeat base directory path according to file count
        paths = list( filter(shutil.os.path.isfile, 
                                     map(shutil.os.path.join,base_path,files)  # join base_path and file paths
                                    ) # check whether path is file
                           )#  convert to list
        return paths   
    
    def checkMake(self, base, ddir):
        if not shutil.os.path.exists(base+ddir): # check whether directory exist or not. if not
            shutil.os.mkdir(base+ddir) # create directory
    
    def getRandomFileNames(self, path, count=(1,10)):
        file_names =[] # empty list to append file names
        if shutil.os.path.isdir(path):   # check whether path is directory or not
                lst = shutil.os.listdir(path) # list files in subdirectories 
                for c in count: # read tuples
                    temp_names=[]  # temporary holder list
                    for i in range(c): # iterate 0 to c-1 number of times
                        a_name= random.choice(lst) # randomly select file from list
                        temp_names.append(a_name) # append selected random name to temporary holder
                        lst.remove(a_name) # remove select random name from list to avoid repeat
                    file_names.append(temp_names) # append selected names to list
        return tuple(file_names) # convert list to tuple
    
    def moveFiles(self, base, sdir, sub_dir, src_s, dest_s=('test/','train/')):
        for d, s_lst in zip(dest_s, src_s): # iterate destination directory and selected file names
            self.checkMake(base, d) # check and create directory
            #if dest_s[0] != d: # if 1st value of tuple is not same as d
            self.checkMake(base+d, sub_dir) # check and create sub directories
            dst_path = base + d + sub_dir+'/' # create destination path
            #else:
                #dst_path = base + d #create destination path
            src_path = base + sdir + sub_dir+'/' # create source path
            for i in s_lst: # iterate 
                shutil.copy(src_path+i, dst_path+i)
    
    def seggregate(self, counts=(1,10)):
        for i in self.__dir_list: # iterate on directory list
            path = self.__base+self.__folders[0]+i # join base path, source and sub directories 
            names = self.getRandomFileNames(path, counts) # get selected file names randomly
            self.moveFiles(self.__base,  # base path
                                    self.__folders[0],  # source folder name
                                    i, # subdirectory name
                                    names, # test and train file names 
                                    self.__folders[1:] # tuple of destination directories
                                  )
   
    def plotImages(self, images_arr, count):
        fig, axes = plt.subplots(1, count, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip( images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
    def augmentTypeParams(self, **type_params):
        return ImageDataGenerator(rescale=1./255, # normalize image
                             height_shift_range=0.15, 
                             width_shift_range=0.15, 
                             horizontal_flip=True,
                             rotation_range=45,
                             brightness_range=[0.3,1.0],
                             zoom_range=[0.3,1.0])
    
    def augmentParams(self, aug_params):
        return self.augmentTypeParams().flow(aug_params['params']['sample'], # image sample to augment
                                             batch_size=aug_params['params']['batch_size'], # batch size to read
                                             shuffle=True, 
                                             save_to_dir=aug_params['params']['save_to_dir'], # path to save augmented images
                                             save_prefix=aug_params['params']['save_prefix'], # prefix to add before file names
                                             save_format=aug_params['params']['save_format'] # image format
                                            )
        
    def augmentFromDirectory(self, ddir, aug_count, **aug_params):
        for dir_path in self.getSubDirs(ddir): # iterate over subdirectories from base directory
            aug_params['params']['save_to_dir']=dir_path
            for file_path in self.getFiles(dir_path): #iterate over files from directory
                img = load_img(file_path) # load the image
                data = img_to_array(img) # convert to numpy array
                aug_params['params']['sample']=expand_dims(data, 0) # expand dimension to one sample
                aug_params['params']['save_prefix']=file_path.split('.')[0].split('/')[-1] # create image data augmentation generator
                datagen_itr = self.augmentParams(aug_params) # prepare iterator
                imgs = [datagen_itr.next()[0] for i in range(aug_count)] # generate samples and plot
                #self.plotImages(imgs, aug_count)
    
    def augmentFromFile(self, file_path, aug_count, **aug_params):
        aug_params['params']['save_to_dir']='/'.join(file_path.split('/')[:-1]) #extract directory from file path
        img = load_img(file_path) # load the image
        data = img_to_array(img) # convert to numpy array
        aug_params['params']['sample']=expand_dims(data, 0) # expand dimension to one sample
        aug_params['params']['save_prefix']=file_path.split('.')[0].split('/')[-1] # create image data augmentation generator
        datagen_itr = self.augmentParams(aug_params) # prepare iterator
        imgs = [datagen_itr.next()[0] for i in range(aug_count)] # generate samples and plot
        #self.plotImages(imgs, aug_count)
        

# In[]:

base='/home/soni/Documents/face_recognition/faces/' #args["base"] 
folders= ('images/', 'training/')  #('images/', 'test/', 'train/') #tuple(args["folders"]) 
obj=SeggregateData(base, folders)
obj.seggregate((5,))

# In[]
augdir = ['/home/soni/Documents/face_recognition/faces/training']
obj1 = SeggregateData()
for ddir in augdir : #args['augdir']
    obj1.augmentFromDirectory(ddir, 5, params={'batch_size':1, 'save_format':'jpeg'})
# In[ ]:


if __name__ == '__main__':
    parser = ArgumentParser(description="parser for various directory paths")
    parser.add_argument("--base_dir", help="base directory path",
                            dest='base')
    parser.add_argument('--folders', help= ' source, test and train directories path',
                            nargs='+', dest='folders')
    parser.add_argument('--aug_dirs', help= ' directories path for augmentation',
                            nargs='+', dest='augdir')
    parser.add_argument('--aug_files', help= ' files path for augmentation',
                            nargs='+', dest='augfiles')
    args = vars(parser.parse_args())
    if all(v is not None for v in [args["base"], args["folders"]]):
        base  =args["base"] 
        folders = tuple(args["folders"]) 
        obj = SeggregateData(base, folders)
        obj.seggregate()
    if args["augdir"] is not None:
        obj1 = SeggregateData()
        for ddir in args['augdir']:
            obj1.augmentFromDirectory(ddir, 5, params={'batch_size':1, 'save_format':'jpeg'})
    if args["augfiles"] is not None:
        obj2 = SeggregateData()
        for filepath in args['augfiles']:
            obj2.augmentFromFile(filepath, 5, params={'batch_size':1, 'save_format':'jpeg'})

