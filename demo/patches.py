from patchify import patchify
import numpy as np 
import cv2
import os
from symmetry import *
from metrics import *
import torchvision.transforms as transforms
import random
import re
transform = transforms.ToPILImage()

def Text_generation(img,infer_model):

    pred=infer_model.run_on_image(img)
    return pred["predictions"][0]["texts"]


def joinimages(image_list,patch_height,patch_width):
    """
    input args: takes list of image which we will join to make one single image
    returns image
    """
    #visualize it in 3 rows and 4 columns
    complete_image = np.zeros((3* patch_height, 4* patch_width, 3), dtype=np.uint8)
    #print(complete_image.shape)
    for i, patch in enumerate(image_list[0]):
        row = i // 4
        col = i % 4
        complete_image[row * patch_height: (row + 1) * patch_height, col * patch_width: (col + 1) * patch_width] = patch
    return complete_image


def Datapatches(path,ndim=2,st_dim=2):
    """
    inputs attributes: path image path, ndim is in how many dim you want split the original image,st_dim means wether you want over lap or not

    """
    sat_dataset=[]

    image=cv2.imread(path)#read _image
    #print(image.shape[0])
    x_patch_size=(image.shape[0]//ndim)
    y_patch_size=(image.shape[1]//ndim)
    #print(image.shape)
    size_x=(image.shape[0]//ndim)*x_patch_size
    size_y=(image.shape[1]//ndim)*y_patch_size

    image = image[:size_x, :size_y]

    patches_img = patchify(image, (x_patch_size, y_patch_size, 3), step=x_patch_size//st_dim)

    #print(patches_img.shape)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i,j,:,:]
            single_patch_img = single_patch_img[0]
            sat_dataset.append(single_patch_img)
    return sat_dataset


def data_triple_loss_pair(data):
    """
    input: Input to the function is a list containing patches of a single images
    function: The purpose of this function is to split each patch into an anchor image and positive image and find a negative image
            from another patch ,
            from the single patch we split the patch into anchor and positive left side of patch image is anchor and right side is the positive
            now to get the negative we get the next patch (or previous ) and fetch the left half of it
            now we append the [anchor,positive,negative] and return the list after performing on every patches
    Outputs: list of [anchor,positive,negative]
    """

    fs=[]
    i=0
    while i<len(data)-1:

        image=data[i]
        x=image.shape[1]
        anchor=image[:,:x//2]
        positive=image[:,x//2:]
        #lets fetch negative one
        false_image=data[i+1]
        negative=false_image[:,:x//2]
        #print(positive.shape,negative.shape,anchor.shape)
        fs.append([anchor,positive,negative])
        i+=1
    if i==len(data)-1:
        image=data[i]
        x=image.shape[1]
        anchor=image[:,:x//2]
        positive=image[:,x//2:]
        #lets fetch negative one
        false_image=data[i-1]
        negative=false_image[:,:x//2]
        #print(positive.shape,negative.shape,anchor.shape)
        fs.append([anchor,positive,negative])

    return fs

def test(anchor,positive,negative,k,infer,model):

    """
    input args:anchor image (image that is comared with the other two images positive and negative)
                positive image (this image is compared with anchor and the high similarity score one is returned)
                negative image (this image i compared with anchor )
                infer is  the text sotting model that we are using
    returns: if the similarity score(cosine similarity) score is more for the negative for the negative image return a black patch
             else join the anchor and the positive image
    """
    cos_positive = []
    cos_negative = []

    model.eval()
    anchor,positive,negative=torch.unsqueeze(anchor,0), torch.unsqueeze(positive,0),torch.unsqueeze(negative,0)
    #print(anchor.shape)
    #print(positive.shape)
    #print(negative.shape)
    anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
    anchor_out, positive_out, negative_out = model(anchor,positive,negative)

    #print(anchor_out[0].shape)

    cp=cosine_similarity(anchor_out[0], positive_out[0])
    cn=cosine_similarity(anchor_out[0],negative_out[0])
    cos_positive.append(cp)
    cos_negative.append(cn)
    joined_image = torch.cat((positive[0],anchor[0],), dim=2)
    joined_image2 = torch.cat((anchor[0],positive[0]), dim=2)
    anc=transform(anchor[0].cpu())
    pst=transform(positive[0].cpu())
    neg=transform(negative[0].cpu() )
    join1=transform(joined_image.cpu())
    join2=transform(joined_image2.cpu())

    #print(anchor[0].cpu().permute(1, 2, 0).shape)
    if cp>cn:
        join1= np.array(join1)
        join2= np.array(join2)
        #print(join1.shape)
        tex1=Text_generation(join1,infer)
        tex2=Text_generation(join2,infer)
        #a,b=Text_detection_generation(join1,ocr)
        #c,d=Text_detection_generation(join1,ocr)
        #if k!=0:

            #plot_images2("/content/drive/MyDrive/ReID/Samples/Vehicle_Output/Symmetric",random.randint(1, 100),joined_image,joined_image2)
        return tex1,tex2
        #return a,b,c,d
    else:
        return "",""