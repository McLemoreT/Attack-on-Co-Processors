#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:24:29 2022

@author: tyler
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import TorchUtils as TorchUtils
#Some demo imports
import  random


import math
class quarry:
    
    def editImage(location, diff, image): #location is location list
        #diff is distance to perturb 
        #image is the image
        for cord in location:
            #Go to the location specified by cord in image, and add diff
            
            image[0][cord[1]][cord[0]] = image[0][cord[1]][cord[0]] + diff 
            
        return image

    
    def binaryString(number):
        return format(number, 'b') #Convert an into to a string of 0's and 1's
    
    def rankFix(image):
        while(image.dim() > 3):
            image = image[0]
        return image
    
    def makeCoordinates(edit_list,image):
        edit_list = edit_list [::-1] #Invert the string edit_list
        itter = 0
        
        arr = np.zeros((edit_list.count('1'),2)) #Create an array is as big as the number of 1's in edit_list
        
        width = image.size(dim=2)# + 1#Subtract 1 because we are counting 0 as the first number
        i = 0
        for index in edit_list:
            if(index == '1'):
                arr[i][0] = itter % width
                arr[i][1] = math.floor(itter / width)
                i = i + 1
            itter = itter + 1
        return arr.astype(int)
            

    def displayImage (image): #Just a quick function to display an image
         plt.figure()
         plt.imshow(image.reshape((image.size(dim=2), image.size(dim=2)))) #shows it
         plt.show()
         plt.close()
         
    def saveImage (image, modifier):
        name = "images/" + str(modifier) + ".png"
        plt.imsave(name, image.reshape((image.size(dim=2), image.size(dim=2))))
    
    def QuarrySave(image, #The image we are testing
                   iterations, #How many perturbed images are we making?
                   starting_number, #What number are we starting at for quarry?
                   model, #What is the model
                   patchedModel, #What is the memtorch model
                   actual_class, #What is the actual classification of image
                   modifier, #Naming modifier for saving the image, typically to identify the perturbed images as a subset of the original
                   ending_number = -1): #Max number defaults to -1 so we can calculate the actual max number
        #Iterations are the number of times this should run
        #Probably shouldset an upper limit when making quarry images
        dims = image.dim();
        if ending_number == -1: #If the max number isn't set, set it to the true max number
            ending_number = pow(2,image.size(dim=dims) * image.size(dim=dims - 1))
        
        for x in range(starting_number, ending_number, math.floor(ending_number/iterations)):
                       #Generate information for new image
                       bin_string = quarry.binaryString(x) 
                       cords = quarry.makeCoordinates(bin_string, image)
                       #Clone and detach old image
                       new_image = image.clone().detach()
                       #Generate new image
                       new_image = quarry.editImage(cords, TorchUtils.getNormParam(image)[2], new_image)
                       quarry.displayImage(new_image)
                       if(TorchUtils.isGoodPlace(model, patchedModel, new_image, actual_class)):
                           #save
                           name = "images/" + modifier + "---" + str(x) + ".png"
                           plt.imsave(name, new_image.reshape((new_image.size(dim=2), new_image.size(dim=2))))
                       else:
                           #Don't save
                           1+1
                       
        plt.close()

         
        
        
        
if __name__ == '__main__': #Basically everything here is just test functions
    test_tensor = torch.load('./FiveTensor.pt')
    
    torch.save(test_tensor, '/home/tyler/Documents/Rowan/Senior_Clinic/ClinicRepo/Attack-on-Co-Processors/FiveTensor.pt')
    # test.displayImage(test_tensor)
    # print(test.binaryConverter(7))
    # print(test.getCoordinates(test.binaryConverter(7), test_tensor))
    #test_tensora = torch.FloatTensor([[[[0,1,1],[0,1,0],[1,0,0]]]])
    test_tensor = quarry.rankFix(test_tensor)
    # small = torch.FloatTensor([[[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,1,0,1]]]])
    # print(small)
    
    # binstring = test.binaryString(17)
    # cords = test.makeCoordinates(binstring, small)
    # oldsmall = small.clone().detach()
    
    # test.editImage(cords, 0.5, small)
    # test.displayImage(small)
    # test.displayImage(oldsmall)
    
    # binstring = test.binaryString(624109300249999999999999999999999944600573)
    # cords = test.makeCoordinates(binstring, test_tensor)
    # test.displayImage(test_tensor)
    # test.editImage(cords, 0.5, test_tensor)
    # test.displayImage(test_tensor)
    
    #quarry.displayImage(test_tensora)
    #quarry.saveImage(test_tensora, 1)
    
    start = time.time()
    number = 1
    current_num = 0
    last_time = time.time()
    while number < 100000:
        if number % 50 == 0:
            print(50/(time.time()-last_time))
            last_time = time.time()
        current_num = current_num  + random.randrange(1,64) 
        #print(current_num)
        binstring = quarry.binaryString( current_num)
        cords = quarry.makeCoordinates(binstring, test_tensor)
        newtest = test_tensor.clone().detach()
        quarry.editImage(cords, 0.5, newtest)
        quarry.displayImage(newtest)
        
       # newtest = test_tensor.clone().detach()
        #quarry.getPerturbedImage(newtest, number)
        
        #name = "images/" + str(number) + ".png"
        #plt.imsave(name, newtest.reshape((newtest.size(dim=2), newtest.size(dim=2))))
        

        
        #test.displayImage(newtest)
        
        #_thread.start_new_thread(test.thread_test, (number, test_tensor))
        #test.thread_test(number, test_tensor)
        
        #Threading test 1
        #_thread.start_new_thread(test.editImage , (test.makeCoordinates(binstring, test_tensor), 0.5, test_tensor.clone().detach()))
        
        #Display image threaded
        #test.displayImage(test.editImage(test.makeCoordinates(binstring, test_tensor), 0.5, test_tensor.clone().detach()))
        
        number = number + 1

        
        
    plt.close()
    end = time.time()
    timer = end - start
    print(timer)
    
    
    # test.displayImage(small)
    # edits = test.binaryConverter(16)
    # cords = test.getCoordinates(edits, small)
    # new_small = test.editImage(cords, 0.5, small)
    # test.displayImage(new_small)
