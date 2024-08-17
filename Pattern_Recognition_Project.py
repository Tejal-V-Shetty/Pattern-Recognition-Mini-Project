# Pattern recognition program
import cv2
from matplotlib import pyplot
import numpy as np

Images=["Checkerboard_Blue","Circles_Blue","Complex","DisjointedFlowers","Flower1","Flower2","Geometric","Honeycomb","MultiFlower1","MultiFlower2","MultiStripes","Spades","Stripes","Stripes_Red","Trees","Vines"]
Dataset=16
Hist_acc=[]
Point_acc=[]
for i in range(31):
    Hist_acc.append(0.0) #Holds accuracy for all patterns in histogram method
    Point_acc.append(0) #Holds accuracy for all patterns in point finder method

hist_tolerance = 7 #Tolerance for error in histogram

#Get the input image whose pattern is to be examined
ip_path="C:\\Users\\Tejal\\Documents\\MVIP\\Test_patterns\\MultiFlower2_Test.jpg"
ip_image = cv2.imread(ip_path,0)

# Histogram method
def hist_finder():
    bin_size = 128 #Size of sampling numbers for histogram levels

    #Calculate histogram for input image
    ip_hist = cv2.calcHist([ip_image],[0],None,[bin_size],[0,bin_size])
    ip_hist=ip_hist/ip_hist.sum()#Finding normalized histogram

    index=0
    for pattern in Images:
        accuracy = 0.0

        #Calculate histogram for sample images
        image = cv2.imread("C:\\Users\\Tejal\\Documents\\MVIP\\Patterns\\"+Images[index]+".jpg",0)
        hist = cv2.calcHist([image],[0],None,[bin_size],[0,bin_size])

        hist=hist/hist.sum()#Finding normalized histogram   
    
        for i in range(0,bin_size):
            accuracy = accuracy + abs(ip_hist[i][0]-hist[i][0])#Finding total difference across the image
        accuracy = accuracy/bin_size #Finding mean difference
        accuracy = 100-(accuracy*10000)
        print(Images[index]," : ",accuracy)
        Hist_acc[index]=(accuracy)#Append the accuracy for each sample image
        index+=1

    #Finding the number of matches
    matches=0
    for ele in Hist_acc:
        if ele>=(100-hist_tolerance):
            matches+=1

    #Checking if there is more than 1 match
    if matches!=1:
        print("Matches found using histogram = ",matches) #More than 1 match found using histogram, or no matches, so finding points is the next step
        point_finder(matches)#Number of histogram matches is passed as a check
    else:
        print("\nThe matching pattern is : " , Images[Hist_acc.index(max(Hist_acc))]) #Print the name of the pattern with highest accuracy
        image = cv2.imread("C:\\Users\\Tejal\\Documents\\MVIP\\Patterns\\"+Images[Hist_acc.index(max(Hist_acc))]+".jpg",0)
        hist = cv2.calcHist([image],[0],None,[bin_size],[0,bin_size])
        hist=hist/hist.sum()
        pyplot.plot(hist)
        pyplot.plot(ip_hist)
        pyplot.suptitle("Histogram comparison")
        pyplot.show()

#Point matching method    
def point_finder(Hist_matches):
        
    gray_ip = cv2.imread(ip_path,cv2.IMREAD_GRAYSCALE)#Read grayscale image for input

    for ele in Hist_acc:
        if ele>=(100-hist_tolerance) or (Hist_matches==0 and Hist_acc.index(ele)<Dataset): #For all matches with accuracy greater than the tolerance value

            gray = cv2.imread("C:\\Users\\Tejal\\Documents\\MVIP\\Patterns\\"+Images[Hist_acc.index(ele)]+".jpg",cv2.IMREAD_GRAYSCALE)#Read grayscale image for samples

            # Initiate ORB detector
            orb = cv2.ORB_create()
            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(gray,None)
            kp2, des2 = orb.detectAndCompute(gray_ip,None)
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1,des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            print(Images[Hist_acc.index(ele)]," : ",len(matches))
            Point_acc[Hist_acc.index(ele)]=len(matches)
    print("\nThe matching pattern is : " , Images[Point_acc.index(max(Point_acc))])
    #Draw first 10 matches.
    gray = cv2.imread("C:\\Users\\Tejal\\Documents\\MVIP\\Patterns\\"+Images[Point_acc.index(max(Point_acc))]+".jpg",cv2.IMREAD_GRAYSCALE)
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray,None)
    kp2, des2 = orb.detectAndCompute(gray_ip,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(gray,kp1,gray_ip,kp2,matches[:10],None,matchColor=(255,0,0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    pyplot.imshow(img3),pyplot.show()

hist_finder()
