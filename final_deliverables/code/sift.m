clc;
clear all;
close all;
distorted=imread('F:\final_project_data\image\rear_final.jpg');
color=imread('F:\final_project_data\image\back.jpg');
%RGB to grayscale
original=rgb2gray(color);
%SIFT feauture points extraction
ptsOriginal  = detectSURFFeatures(original);
ptsDistorted = detectSURFFeatures(distorted);
[featuresOriginal,   validPtsOriginal]  = extractFeatures(original,  ptsOriginal);
[featuresDistorted, validPtsDistorted]  = extractFeatures(distorted, ptsDistorted);
indexPairs = matchFeatures(featuresOriginal, featuresDistorted);
matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));
figure;
showMatchedFeatures(original,distorted,matchedOriginal,matchedDistorted);
title('SIFT feature matched points (including outliers)');
[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
    matchedDistorted, matchedOriginal, 'similarity');
%Transformation matrix
Tinv  = tform.invert.T
