#!/usr/bin/env bash
set -e

# download image data
wget http://www.cs-chan.com/source/ICDAR2017/totaltext.zip

unzip totaltext.zip
chmod -R o-w Images
rm -rf __MACOSX
mv Images/Train/img61.JPG Images/Train/img61.jpg

# download ground truth data
wget http://www.cs-chan.com/source/ICDAR2017/groundtruth_text.zip

unzip groundtruth_text.zip -d gt
chmod -R o-w gt
rm -rf gt/__MACOSX
mv gt/Groundtruth/Polygon/* gt/
rm -rf gt/Groundtruth

mkdir total-text
mv Images gt total-text

mv total-text ../../data
 
