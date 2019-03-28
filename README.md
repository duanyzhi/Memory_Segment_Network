# Reference


# Platforms
1. python3.5
2. tensorflow1.3
3. opencv3.3

# Datasets
[Oxford](http://robotcar-dataset.robots.ox.ac.uk/)
[Google Street](http://cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/)

## Google Street  
train: part1~part9
test: part10

test: 220 memory location, 5 frames image for each location
      220 query location, 5 frames image for each location

you should change datasets folder in config.py

# Run
## run with google
python main.py --pattern train --data google
python main.py --pattern test --data google

## run with oxford
python main.py --pattern train --data oxford
python main.py --pattern test --data oxford

## run with uestc
python main.py --pattern train --data uestc
python main.py --pattern train --data uestc



# Results

click to see the video:  
[![Watch the video](https://raw.githubusercontent.com/duanyzhi/Memory_Segment_Network/master/data/Oxford/test/04911_1435938031486259.png)](https://www.youtube.com/watch?v=hKzVXFhiN-Q&feature=youtu.be)