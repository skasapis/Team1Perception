# Team1 -- Perception

# To reproduce our results:

First, install the following matlab toolboxes:
to access matlab toolboxes: from matlab go to the tab home > add ons > search

install:
```
googlenet
alexnet
deep learning toolbox
Deep Learning Toolbox Importer for TensorFlow-Keras Models
```


# Downloading our code: 
To download the repo through terminal 
-- it will download the repo into a folder called ```Team1Perception``` within whatever folder is your current working directory 
```
git clone https://github.com/skasapis/Team1Perception.git
```

# To recreate our best submission results
Run ```googlenet_full.m``` as is. Open the file ```Team1_submission.txt```. This file will contain the file ID plus classifications in the desired format for Kaggle submission.


# To recreate our cropped images
Run ```F_RCNN2.m``` as is. This will create a folder named CroppedPics containing the same image organizational format as /deploy/test/ available on the Kaggle page. 
```
https://www.kaggle.com/c/fall2018-rob535-task1/leaderboard
```
Our best cropped images are contained in the folder CroppedPics2.


# To recreate our attempt at classifying based on a combination of cropped images and full sized images
First, run ```googlenet_full.m``` but uncomment line 82 to save the trained net ```save net4Full```

Next, run ```googlenet_full.m```  but change line 9
```
all_imds=imageDatastore('deploy/trainval','IncludeSubfolders',1,'FileExtensions','.jpg');
```
to the commented out line 8
```
all_imds=imageDatastore('deployCropped2/trainval','IncludeSubfolders',1,'FileExtensions','.jpg');
```
AND in line  79, replace ```net4Full``` with ```net4Crop```
```
[net4Full, info] = trainNetwork(augimdsTrain,lgraph,options);
```
AND in line 92, replace  ```net4Full``` with ```net4Crop```
```
[test_labels,~] = classify(net4Full, augimdstest);
```

Then run ```googlenet_full2.m``` as is. Open the file ```Team1_submission.txt``` . This file will contain the file ID as well as classifications in the desired format for Kaggle submission.


# If you are going to add to material to the repo:

whenever start to work - pull the newest version in the repository: 
```
git pull
```

Whenever finish working and are ready to update the newest version:                          
```
git add .
```

or to only do certain files:
```
git add file1.m file2.m file3.m    
```
to see that you are pushing:
```
git status   
```

create new commit: 
``` 
git commit -m "[the note describing what you changed]"                            
git push origin master
```

to remove file from github:        
``` 
git rm --file1.m
git commit -m "removed file1.m from the repo"
git push origin master
```

to add files to the .gitignore file
```
vim .gitignore
```
    once it opens press "i" to insert text in the file,
    then press esc to stop inserting text,
    ":wq" followed by enter to save and quit



