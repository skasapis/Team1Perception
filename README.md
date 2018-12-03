# Team1Perception
GitHub: 

to download the repo through terminal -- it will download the repo into a folder called Team1Perception within whatever folder is your current working directory in terminal
```
git clone https://github.com/skasapis/Team1Perception.git
```

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



Matlab Neural Nets:

If want to run the file googlenet_full.m you will need the following toolboxes in matlab:
to access matlab toolboxes: from matlab go to the tab home > add ons > search

install:
```
googlenet
alexnet
deep learning toolbox
Deep Learning Toolbox Importer for TensorFlow-Keras Models
```

If you plan on having the correctly formatted Team1.txt file you must also edit the line
```
fullName=snapshot(63:end); %wrt Marie path
```
to crop off the correct number of characters for your specific path setup. This line is in
the function getPrintName at the bottom of the file googlenet_full.m

Finally, you must run the python file called readFile.py
In terminal:
```
cd <appropriate folder>
python readFile.py
```
To adjust the labels from 1,2,3 to 0,1,2 (haven't yet adjusted the matlab code to adjust by itself)
