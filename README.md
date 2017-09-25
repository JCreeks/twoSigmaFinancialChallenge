"# Useful-Codes" 
#To initialize a git repository
$ git init

#Create a new repository
$ git remote add origin https://

#Check Github repository
$ git remote -v

#Change Github repository
$ git remote set-url origin https://

To update your local copy of the code, run this in the terminal:

git pull origin master

When you have made some changes and want to register all these changes to git, run this in the terminal:

git commit --all -m "XXXXX"

where XXXXX is some message that you want to associate with this update. You can write things like "I fixed a small typo in the file main.py" or "Implemented a new machine-learning strategy".

If you added a new file, say "output.csv", you will need to add it to the git tracking system first with the following command:

git add output.csv

And then you can commit it using the previous command. The message for this could be something like "Added a new file called output.csv".

To upload your changes to the server (github) after you committed them, run this in the terminal:

git push origin master


