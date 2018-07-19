import os

'''
if tmp is not exist,create it.
then remove all files in writeBoard/tmp 
'''
def clearDir():
    print(os.getcwd())
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")
    for file in os.listdir("tmp"):
        filePath = os.path.join("tmp",file)
        os.remove(filePath)