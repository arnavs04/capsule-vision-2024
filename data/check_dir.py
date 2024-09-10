import os

directory = "data/"

if os.path.exists(directory):
    print("Exists")
else:
    print("Doesnt exist")