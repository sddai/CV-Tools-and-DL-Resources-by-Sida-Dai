import os
for parent, dirnames, filenames in os.walk(os.getcwd()):
    for filename in filenames:
        os.rename(os.path.join(parent, filename), os.path.join(parent, filename.replace(' ', '')))
