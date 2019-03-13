import random
from shutil import copyfile
 

Age_label_dir = '/home/test.txt'
 

Age_train_label_dir = '/home/_test.txt'
Age_val_label_dir = '/home/_val.txt'

 
random_ins = random.Random(4)

Age_label_files =  open(Age_label_dir, 'r').readlines()

random_ins.shuffle(Age_label_files)

Age_val_labels, Age_train_labels = (Age_label_files[0:int((len(Age_label_files)*0.5))], Age_label_files[int((len(Age_label_files)*0.5)):])

Age_val_label_file =  open(Age_val_label_dir, 'w')
Age_train_label_file =  open(Age_train_label_dir, 'w')


count = 0
l = len(Age_val_labels)
for label in Age_val_labels:
	if count < l - 1:
		Age_val_label_file.write(' '.join(label.strip().split())+ '\n')
	else:
		Age_val_label_file.write(' '.join(label.strip().split()))
	count += 1

count = 0
l = len(Age_train_labels)
for label in Age_train_labels:
	if count < l - 1:
		Age_train_label_file.write(' '.join(label.strip().split())+ '\n')
	else:
		Age_train_label_file.write(' '.join(label.strip().split()))
	count += 1
