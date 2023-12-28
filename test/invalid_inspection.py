import os

print('Please enter the folder name you want to inspect:')
folder_name = input()
folder_path = './' + str(folder_name) + '/'


for file in os.listdir(folder_path):
    if file.endswith('.log'):
        with open(folder_path + file) as f:
            for line in f:
                if 'invalid' in line:
                    print(file)
                    break
