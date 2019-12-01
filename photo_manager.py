#%%
from datetime import datetime
import os

def get_time():
	current_time = datetime.now().strftime("%Y-%m-%d")
	current_folder = os.path.abspath(os.path.dirname(__file__))+"/"+current_time
	if not os.path.exists(current_folder):os.makedirs(current_folder)
	return current_time, current_folder

# %%
def analysis_folder():
	'''
	return first param is folder_list, second is file_list
	'''
	file_name_lists = os.listdir('.')
	folder_list =[]
	file_list = []
	for file_name_list in file_name_lists:
		if file_name_list.find(".")!=-1 and file_name_list.find(".")!=0 and file_name_list.find(os.path.basename(__file__))==-1:
			file_list.append(file_name_list)
		else:
			folder_list.append(file_name_list)
	return folder_list,file_list
# %%
import time
import shutil
def loop_command():
	while True:
		current_time,current_folder = get_time()
		folder_list,file_list = analysis_folder()
		for file_name in file_list:
			shutil.move(f"{file_name}", f"{current_folder}/{file_name}")
		time.sleep(5)
loop_command()


# %%
