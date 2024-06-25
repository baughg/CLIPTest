from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm
import numpy as np
import socket
import sys

# Here we load the multilingual CLIP model. Note, this model can only encode text.
# If you need embeddings for images, you must load the 'clip-ViT-B-32' model
model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

import numpy as np
import matplotlib.pyplot as plt
  
def plot_images(images, query, n_row=2, n_col=2):
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.set_title(query)
        ax.imshow(img)
    plt.show()


# Lets compute the image embeddings.

#For embedding images, we need the non-multilingual CLIP model
img_model = SentenceTransformer('clip-ViT-B-32')

def send_response(host,port,msg) :
    #host = '192.168.1.63'
    #port = 24069
    print(host + ":"+ str(port))
    
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.settimeout(2)
     
    # connect to remote host
    try :
        sc.connect((host, port))
    except :
        print('Unable to connect')
        return
     
    print('Connected to remote host. Start sending messages')
    print("message: " + msg)
    sc.send(msg)
    #data = sc.recv(4096)
    #print data
    sc.shutdown(socket.SHUT_RDWR)
    sc.close()

HOST = ''   # Symbolic name meaning the local host
PORT = 24070    # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
try:
    s.bind((HOST, PORT))
except socket.error:
    print('Bind failed. Error code: ')
    sys.exit()
print('Socket bind complete')
s.listen(1)
print('Socket now listening')

while 1:
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    data = conn.recv(1024)
    
    if(data == 'quit'):
        conn.send(data)
        conn.close()
        s.close()
        break
        
    if not data:
        break

    conn.sendall(data)
    
    command_array = str(data,'utf-8').split('#')
    img_list = command_array[0]
    img_list_root = command_array[1]
    output_root = command_array[2]

    # opening the file in read mode 
    my_file = open(img_list, "r") 
    
    # reading the file 
    data_l = my_file.read() 
    
    # replacing end of line('/n') with ' ' and 
    # splitting the text it further when '.' is seen. 
    data_into_list = data_l.split("\n") 
    
    # printing the data 
    print(data_into_list) 
    my_file.close() 

    img_names = []
    valid_images = []

    for file_name in data_into_list :
        full_filepath = img_list_root + file_name
        
        if os.path.isfile(full_filepath) :
            img_names.append(full_filepath)
            valid_images.append(file_name)

    

# img_names = list(glob.glob(f'{data_path}*.jpg'))
    print("Images:", len(img_names))
    img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
    img_emb.shape, type(img_emb)

    img_emb_np = img_emb.numpy()
    np.save(output_root + "features.npy",img_emb_np)
    
    my_file = open(output_root + "valid_images.txt", "w") 

    for file_name in valid_images:
        my_file.write(file_name + '\n')
    my_file.close() 