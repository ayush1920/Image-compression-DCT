import numpy as np
import cv2
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc
from scipy import fftpack


with open("image_0.04.xyz", "rb") as f:
    image_data = f.read()

for index, char in enumerate(image_data):
    if char == 0:
        break

metadata = image_data[:index]
metadata = metadata.decode("ascii")
data = eval(metadata)
red, green,blue = data['red'], data['green'], data['blue']
imsize = data['imsize']
thresh = data['thresh']

r_data = image_data[index+1: index+1 + red[1]]
g_data = image_data[index+1 + red[1]: index+1 + red[1] + green[1]]
b_data = image_data[index+1 + red[1] + green[1]: index+1 + red[1] + green[1] + blue[1]]

def huffman_decode(ch_data, symbol_dict):
    binary = ""
    symbol_dict = {symbol_dict[i]: i for i in symbol_dict}
    
    for i in ch_data:
        binary_num   = bin(i)[2:]
        binary_num = "0"*(8 - len(binary_num))+binary_num
        binary +=binary_num
    data = ""
    curr = ""
    for i in binary:
        curr+=i
        x = symbol_dict.get(curr,None)
        if x:
            curr=""
            data += x
    return data
    
r_matrix = huffman_decode(r_data, red[0])
g_matrix = huffman_decode(g_data, green[0])
b_matrix = huffman_decode(b_data, blue[0])

# remove garbage value. Values occuring after matrix
# ended because of 0 padding of byte.
# converts string to matrix 
r_matrix = eval(r_matrix.split("]]]")[0]+"]]]")
g_matrix = eval(g_matrix.split("]]]")[0]+"]]]")
b_matrix = eval(b_matrix.split("]]]")[0]+"]]]")

# regenerate matrix by reversing run length coding
# Doing runlength encoding in zig-zag for 8X8 matrix
def run_len():
    i,j = 0,0
    up = True
    while True:
        yield (i,j)
        if i==j==7:
            break
        if up:
            if i==0 and j!=7:
                j+=1
                up = False
            elif j==7 and i!=7:
                i+=1
                up = False
            elif (j!=7 and i!=0):
                i-=1
                j+=1     
        else:
            if j==0 and i!=7:
                i+=1
                up = True
            elif i==7 and j!=7:
                j+=1
                up = True
            elif (j!=0 and i!=7):
                j-=1
                i+=1
    yield 0


def decompress_block(x): 
    gen = run_len()
    index = 0
    num = x[0]
    max_cnt = x[1]
    cnt = 0
    y = np.zeros((8,8))

    for i in range(8):
        for j in range(8):
            a,b = next(gen)
            y[a][b] = num
            cnt+=1
            if (cnt == max_cnt and ((i,j) != (7,7))):
                cnt=0
                index +=2
                num = x[index]
                max_cnt = x[index + 1]
    return y


def decode_channel(channel):
    fin = []
    for ind1, i in enumerate(channel):
        x = []
        for ind2, j in enumerate(i):
            x.append(decompress_block(j))
        x = np.concatenate(x, axis = 1)
        fin.append(x)
    fin = np.concatenate(fin, axis = 0)
    # trim extra bits
    fin = fin[:imsize[0],:imsize[1]]
    return fin

# Decompressing matrix with reverse run-length coding
r_channel = decode_channel(r_matrix)
g_channel = decode_channel(g_matrix)
b_channel = decode_channel(b_matrix)

def idct2(a):
    return fftpack.idct(fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

# Reconstructing Image
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        r_channel[i:(i+8),j:(j+8)] = idct2( r_channel[i:(i+8),j:(j+8)] )
        g_channel[i:(i+8),j:(j+8)] = idct2( g_channel[i:(i+8),j:(j+8)] )
        b_channel[i:(i+8),j:(j+8)] = idct2( b_channel[i:(i+8),j:(j+8)] )

        
# Post processing to keep the pixel values b/w [0,255]    
x = cv2.merge((np.array(np.clip(b_channel, 0, 255), dtype= np.uint8),
              np.array(np.clip(g_channel, 0, 255), dtype= np.uint8),
              np.array(np.clip(r_channel, 0, 255), dtype= np.uint8)))


##Display the dct of that block
cv2.imshow(f'Compression with threshold {thresh}',x)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'nature_compressed_{thresh}.png', x)


