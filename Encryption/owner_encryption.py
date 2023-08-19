## laod plain-images and secret keys
import numpy as np
import scipy.io as scio
from encryption_utils import ksa
from encryption_utils import prga
from encryption_utils import yates_shuffle
import tqdm
from encryption_utils import loadImageSet, loadImageFiles
from JPEG.rgbandycbcr import rgb2ycbcr
import cv2
import copy
from JPEG.jdcencColor import jdcencColor
from JPEG.zigzag import zigzag
from JPEG.invzigzag import invzigzag
from JPEG.jacencColor import jacencColor
from JPEG.Quantization import *
from cipherimageRgbGenerate import Gen_cipher_images
import hashlib
import scipy
np.random.seed(2023)
PDK=np.random.randint(0,255,size=32)#predifined keys

def sign(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    elif x<0:
        return -1

def TS1_Transformation(y,key,TS1):
    rindex = int('0b' + key[0:7],2)
    cindex = []
    for i in range(8):
        temp = int('0b' + key[7+7*i:14+7*i],2)
        cindex.append(temp)
    y = y@TS1[:,:,rindex].T
    for i in range(8):
        y[:,i]=TS1[:,:,cindex[i]]@y[:,i]
    return y



def encryption_each_component(image_component, keys,TS1, type, row, col, N, QF,embedding):
    # generate block permutation vector
    block8_number = int((row * col) / (8 * 8))
    data = [i for i in range(0, block8_number)]
    p_blockY = yates_shuffle(data, keys)
    keys = keys[64:]

    allblock8 = np.zeros([8, 8, int(row * col / (8 * 8))])
    allblock8_number = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            t = image_component[m:m + N, n:n + N] - 128
            y = TS1_Transformation(t, keys,TS1)
            keys = keys[63:]
            for i in range(0, N, 8):
                for j in range(0, N, 8):
                    temp = Quantization(y[i:i + 8, j:j + 8], QF, type=type)  # Quanlity
                    allblock8[:, :, allblock8_number] = temp
                    allblock8_number = allblock8_number + 1

    # block permutation
    permuted_blocks = copy.copy(allblock8)
    for i in range(len(p_blockY)):
         permuted_blocks[:, :, i] = allblock8[:, :, p_blockY[i]]
    
    # Huffman coding
    dccof = []
    accof = []
    for i in range(0, allblock8_number):
        temp = copy.copy(permuted_blocks[:, :, i])
        if i == 0:
            dc = temp[0, 0]
            dc_component = jdcencColor(dc, type)
            dccof = np.append(dccof, dc_component)
        else:
            dc = temp[0, 0] - dc
            dc_component = jdcencColor(dc, type)
            dccof = np.append(dccof, dc_component)
            dc = temp[0, 0]
        acseq = []
        aczigzag = zigzag(temp)
        eobi = 0
        for j in range(63, -1, -1):
            if aczigzag[j] != 0:
                eobi = j
                break
        if eobi == 0:
            acseq = np.append(acseq, [999])
        else:
            acseq = np.append(acseq, aczigzag[1: eobi + 1])
            acseq = np.append(acseq, [999])
        #secret key embedding
        for k in range(len(acseq)-1):
            if len(embedding)!=0:
                if np.abs(acseq[k])==1:
                    b=embedding[0]
                    embedding=embedding[1:]
                    acseq[k]=acseq[k]+sign(acseq[k])*int(b)
                else:
                    acseq[k]=acseq[k]+sign(acseq[k])
        t=len(embedding)
        ac_component = jacencColor(acseq, type)
        accof = np.append(accof, ac_component)

    return dccof, accof, embedding


def encryption(img, keyY, keyCb, keyCr, QF, TS1 ,N=8,embedding=None):
    embedding_bitstring=""
    for i in range(len(embedding)):
        bit=bin(embedding[i])[2:-1]
        for _ in range(8-len(bit)):
            bit='0'+bit
        embedding_bitstring=embedding_bitstring+bit

    # N: block size
    # QF: quality factor
    row, col, _ = img.shape
    plainimage = rgb2ycbcr(img)
    plainimage = plainimage.astype(np.float64)
    Y = plainimage[:, :, 0]
    Cb = plainimage[:, :, 1]
    Cr = plainimage[:, :, 2]

    for i in range(0, int(32 * np.ceil(col / 32) - col)):
        Y = np.c_[Y, Y[:, -1]]
        Cb = np.c_[Cb, Cb[:, -1]]
        Cr = np.c_[Cr, Cr[:, -1]]

    for i in range(0, int(32 * np.ceil(row / 32) - row)):
        Y = np.r_[Y, [Y[-1, :]]]
        Cb = np.r_[Cb, [Cb[-1, :]]]
        Cr = np.r_[Cr, [Cr[-1, :]]]

    [row, col] = Y.shape

    Cb = cv2.resize(Cb,
                    (int(col / 2), int(row / 2)),
                    interpolation=cv2.INTER_CUBIC)
    Cr = cv2.resize(Cr,
                    (int(col / 2), int(row / 2)),
                    interpolation=cv2.INTER_CUBIC)

    # Y component
    dccofY, accofY,embedding_bitstring = encryption_each_component(Y, keyY, TS1 , type='Y', row=row, col=col,N=N, QF=QF
                                            ,embedding=embedding_bitstring)
    ## Cb and Cr component
    dccofCb, accofCb, embedding_bitstring = encryption_each_component(Cb, keyCb,TS1, type='Cb', row=int(row / 2), col=int(col / 2), N=N
                                            , QF=QF,embedding=embedding_bitstring)
    dccofCr, accofCr, embedding_bitstring = encryption_each_component(Cr, keyCr,TS1, type='Cr', row=int(row / 2), col=int(col / 2), N=N
                                            , QF=QF,embedding=embedding_bitstring)
    assert len(embedding_bitstring)==0
    accofY = accofY.astype(np.int8)
    dccofY = dccofY.astype(np.int8)
    accofCb = accofCb.astype(np.int8)
    dccofCb = dccofCb.astype(np.int8)
    accofCr = accofCr.astype(np.int8)
    dccofCr = dccofCr.astype(np.int8)
    return accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr


# read plain-images
def read_plain_images():
    plainimage_all = loadImageSet('../data/plainimages/*.jpg')
    # save size information
    img_size = []
    for temp in plainimage_all:
        row, col, _ = temp.shape
        img_size.append((row, col))
    np.save("../data/plainimages.npy", plainimage_all)
    np.save("../data/img_size.npy", img_size)
    return plainimage_all


# generate encryption key and embedding key
# keys are independent from plainimage
# encryption key generation - RC4
hash = hashlib.sha256()


def generate_hash(inp):
    hash.update(bytes(str(inp), encoding='utf-8'))
    res = hash.hexdigest()
    hash_list = []
    for i in range(0, len(res), 2):
        hash_list.append(int(res[i:i + 2], 16))
    return hash_list


def generate_keys(img, control_length=256 * 284):
    # secret keys
    data_lenY = np.ones([1, int(control_length)])

    keyY = generate_hash(img)
    keyCb = generate_hash(img)
    keyCr = generate_hash(img)
    hash_len=len(keyY)
    hashKeys = np.zeros(3*hash_len,dtype=np.int32)
    hashKeys[0:hash_len]=keyY
    hashKeys[hash_len:2*hash_len]=keyCb
    hashKeys[2*hash_len:3*hash_len]=keyCr

    # keys stream
    s = ksa(keyY)
    r = prga(s, data_lenY)
    encryption_keyY = ''
    for i in range(0, len(r)):
        temp1 = str(r[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyY = encryption_keyY + temp2

    data_lenC = np.ones([1, int(control_length // 4)])
    s1 = ksa(keyCb)
    r1 = prga(s1, data_lenC)
    encryption_keyCb = ''
    for i in range(0, len(r1)):
        temp1 = str(r1[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyCb = encryption_keyCb + temp2

    s2 = ksa(keyCr)
    r2 = prga(s2, data_lenC)
    encryption_keyCr = ''
    for i in range(0, len(r2)):
        temp1 = str(r2[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyCr = encryption_keyCr + temp2
    
    return encryption_keyY, encryption_keyCb, encryption_keyCr,hashKeys


if __name__ == '__main__':
    # image encryption
    QF = 90
    plainimage_all = read_plain_images()
    num = len(plainimage_all)  # test images num
    del plainimage_all

    imageFiles = loadImageFiles('../data/plainimages/*.jpg')[:num]
    matfile = scipy.io.loadmat('./transform_stage4change.mat')
    TS1 = datafile = list(matfile.values())[-1]

    for k in tqdm.tqdm([i for i in range(len(imageFiles))]):
        # read plain-image
        img = cv2.imread(imageFiles[k])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encryption_keyY, encryption_keyCb, encryption_keyCr,hashKeys = generate_keys(img)
        t=len(encryption_keyY)

        for i in range(len(hashKeys)):#embedding keys
            index=i%(int(len(hashKeys)/3))
            hashKeys[i]=np.bitwise_xor(PDK[index],hashKeys[i])

        accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr = encryption(img, encryption_keyY, encryption_keyCb,
                                                                        encryption_keyCr, QF, TS1 , N=8,embedding=hashKeys)
        
        np.save(f'../data/JPEGBitStream/YAC/acallY_{k}.npy', accofY)
        np.save(f'../data/JPEGBitStream/YDC/dcallY_{k}.npy', dccofY)
        np.save(f'../data/JPEGBitStream/CbAC/acallCb_{k}.npy', accofCb)
        np.save(f'../data/JPEGBitStream/CbDC/dcallCb_{k}.npy', dccofCb)
        np.save(f'../data/JPEGBitStream/CrAC/acallCr_{k}.npy', accofCr)
        np.save(f'../data/JPEGBitStream/CrDC/dcallCr_{k}.npy', dccofCr)

    # generate cipher-images
    Gen_cipher_images(QF=QF, Image_num=len(imageFiles))
