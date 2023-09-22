import torch
import cv2
import numpy as np
import Self_net_architecture
import tifffile
import threading


def reslice(img,position,x_res,z_res):
    scale=z_res/x_res
    z,y,x=img.shape
    if position=='xz':
        reslice_img=np.transpose(img,[1,0,2])

        scale_img=np.zeros((y,round(z*scale),x),dtype=np.uint16)
        for i in range(y):
            scale_img[i]=cv2.resize(reslice_img[i],(x,round(z*scale)),interpolation=cv2.INTER_CUBIC)

    else:
        reslice_img=np.transpose(img,[2,0,1])
        scale_img = np.zeros((x, round(z * scale), y), dtype=np.uint16)
        for i in range(x):
            scale_img[i] = cv2.resize(reslice_img[i], (y,round(z * scale)), interpolation=cv2.INTER_CUBIC)

    return scale_img



def output_img(deblur_net,device,min_v,max_v,write_stack,raw_img):
    deblur_net.eval()
    device_cpu = torch.device('cpu')

    batch_size=3

    z_shape=raw_img.shape[0]

    idx = z_shape// batch_size

    res=z_shape-idx*batch_size

    input_img = (raw_img.astype(np.float32) - min_v) / (max_v - min_v)
    input_img[input_img> 1] = 1
    input_img[input_img < 0] = 0
    input_img = np.expand_dims(input_img, axis=1)

    input_tensor = torch.from_numpy(input_img)

    for ii in range(idx):
        with torch.no_grad():
            test_tensor=input_tensor[ii * batch_size:(ii + 1) * batch_size].to(device)
            net_output = deblur_net(test_tensor)
            print('{}/{}'.format((ii + 1) * batch_size, z_shape))
        net_output = net_output.squeeze_(1).to(device_cpu).numpy()
        net_output = net_output * (max_v - min_v) + min_v
        net_output = np.clip(net_output, 0, max_v).astype(np.uint16)

        write_stack[ii * batch_size:(ii + 1) * batch_size] = net_output


    if res!=0:
        test_tensor = input_tensor[idx * batch_size:].to(device)
        with torch.no_grad():
            net_output = deblur_net(test_tensor)
            print('{}/{}'.format(z_shape, z_shape))

        net_output = net_output.squeeze_(1).to(device_cpu).numpy()
        net_output = net_output * (max_v - min_v) + min_v
        net_output = np.clip(net_output, 0, max_v).astype(np.uint16)

        write_stack[idx * batch_size:] = net_output



class myThread (threading.Thread):
    def __init__(self, threadID, name,deblur_net,device,min_v,max_v,write_stack,raw_img):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.deblur_net=deblur_net
        self.device=device
        self.min_v=min_v
        self.max_v=max_v
        self.raw_img=raw_img
        self.write_stack=write_stack

    def run(self):
        print ("start threading：" + self.name)
        output_img(self.deblur_net, self.device, self.min_v,self.max_v,self.write_stack,self.raw_img)
        print ("quit threading：" + self.name)


def upsample_block(raw_img,x_res,z_res,deblur_netA,deblur_netB, min_v, max_v):

    device1=torch.device('cuda:0')
    device2=torch.device('cuda:1')


    xz_img=reslice(raw_img,'xz',x_res,z_res)
    yz_img=reslice(raw_img,'yz',x_res,z_res)

    print(xz_img.shape,yz_img.shape)


    out_xz_img=np.zeros_like(xz_img,dtype=np.uint16)
    out_yz_img=np.zeros_like(yz_img,dtype=np.uint16)

    thread1=myThread(1,'thread:1',deblur_netA,device1,min_v,max_v,out_xz_img,xz_img)
    thread2=myThread(2,'thread:2',deblur_netB,device2,min_v,max_v,out_yz_img,yz_img)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    re_out_xz=np.transpose(out_xz_img,[1,0,2])
    re_out_yz=np.transpose(out_yz_img,[1,2,0])

    fusion_stack=re_out_xz/2+re_out_yz/2
    fusion_stack=np.array(fusion_stack,dtype=np.uint16)


    return fusion_stack

if __name__ == "__main__":

    test_path=r'D:\confocal_Thy1_neuron/'
    model_path=test_path+'checkpoint/saved_models/deblur_net_15_800.pkl'

    device1 = torch.device('cuda:0')
    device2 = torch.device('cuda:1')

    deblur_net_A = Self_net_architecture.define_G(input_nc=1, output_nc=1, ngf=64, netG='deblur_net', device=device1,use_dropout=False,norm='instance')
    deblur_net_A.load_state_dict(torch.load(model_path,map_location={'cuda:1':'cuda:0'}))

    deblur_net_B = Self_net_architecture.define_G(input_nc=1, output_nc=1, ngf=64, netG='deblur_net', device=device2,use_dropout=False,norm='instance')
    deblur_net_B.load_state_dict(torch.load(model_path))

    min_v = 0
    max_v = 4095

    raw_img=tifffile.imread(test_path+'/raw_data/raw_data.tif')

    scale=0.21

    fusion_stack=upsample_block(raw_img,scale,1,deblur_net_A,deblur_net_B,min_v,max_v)

    tifffile.imwrite(test_path+'Self_Net_output.tif',fusion_stack)
