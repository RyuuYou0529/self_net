import torch
import os
import numpy as np
import tifffile


def intermediate_output(deblur_net,device,path,normalize_mode,epoch,iters):
    deblur_net.eval()
    device_cpu = torch.device('cpu')

    output_path = os.path.join(path, 'checkpoint/', 'intermediate_results/')
    input_img_path=os.path.join(path, 'raw_data/', 'lr/')

    if normalize_mode == 'min_max':
        normalize = lambda t:(t-t.min())/(t.max()-t.min())
    elif normalize_mode == 'z_score':
        normalize = lambda t:(t-t.mean())/t.std()
    else:
        normalize = lambda t:t

    index=1 # users define

    input_img = tifffile.imread(os.path.join(input_img_path, f'{index}.tif'))
    input_img = normalize(input_img.astype(np.float32))
    
    input_img = np.expand_dims(input_img, axis=0)
    input_img = np.expand_dims(input_img, axis=0)

    input_tensor = torch.from_numpy(input_img).to(device)

    with torch.no_grad():
        net_output = deblur_net(input_tensor)

    net_output = net_output.squeeze_(0).squeeze_(0).to(device_cpu).numpy()

    net_output = np.asarray(net_output, dtype=np.float32)

    tifffile.imwrite(os.path.join(output_path, f'Self_net_epoch_{epoch+1}_iters_{iters}_{index}.tif'),
                     net_output)

    deblur_net.train()




