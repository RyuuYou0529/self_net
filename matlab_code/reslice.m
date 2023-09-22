function reslice_img=reslice(stack,orientation,x_res,z_res)    
    scale=z_res/x_res;
    
    [y,x,z]=size(stack);
    
    if orientation=='xz'
        reslice_stack=permute(stack,[3,2,1]);
        reslice_img=uint16(zeros(round(z*scale),x,y));
        
        for k=1:y
            im_k=squeeze(reslice_stack(:,:,k));
            im_k=imresize(im_k,[round(z*scale),x],'bicubic');
            reslice_img(:,:,k)=im_k;
        end       
        
    elseif orientation=='yz'
        reslice_stack=permute(stack,[3,1,2]);
        reslice_img=uint16(zeros(round(z*scale),y,x));
        
        for k=1:x
            im_k=squeeze(reslice_stack(:,:,k));
            im_k=imresize(im_k,[round(z*scale),y],'bicubic');
            reslice_img(:,:,k)=im_k;
        end       
    else
        disp('Parameter input error');
    end
end


