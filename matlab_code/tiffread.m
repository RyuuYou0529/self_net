
function Stack = tiffread(FileTif)
    InfoImage = imfinfo(FileTif);
    NumberImages = length(InfoImage);
    TifLink = Tiff(FileTif, 'r');
    for i = 1:NumberImages
        TifLink.setDirectory(i);
        Stack(:,:,i) = TifLink.read();
    end
    TifLink.close();
end

