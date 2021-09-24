import argparse

import os, glob, tqdm
from PIL import Image
BASE_PATH = 'E:/Maria_dev/github/Mariia844/srnet_keras/debug'


# im = Image.open('E:/Mary/data/stego/MiPOD/40/00001.tif')
# im2 = Image.open('E:/Mary/bpm_data/40/00001.bmp')


parser = argparse.ArgumentParser(description='Convert an image folder')

parser.add_argument('-s', '--source', help='Source directory', required=True)
parser.add_argument('-d', '--destination', help='Destination directory', required=True)
parser.add_argument('-c', '--count', help='Images to take', default=-1)
parser.add_argument('-i', '--input-format', help='Input image format',
    choices=['png', 'jpeg', 'bmp', 'gif', 'pgm', 'tif'], required=True)
parser.add_argument('-o', '--output-format', help='Output image format',
    choices=['png', 'jpeg', 'bmp', 'gif', 'tif', 'gif'], required=True)
parser.add_argument('-w', '--size', nargs='+', type=int)

args = parser.parse_args()

if (args.input_format == args.output_format and len(args.size) == 0):
    print('File formats should be different!')
elif (args.source == args.destination):
    print('Source and destination should be different!')
else:
    size = ()
    if len(args.size) > 0:
        size = tuple(args.size)
    images = glob.glob(f"{args.source}\*.{args.input_format}")
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    if (len(images) == 0):
        print('Directory is empty!')
    else:
        pbar = tqdm.tqdm(images)
        pbar.set_description('Converting images') 
        for image in pbar:
            file_name = os.path.split(image)[1]
            file_name_without_extension = os.path.splitext(file_name)[0]
            target_file = os.path.join(args.destination, f"{file_name_without_extension}.{args.output_format}")
            if os.path.exists(target_file):
                continue
            im = Image.open(image)
            if (len(size) == 0):
                im.thumbnail(im.size)   
            else:
                im.thumbnail(size)
            im.save(target_file)




# Image.open(f"{BASE_PATH}/00001.tiff").save(f"{BASE_PATH}/00001.bmp")