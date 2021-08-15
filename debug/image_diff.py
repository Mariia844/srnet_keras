from PIL import Image, ImageChops
img1 = Image.open("E:/Mary/bmp_data_non_used/50/00002.bmp")
img2 = Image.open("E:/Mary/bmp_data_non_used/cover/00002.bmp")
# finding difference
diff = ImageChops.difference(img1, img2)
  
# showing the difference
diff.show()