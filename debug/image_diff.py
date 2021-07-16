from PIL import Image, ImageChops
img1 = Image.open("C:/mary_study/converted/50/00001.bmp")
img2 = Image.open("C:/mary_study/converted/Cover/00001.bmp")
# finding difference
diff = ImageChops.difference(img1, img2)
  
# showing the difference
diff.show()