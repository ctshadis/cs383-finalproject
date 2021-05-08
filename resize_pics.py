from PIL import Image
import os

melanoma_directory = r"C:/Users/cshad/PycharmProjects/melanoma/picscopy/melanoma"
nevus_directory = r"C:/Users/cshad/PycharmProjects/melanoma/picscopy/nevus"
seb_directory = r"C:/Users/cshad/PycharmProjects/melanoma/picscopy/seborrheic_keratosis"
removalcount = 0
resizecount = 0
IMG_SIZE = (100, 100)

melfilenames = os.listdir(melanoma_directory)
for filename in melfilenames:
    if filename.startswith("aug"):
        os.remove(os.path.join(melanoma_directory, filename)) #clean out augmented files
        removalcount = removalcount + 1
        continue
    if filename.endswith(".jpg"):
         im = Image.open(melanoma_directory+"/"+filename)
         width, height = im.size
         if(width != IMG_SIZE[0] or height != IMG_SIZE[1]):
            im = im.resize((100,100))
            resizecount = resizecount+1
            im.save(os.path.join(melanoma_directory, filename))
         continue
    else:
        continue
print("Done resizing melanoma files")



nevfilenames = os.listdir(nevus_directory)
for filename in nevfilenames:
    if filename.startswith("aug"):
        os.remove(os.path.join(nevus_directory, filename))
        removalcount = removalcount + 1
        continue
    if filename.endswith(".jpg"):
         im = Image.open(nevus_directory+"/"+filename)
         width, height = im.size
         if(width != IMG_SIZE[0] or height != IMG_SIZE[1]):
            resizecount = resizecount+1
            im = im.resize((100,100))
            im.save(os.path.join(nevus_directory, filename))
         continue
    else:
        continue

print("Done resizing nevus files")


sebfilenames = os.listdir(seb_directory)
for filename in sebfilenames:
    if filename.startswith("aug"):
        os.remove(os.path.join(seb_directory, filename))
        removalcount = removalcount + 1
        continue
    if filename.endswith(".jpg"):
         im = Image.open(seb_directory+"/"+filename)
         width, height = im.size
         if(width != IMG_SIZE[0] or height != IMG_SIZE[1]):
            resizecount = resizecount+1
            im = im.resize((100,100))
            im.save(os.path.join(seb_directory, filename))
         continue
    else:
        continue
print("Done resizing seborrheic keratosis files")
print('Removed ', removalcount, ' files')
print('Resized ', resizecount, ' files')