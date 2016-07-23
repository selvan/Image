from PIL import Image

img = Image.open('picture.jpg')
img = img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if ( 255 - item[0] < 56 )  or ( 255 - item[1] < 56) or (255 - item[2] < 56 ):
        newData.append((0, 0, 0, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save("picture-2.png", "PNG")
