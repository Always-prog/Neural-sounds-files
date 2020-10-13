"""This file is for normalize list funciton (reshape list)"""
print("import normalize list/reshape library")
from PIL import Image

class resize_list():
    def __init__(self):
        self.lets = {}
        self.index = 0
        self.lets_reverse = {}#this dict need to get key by value
        self.lst_new = []
    def resize(self, lst: list, resize_to: int):
        """Let know count elements in list"""
        self.lets = {}
        self.lets_reverse = {}#this dict need to get key by value
        self.lst_new = []
        self.index = 0
        for let in range(len(lst)):
            if not lst[let] in self.lets:
                self.lets.update({lst[let]:self.index})
                self.index += 1#update to set new index in new element in list
            else:
                pass

        if len(self.lets) > 255:
            return False

        img_list = Image.new("RGB", (len(lst),1))
        for pixel in range(len(lst)):
            img_list.putpixel((pixel,0),(0,0,self.lets[lst[pixel]]))
        img_list = img_list.resize((resize_to,1))

        [self.lets_reverse.update({i[1]:i[0]}) for i in self.lets.items()]
        for pixel in range(resize_to):
            self.lst_new.append(self.lets_reverse[img_list.getpixel((pixel,0))[2]])
        return self.lst_new
