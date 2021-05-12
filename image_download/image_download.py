import requests
from tqdm import tqdm
import os
from torchvision import models, transforms
import os
from PIL import Image
path = './babel_images.txt'

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    ])

def download_image(babelnet_id, urls):
    count = 0
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                image_name = babelnet_id+'_'+str(count)+ '.jpg'
                f = open('/data/private/lvchuancheng/babel_images_full/'+image_name,'wb')
                f.write(r.content)
                f.close()
                input_image = Image.open('/data/private/lvchuancheng/babel_images_full/'+image_name).convert('RGB')
                input_tensor = transform(input_image)
                count += 1
                if count >= 10:
                    break
        except:
            continue
    return

def read_list(fin, **kword):
    line = fin.readline()
    if 'sep' in kword:
            line = line.strip().split('\t')
    else:
            line = line.strip().split()
    try:
            num = eval(line[0])
            line = line[1:]
    except:
            return []
    return line

def check_data():
    images = []
    for root, dirs, files in os.walk('./babel_images'):
        for f in files:
            images.append(f[:-6])
    print(len(images))
    images_dic = {}
    image_num = []
    with open(path, 'r', encoding = 'utf-8') as f:
        while True:
            num_str = f.readline()[:-1]
            if not num_str:
                break
            urls = f.readline()
            url_num = int(urls[0])
            if url_num > 0:
                url_list = urls.split()
                url_num = int(url_list[0])
                urls = url_list[1:]
                image_num.append(url_num)
            else:
                urls = []
            images_dic[num_str] = urls
    f.close()
    count = 0
    num = []
    for k in images_dic.keys():
        if k in images:
            num.append(count)
        count += 1
    print(count)
    return num, image_num

def download_dataset():
    images_dic = {}
    with open(path, 'r', encoding = 'utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            bn = line.split()[0]
            line = f.readline().split()
            if len(line) == 1:
                continue
            num = line[0]
            urls = line[1:]
            images_dic[bn] = urls
    f.close()
    count = 0
    for k in tqdm(images_dic.keys()):
        count += 1
        if count>=5500 and count<7000:
            download_image(k, images_dic[k])

if __name__ == '__main__':
    download_dataset()