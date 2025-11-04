import os
import shutil
import urllib.request
import zipfile

def download_and_prepare_tiny_imagenet(data_dir='data'):
    """
    Scarica Tiny ImageNet e prepara la struttura delle cartelle in formato compatibile con torchvision.ImageFolder
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')

    #Download
    if not os.path.exists(zip_path):
        print("Scaricamento Tiny ImageNet...")
        urllib.request.urlretrieve(dataset_url, zip_path)
    else:
        print("File ZIP già presente, skip del download.")

    #Estrazione
    extract_path = os.path.join(data_dir, 'tiny-imagenet-200')
    if not os.path.exists(extract_path):
        print("Estrazione del dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        print("Dataset già estratto.")

    #Fix della validation set
    val_dir = os.path.join(extract_path, 'val')
    ann_file = os.path.join(val_dir, 'val_annotations.txt')
    img_dir = os.path.join(val_dir, 'images')

    print("Riorganizzazione della validation set...")
    with open(ann_file, 'r') as f:
        for line in f:
            fname, cls, *_ = line.split('\t')
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copyfile(os.path.join(img_dir, fname),
                            os.path.join(cls_dir, fname))
    shutil.rmtree(img_dir, ignore_errors=True)
    print("Dataset pronto in:", extract_path)

if __name__ == "__main__":
    download_and_prepare_tiny_imagenet()
