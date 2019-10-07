import torch
from torchvision import transforms, models
from PIL import Image
import os.path
from prep import ImageNetDataset
import getimagenetclasses



def get_imgnet_classes():
    """
    Get labels/classes for ImageNet (AlexNet).
    """
    return eval(open('imagenet1000_clsid_to_human.txt').read())

def prep_pretrained(imgf):
    """
    Process an image so it can be used with
    pre trained models available in PyTorch
    (including AlexNet).

    imgf - a name of the file to process
    """
    # First Task
    p_transform = transforms.Compose([
     transforms.Resize(226),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(imgf).convert('RGB')
    new_img=p_transform(img)
    return new_img

def classify(img_dir, prep1, model, classes):
    """
    For each image in img_dir, first process an image
    using prep function,  use a an model to
    classify it according to classes defined in classes.
    """
    lst = []
    index = 0
    for f in os.listdir(img_dir):
        if index >= 20:
            break
        msg=f
        # First we need to preprocess an image
        # so it will be good fit for our model
        ni=prep1(os.path.join(img_dir,f))
        # Our preparation function will return
        # a tensor with data from an image
        #
        # Tensor is bascily an array.
        # If you want to use it with a model
        # it has to be in the right "format"
        # unqueeze(0) means add this tensor
        # into an extra array
        uni=ni.unsqueeze(0)
        # No we're ready to use our mode.
        # print(uni.size())
        out=model(uni)
        # We need to convert our results from
        # tensor to an array that we can easily examine.
        mout=out.detach().numpy()
        # Creating an array with class indexes,
        # "score" (also called an energy) and name of
        # the classess.
        # The higher the energy for a give class the more
        # probable it is that our image belongs to it.
        aao=[]
        for i, o in enumerate(mout[0]):
            iv='?'
            try:
                iv=classes[i]
            except KeyError:
                pass
            aao.append((i, o, iv))
        # Just sort our array to show most probable classes first.
        aao.sort(key=lambda x: x[1], reverse=True)
        msg+=' Most probable classes: %s' % ','.join([ (aao[ci][2]+'(%f)' % aao[ci][1]) for ci in range(3) ])
        lst.append(aao[0][0])
        print(msg)
        index += 1
    return lst

def comparison(predict, synsetstoindices):
    lst = []
    len_labels = len(predict)
    correct = 0
    directory = os.fsencode("C://Users//ongajong//Documents//DeepLearning//Week4//imagenet_first2500//val")
    index = 0
    for file in os.listdir(directory):
        # print(index)
        if index >= len_labels:
            break
        else:
            index += 1
            filename = os.fsdecode(file)
            if filename.endswith(".xml"): 
                nm = os.fsdecode(os.path.join(directory, file))
                labe, firstname = getimagenetclasses.parseclasslabel(nm,synsetstoindices)
                lst.append(labe)
                continue
            else:
                continue
    print('lst', len(lst))
    for i in range(len_labels):
        if lst[i] == predict[i]:
            correct += 1
    return correct


if __name__ ==  '__main__':
    print('Classification using pretrained AlexNet CNN model:')
    an=models.squeezenet1_1(pretrained=True)
    imgnc=get_imgnet_classes()
    labels = classify("C://Users//ongajong//Documents//DeepLearning//Week4//imagenet_first2500//imagespart", prep_pretrained, an, imgnc)

    filen='C://Users//ongajong//Documents//DeepLearning//Week4//imagenet_first2500//synset_words.txt'
    indicestosynsets,synsetstoindices,synsetstoclassdescr=getimagenetclasses.parsesynsetwords(filen)
    t = comparison(labels, synsetstoindices)
    print(t)
#28/250 for Task 1 with normalization
#148/250 for Task 1 with normalization
