import multiprocessing.process
import os
import time
import tkinter.filedialog
import tkinter.ttk
import easyocr.recognition
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product as product
import numpy as np
from math import ceil
import cv2
import shutil
import re
from xpinyin import Pinyin
import jieba,random

import tkinter
import multiprocessing
from retinaface import RetinaFace
from collections import Counter
import easyocr
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
#pytesseract cant be packed
#paddleocr cause version error
#keras_ocr dont work on windows
p = Pinyin()
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
class Inception(nn.Module):
  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
  def forward(self, x):
    branch1x1 = self.branch1x1(x)
    branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)
    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)
    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    branch3x3_3 = self.branch3x3_3(branch3x3_2)
    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return torch.cat(outputs, 1)
class CRelu(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x
class FaceBoxes(nn.Module):
  def __init__(self, phase, size, num_classes):
    super(FaceBoxes, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size
    self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
    self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)
    self.inception1 = Inception()
    self.inception2 = Inception()
    self.inception3 = Inception()
    self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
    self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
    self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
    self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
    self.loc, self.conf = self.multibox(self.num_classes)
    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)
    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)
  def forward(self, x):
    detection_sources = list()
    loc = list()
    conf = list()
    x = self.conv1(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = self.conv2(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.inception3(x)
    detection_sources.append(x)
    x = self.conv3_1(x)
    x = self.conv3_2(x)
    detection_sources.append(x)
    x = self.conv4_1(x)
    x = self.conv4_2(x)
    detection_sources.append(x)
    for (x, l, c) in zip(detection_sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)))
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))
    return output
class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1]
                            for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0]
                            for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1]
                            for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0]
                            for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
def mymax(a, b):
    if a >= b:
        return a
    else:
        return b
def mymin(a, b):
    if a >= b:
        return b
    else:
        return a
def cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=int)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = mymax(ix1, x1[j])
            yy1 = mymax(iy1, y1[j])
            xx2 = mymin(ix2, x2[j])
            yy2 = mymin(iy2, y2[j])
            w = mymax(0.0, xx2 - xx1 + 1)
            h = mymax(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep
def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if force_cpu:
        #return cpu_soft_nms(dets, thresh, method = 0)
        return cpu_nms(dets, thresh)
    return cpu_nms(dets, thresh)
# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def load_model(model, pretrained_path, load_to_gpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_gpu:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    else:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
def load_anime_face_model(model_path,gpu):
    # cpu = True
    torch.set_grad_enabled(False)
    # net and model
    # initialize detector
    net = FaceBoxes(phase='test', size=None, num_classes=2)
    net = load_model(net, model_path, gpu)
    net.eval()
    #print('Finished loading model!')
    #print(net)
    return net
def detect_anime_face(old_net,image_path,gpu):
    # start_time = int(time.time())
    cfg = {
        'name': 'FaceBoxes',
        #'min_dim': 1024,
        #'feature_maps': [[32, 32], [16, 16], [8, 8]],
        # 'aspect_ratios': [[1], [1], [1]],
        'min_sizes': [[32, 64, 128], [256], [512]],
        'steps': [32, 64, 128],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True
    }
    confidenceTh = 0.05
    # confidenceTh = 0.1
    nmsTh = 0.3
    keepTopK = 750
    top_k = 5000

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if gpu else "cpu")
    net = old_net.to(device)
    imgOrig = cv2.imread(image_path, cv2.IMREAD_COLOR) if type(image_path)==str else image_path
    img = np.float32(imgOrig)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    loc, conf = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]
    # ignore low scores
    inds = np.where(scores > confidenceTh)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    #keep = py_cpu_nms(dets, args.nms_threshold)
    keep = nms(dets, nmsTh, force_cpu=(not gpu))
    dets = dets[keep, :]
    # keep top-K faster NMS
    dets = dets[:keepTopK, :]
    # print(image_path)
    # print(dets)
    # print("anime_used_time:",int(time.time())-start_time)
    if dets.any():
        return True
    else:
        return False
def load_human_face_model():
    return RetinaFace.build_model()
def detect_human_face(model,image_path):
    # start_time = int(time.time())
    obj = RetinaFace.detect_faces(img_path=image_path,model=model)
    # print("anime_used_time:",int(time.time())-start_time)
    if obj:
        return True
    else:
        return False
def load_ocr_model(gpu):
    return easyocr.Reader(['ch_sim','en'],gpu=gpu)
def detect_ocr(model,image_path):
    # return
    try:
        result = model.readtext(image_path)
    except cv2.error:
        print("Error:",image_path)
        return False
    except ValueError:
        print("ValueError:",image_path)
        return False
    except OSError:
        print("OS_Error:",image_path)
        return False
    except TimeoutError:
        print(image_path,":timeout")
        return False
    all_words = 0
    words_all = ""
    word_en = ""
    word_cn = ""
    for k in result:
        words_all += k[1]+"/"
        if k[1].isalpha(): # totally english
            word_en = k[1].split(" ") #split by space
            if len(word_en) > 4:
                return True
            else:
                all_words += len(word_en)
        word_cn = re.findall(re.compile(u"[\u4e00-\u9fa5]+"), k[1])
        if word_cn:
            for word in word_cn:
                if len(word)>5:
                    cut_words = ("/".join(jieba.cut(word, cut_all=False))).split("/")
                    for cut in cut_words:
                        if len(cut)>=2:
                            all_words += len(cut_words)
                            break
                else:
                    all_words += 1
        if all_words > 5:
            # print("True:"+words_all)
            # print(word_cn)
            # print(word_en)
            return True
    if len(words_all) > 20:
        # print("True:"+words_all)
        # print(word_cn)
        # print(word_en)
        return True
    # print("False:"+words_all)
    # print(word_cn)
    # print(word_en)
    return False
def calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    if total_p % 2 != 0:
        center_p = img_patch[int(img_patch.shape[0] / 2), int(img_patch.shape[1] / 2)]
        mean_p = (np.sum(img_patch) - center_p) / (total_p - 1)
        return (center_p, mean_p)
    else:
        pass
def calcEntropy2dSpeedUp(img, win_w=3, win_h=3):
    height = img.shape[0]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    # 最耗时的步骤，遍历计算二元组
    IJ = []
    for i in range(ext_x, new_width - ext_x):
        for j in range(ext_y, new_height - ext_y):
            patch = final_img[j - ext_y:j + ext_y + 1, i - ext_x:i + ext_x + 1]
            ij = calcIJ(patch)
            IJ.append(ij)

    Fij = Counter(IJ).items()

    # 第二耗时的步骤，计算各二元组出现的概率
    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    H = np.sum(H_tem)
    return H
def detect_entropy(image_path):
    # start_time = int(time.time())
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        H1 = calcEntropy2dSpeedUp(img, 3, 3)
    except AttributeError:
        H1 = 0
    # print("entropy_used_time:",int(time.time())-start_time)
    if H1 < 3:
        return True
    else:
        return False
def detect_picture_size(image_path):
    img = cv2.imread(image_path)
    size = img.shape
    h,w = size[0],size[1]
    if h/w > 3 or w/h > 4 or (h<=600 and w<=600):
        return True
    size_list = ['1920x864','1080x192','1920x144','2400x108','1080x108','1920x886','1920x108','864x192','1216x832','1080x24','2340x108','1024x1024','960x96','1440x192','2532x117','1280x128','1440x108','720x72','600x6','1706x128','4096x189','1707x128','1280x96','800x8','1000x1','3200x144','640x64','960x128','1920x128','3648x2736','720x128','1600x256','1920x861','1600x72','300x3','2436x1124','1280x9','1920x86','2436x1827']
    a = str(h)+"x"+str(w)
    if a in size_list:
        return True
    return False
def file_name_correction(folder_in:str,picture:str):
    '''
    cv2无法识别中文名称的图片,所以对图片进行重命名,新的命名中会将中文替换成拼音
    '''
    if re.findall(re.compile(u"[\u4e00-\u9fa5]+"), picture): #do_chinese
        new = p.get_pinyin(chars=picture,splitter=" ")
        try:
            os.rename(folder_in+picture,folder_in+new)
        except FileExistsError:
            print(picture+"正在被使用中")
            return picture
        return new
    return picture
def move_file(folder_in:str,file_name:str,folder_out:str):
    for i in range(10):
        try:
            a = file_name[-5:]
            new_name = str(random.random()).replace(".","")+a[a.index("."):]
            os.rename(folder_in+"/"+file_name,folder_in+"/"+new_name)
            shutil.move(folder_in+"/"+new_name,folder_out+"/"+new_name)
            # print(new_name)
            return True
        except ValueError:
            return False
        except shutil.Error or FileExistsError:
            continue
    return False
def Task(index:int,picture_list:list,cpu_count:int,detect_dic:list,current_num,setu_number,do_stop):
    # index range from (0 ~ cpu_count-1),but 0 wont appear in this func
    task_round = 0
    total_num = len(picture_list)
    gpu = detect_dic["GPU"]
    folder_in = detect_dic["folder_in"]
    folder_out = detect_dic["folder_out"]
    if detect_dic["Dectect_anime"]:
        anime_model = load_anime_face_model(model_path=detect_dic["anime_model_path"],gpu=gpu)
    if detect_dic["Dectect_human"]:
        human_model = load_human_face_model()
    if detect_dic["Dectect_word"]:
        ocr_model = load_ocr_model(gpu=gpu)
    while (task_round*cpu_count+index < total_num):
        if do_stop.value == -1:
            return
        picture_name = file_name_correction(folder_in+"/",picture_list[task_round*cpu_count+index])
        picture = folder_in+"/"+picture_name
        # print(picture)
        # print("index:",index," ,picture: "+picture)
        read_picture = cv2.imread(picture, cv2.IMREAD_COLOR)
        if read_picture is None:
            if not move_file(folder_in,picture_name,folder_out+"/Others"):
                shutil.move(picture,folder_out+"/Others/"+picture)
            task_round += 1
            current_num.value += 1
            continue
        do_continue = True
        do_faces = False
        if detect_dic["Dectect_picture_size"]:
            if detect_picture_size(image_path=picture):
                do_continue = False
        # picture that dont have specialized sizes goes on
        if do_continue and not do_faces and detect_dic["Dectect_anime"]:
            if detect_anime_face(old_net=anime_model,image_path=picture,gpu=gpu):
                do_faces = True
        # picture that have animefaces goes on
        if do_continue and not do_faces and detect_dic["Dectect_human"]:
            if detect_human_face(model=human_model,image_path=picture):
                do_faces = True
        if (detect_dic["Dectect_anime"] or detect_dic["Dectect_human"]) and not do_faces:
            do_continue = False
        if do_continue and detect_dic["Dectect_word"]:
            if detect_ocr(model=ocr_model,image_path=picture):
                do_continue = False
        if do_continue and detect_dic["Dectect_entropy"]:
            if detect_entropy(image_path=picture):
                do_continue = False
        if do_continue:
            setu_number.value += 1
            move_file(folder_in,picture_name,folder_out+"/Setu")
        else:
            if not move_file(folder_in,picture_name,folder_out+"/Others"):
                shutil.move(picture,folder_out+"/Others/"+picture)
        task_round += 1
        current_num.value += 1
    if do_stop.value != -1:
        do_stop.value += 1
    return

class main_window(object):
    def __init__(self):
        """
        创建整个tkinter内所需的变量
        """
        self.root = tkinter.Tk()
        self.root.resizable(0,0)
        self.root.geometry('600x400')
        self.root.title("色图分离程序")
        self.window_height = 400
        self.window_width = 600
        self.folder_in = ""
        self.folder_out = ""
        self.folder_now = os.getcwd()
        self.total_amount = 0
        self.start_time = time.time()
        self.file_list = []
        self.detect_dic = {"Dectect_anime":False,"Dectect_human":False,"Dectect_word":False,"Dectect_entropy":False,"GPU":False,"folder_in":"","folder_out":"","anime_model_path":"","human_model_path":""}
        self.setu_number = multiprocessing.Value("L",0)
        self.current_num = multiprocessing.Value("L",0)
        self.do_stop = multiprocessing.Value("i",0)
    def StringVars(self):
        self.core_number = tkinter.StringVar(value="1")
        self.text_in = tkinter.StringVar()
        self.text_out = tkinter.StringVar()
        self.text_time = tkinter.StringVar()
        self.text_time2 = tkinter.StringVar()
        self.text_setu = tkinter.StringVar()
    def Labels(self):
        self.lable_in = tkinter.Label(self.root,textvariable=self.text_in)
        self.lable_out = tkinter.Label(self.root,textvariable=self.text_out)
        self.lable_time = tkinter.Label(self.root,textvariable=self.text_time)
        self.lable_time2 = tkinter.Label(self.root,textvariable=self.text_time2)
        self.lable_setu = tkinter.Label(self.root,textvariable=self.text_setu)
    def Frames(self):
        self.frame_checkbutton = tkinter.Frame(self.root)
    def BooleanVars(self):
        """
        
        """
        self.Dectect_human = tkinter.BooleanVar()
        self.Dectect_anime = tkinter.BooleanVar()
        self.Dectect_entropy = tkinter.BooleanVar()
        self.Dectect_word = tkinter.BooleanVar()
        self.Dectect_size = tkinter.BooleanVar()
        self.use_gpu = tkinter.BooleanVar()
    def OptionMenus(self):
        self.combox_core = tkinter.OptionMenu(self.frame_checkbutton,self.core_number,*[i for i in range(1,int(multiprocessing.cpu_count()*2/3))])
    def Progressbars(self):
        self.progressbarTime = tkinter.ttk.Progressbar(self.root,length=400,maximum=400)
    def get_folders(self):
        """
        获取历史上运行该软件时，输入文件夹和输出文件夹的位置
        """
        if os.path.exists(self.folder_now+"/ImageDiliverage") == False:
            os.mkdir(self.folder_now+"/ImageDiliverage")
        folder = self.folder_now+"/ImageDiliverage/last.txt"
        try:
            txt = open(folder,'r',encoding='utf-8')
            self.folder_in = txt.readline().strip("\n")
            self.folder_out = txt.readline().strip("\n")
            txt.close()
        except FileNotFoundError:
            self.folder_in = self.folder_now+"/Input"
            self.folder_out = self.folder_now+"/Output"
        if not self.folder_in:
            self.folder_in = self.folder_now+"/Input"
        if not self.folder_out:
            self.folder_out = self.folder_now+"/Output"
        if os.path.exists(self.folder_in) == False:
            os.mkdir(self.folder_in)
        if os.path.exists(self.folder_out) == False:
            os.mkdir(self.folder_out)
        txt = open(folder,'w',encoding='utf-8')
        txt.write(self.folder_in+"\n"+self.folder_out)
        self.file_list = os.listdir(self.folder_in)
        self.total_amount = len(self.file_list)
        txt.close()
        return
    def choose_in(self):
        """
        选择输入文件夹，并获取文件夹内文件的数量
        """
        self.folder_in = tkinter.filedialog.askdirectory()
        if not self.folder_in:
            self.folder_in = self.folder_now+"/Input"
        self.file_list = os.listdir(self.folder_in)
        self.total_amount = len(self.file_list)
        self.text_in.set("输入文件夹:\n"+self.folder_in)
        self.text_time.set("任务进度:0/"+str(self.total_amount))
        txt = open(self.folder_now+"/ImageDiliverage/last.txt",'w',encoding='utf-8')
        txt.write(self.folder_in+"\n"+self.folder_out)
        txt.close()
        self.root.update()
        return
    def choose_out(self):
        """
        选择输出文件夹
        """
        self.folder_out = tkinter.filedialog.askdirectory()
        if not self.folder_out:
            self.folder_out = self.folder_now+"/Output"
        self.text_out.set("输出文件夹:\n"+self.folder_out)
        txt = open(self.folder_now+"/ImageDiliverage/last.txt",'w',encoding='utf-8')
        txt.write(self.folder_in+"\n"+self.folder_out)
        txt.close()
        self.root.update()
        return
    def Button(self):
        self.button_in = tkinter.Button(self.root,text="输入文件夹浏览",command=self.choose_in)
        self.button_out = tkinter.Button(self.root,text="输出文件夹浏览",command=self.choose_out)
        self.button_start = tkinter.Button(self.root,text="冲刺!",height=3,width=15,command=self.Run_Task)
        self.button_pause = tkinter.Button(self.root,text="!刺冲",height=3,width=15,command=self.Pause)
        self.checkbutton_gpu = tkinter.Checkbutton(self.frame_checkbutton,text="启用GPU",variable=self.use_gpu,onvalue=True,offvalue=False)
        self.checkbutton_face = tkinter.Checkbutton(self.frame_checkbutton,text="启用动漫人脸识别",variable=self.Dectect_anime,onvalue=True,offvalue=False)
        self.checkbutton_humanface = tkinter.Checkbutton(self.frame_checkbutton,text="启用人脸识别",variable=self.Dectect_human,onvalue=True,offvalue=False)
        self.checkbutton_word = tkinter.Checkbutton(self.frame_checkbutton,text="排除含文字图片",variable=self.Dectect_word,onvalue=True,offvalue=False)
        self.checkbutton_entropy = tkinter.Checkbutton(self.frame_checkbutton,text="启用图像熵识别",variable=self.Dectect_entropy,onvalue=True,offvalue=False)
        self.checkbutton_size = tkinter.Checkbutton(self.frame_checkbutton,text="排除指定尺寸的图片",variable=self.Dectect_size,onvalue=True,offvalue=False)
        self.button_list = [
            self.button_in,
            self.button_out,
            self.checkbutton_face,
            self.checkbutton_humanface,
            self.checkbutton_word,
            self.checkbutton_entropy,
            self.checkbutton_gpu,
            self.checkbutton_size
        ]
    def able_all(self):
        """
        启用所有按钮
        """
        for button in self.button_list:
            button["state"] = "normal"
        return
    def disable_all(self):
        """
        禁用所有按钮
        """
        for button in self.button_list:
            button["state"] = "disabled"
        return
    def get_runtime(self):
        """
        根据处理'period'数量的图片所消耗的时间来确定period的大小以及总共所需的时间
        """
        
        time_used = time.time()-self.start_time
        total_time = int(time_used/self.current_num.value*(self.total_amount-self.current_num.value))
        time1 = total_time//3600
        time2 = (total_time-3600*time1)//60
        time3 = total_time-3600*time1-60*time2
        self.text_time2.set("预计剩余时间:%02d"%time1+":"+"%02d"%time2+":"+"%02d"%time3)
    def window_formalize(self):
        """
        
        """
        self.checkbutton_size.pack()
        self.checkbutton_gpu.pack()
        self.checkbutton_face.pack()
        self.checkbutton_humanface.pack()
        self.checkbutton_word.pack()
        self.checkbutton_entropy.pack()
        self.combox_core.pack()
        self.frame_checkbutton.pack(side="left")
        self.lable_in.pack(side="top")
        self.button_in.pack(side="top",pady=10)
        self.lable_out.pack(side="top",pady=0)
        self.button_out.pack(side="top",pady=0)
        self.lable_time.pack(side="top",pady=5)
        self.progressbarTime.pack(side="top",pady=0)
        self.lable_setu.pack(side="top",pady=5)
        self.lable_time2.pack(side="top",pady=0)
        self.button_start.pack(side="bottom")
    def Initialization(self):
        """
        
        """
        self.setu_number.value = 0
        self.current_num.value = 0
        self.do_stop.value = -1
        self.start_time = time.time()
        self.progressbarTime['value'] = 0
        self.get_folders()
        self.text_in.set("输入文件夹:\n"+self.folder_in)
        self.text_out.set("输出文件夹:\n"+self.folder_out)
        self.text_time.set("任务进度:0/"+str(len(os.listdir(self.folder_in))))
        self.text_time2.set("预计剩余时间:00:00:00")
        self.text_setu.set("已分离出"+str(self.current_num.value-self.setu_number.value)+"张非色图照片")
    def Run_Task(self):
        """
        
        """
        self.do_stop.value = 0
        self.process_list = []
        self.button_start.destroy()
        self.button_pause = tkinter.Button(self.root,text="!刺冲",height=3,width=15,command=self.Pause)
        self.button_pause.pack(side="bottom")
        self.disable_all()
        self.start_time = time.time()
        if os.path.exists(self.folder_out+"/Setu") == False:
            os.mkdir(self.folder_out+"/Setu")
        if os.path.exists(self.folder_out+"/Others") == False:
            os.mkdir(self.folder_out+"/Others")

        if not self.core_number.get():
            self.core_number.set("1")
        
        self.detect_dic["Dectect_anime"] = self.Dectect_anime.get()
        self.detect_dic["Dectect_human"] = self.Dectect_human.get()
        self.detect_dic["Dectect_entropy"] = self.Dectect_entropy.get()
        self.detect_dic["Dectect_word"] = self.Dectect_word.get()
        self.detect_dic["Dectect_picture_size"] = self.Dectect_size.get()
        self.detect_dic["anime_model_path"] = self.folder_now+"/model/ssd_anime_face_detect.pth"
        self.detect_dic["human_model_path"] = ""
        self.detect_dic["ocr_model_path"] = self.folder_now+"/model/"
        self.detect_dic["GPU"] = self.use_gpu.get()
        self.detect_dic["folder_in"] = self.folder_in
        self.detect_dic["folder_out"] = self.folder_out

        if not self.file_list:
            self.file_list = os.listdir(self.folder_in)
            self.total_amount = len(self.file_list)
            self.text_in.set("输入文件夹:\n"+self.folder_in)
            self.text_time.set("任务进度:0/"+str(self.total_amount))
            #self.text_setu.set("已分离出"+str(self.current_num.value-self.setu_number.value)+"张非色图照片")

        cpu_count = int(self.core_number.get())
        for i in range(1,cpu_count): # 1 ~ cpu_count-1
            self.process_list.append(multiprocessing.Process(target=Task,args=(i,self.file_list,cpu_count,self.detect_dic,self.current_num,self.setu_number,self.do_stop)))
        for process in self.process_list:
            process.start()
        task_round = 0
        if self.detect_dic["Dectect_anime"]:
            anime_model = load_anime_face_model(model_path=self.detect_dic["anime_model_path"],gpu=self.detect_dic["GPU"])
        if self.detect_dic["Dectect_human"]:
            human_model = load_human_face_model()
        if self.detect_dic["Dectect_word"]:
            ocr_model = load_ocr_model(gpu=self.detect_dic["GPU"])
        while (task_round*cpu_count < self.total_amount) and self.do_stop.value >= 0:
            picture_name = file_name_correction(self.folder_in+"/",self.file_list[task_round*cpu_count])
            picture = self.folder_in+"/"+picture_name
            # print(picture)
            # print("index:0 ,picture: "+picture)
            read_picture = cv2.imread(picture, cv2.IMREAD_COLOR)
            if read_picture is None:
                if not move_file(self.folder_in,picture_name,self.folder_out+"/Others"):
                    shutil.move(picture,self.folder_out+"/Others/"+picture_name)
                task_round += 1
                self.current_num.value += 1
                continue
            do_continue = True
            do_faces = False
            if self.detect_dic["Dectect_picture_size"]:
                if detect_picture_size(image_path=picture):
                    do_continue = False
            # picture that dont have specialized sizes goes on
            if do_continue and not do_faces and self.detect_dic["Dectect_anime"]:
                if detect_anime_face(old_net=anime_model,image_path=picture,gpu=self.detect_dic["GPU"]):
                    do_faces = True
            # picture that have animefaces goes on
            if do_continue and not do_faces and self.detect_dic["Dectect_human"]:
                if detect_human_face(model=human_model,image_path=picture):
                    do_faces = True
            if (self.detect_dic["Dectect_anime"] or self.detect_dic["Dectect_human"]) and not do_faces:
                do_continue = False
            if do_continue and self.detect_dic["Dectect_word"]:
                if detect_ocr(model=ocr_model,image_path=picture):
                    do_continue = False
            if do_continue and self.detect_dic["Dectect_entropy"]:
                if detect_entropy(image_path=picture):
                    do_continue = False
            if do_continue:
                self.setu_number.value += 1
                move_file(self.folder_in,picture_name,self.folder_out+"/Setu")
            else:
                if not move_file(self.folder_in,picture_name,self.folder_out+"/Others"):
                    shutil.move(picture,self.folder_out+"/Others/"+picture)
            self.current_num.value += 1
            task_round += 1
            self.progressbarTime['value'] = int((self.current_num.value/self.total_amount)*400)
            self.text_time.set("任务进度:"+str(self.current_num.value)+"/"+str(self.total_amount))
            self.text_setu.set("已分离出"+str(self.current_num.value-self.setu_number.value)+"张非色图照片")
            self.get_runtime()
            self.root.update()
        if self.do_stop.value != -1:
            self.do_stop.value += 1
        while self.do_stop.value < cpu_count:
            if self.do_stop.value <= 0:
                break
            self.progressbarTime['value'] = int((self.current_num.value/self.total_amount)*400)
            self.text_time.set("任务进度:"+str(self.current_num.value)+"/"+str(self.total_amount))
            self.text_setu.set("已分离出"+str(self.current_num.value-self.setu_number.value)+"张非色图照片")
            self.get_runtime()
            self.root.update()
            time.sleep(1)
        del self.process_list
        self.process_list = []
        print("Finish")
        self.Pause()
    def Pause(self):
        self.do_stop.value = -1
        self.Initialization()
        try:
            self.button_start.destroy()
        except NameError:
            pass
        self.button_pause.destroy()
        self.button_start = tkinter.Button(self.root,text="冲刺!",height=3,width=15,command=self.Run_Task)
        self.button_start.pack(side="bottom")
        self.able_all()
    def Main(self):
        """
        
        """
        self.StringVars()
        self.Labels()
        self.Frames()
        self.BooleanVars()
        self.OptionMenus()
        self.Progressbars()
        self.Button()
        self.Initialization()
        self.window_formalize()
        self.root.mainloop()
        return
if __name__ =='__main__':
    main = main_window()
    main.Main()

