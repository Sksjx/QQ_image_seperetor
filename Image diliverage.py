import tkinter,os,tkinter.ttk,shutil,time,cv2,multiprocessing
from cnocr import CnOcr
from tkinter import filedialog
from PIL import Image
import numpy as np
from collections import Counter
from numba import jit
Image.MAX_IMAGE_PIXELS = 9999999999
ocr = CnOcr()
size_list = [(1920,1080),(1680,1050),(1632,918),(1477,831),(1366,768),(1360,768),(1280,1024),(1280,800),(1280,768),(1280,720),(1130,635),(1024,768),(800,600)]
global setu_number
setu_number = 0
format_list = ["png","PNG","jpg","JPG","bmp","BMP"]
n_cpu = int(multiprocessing.cpu_count())
n_cpu_list = []
@jit(nopython=True)
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
def entropy(picture,folder):
    try:
        img = cv2.imread(folder+"/"+picture, cv2.IMREAD_GRAYSCALE)
        H1 = calcEntropy2dSpeedUp(img, 3, 3)
    except AttributeError:
        H1 = 0
    if H1 < 3:
        shutil.move(folder+"/"+picture,folder+"/Entropy_low")
        global setu_number
        setu_number += 1
    #     q.put(picture+str(H1))
    # else:
    #     q.put(False)
    return
def dectect_face(picture,folder):
    if Dectect_humanface == True:
        return
def seperate(picture,folder_in,folder_out):
    img = Image.open(folder_in + "/" + picture)
    height,width = img.height,img.width
    if img.size in size_list or (width < 600 and height < 700) or (height < 600 and width < 700) or (height < 512) or (width < 512) or (width >= 3*height) or (height >= 3*width):
        img.close()
        return False
    try:
        word_list = ocr.ocr(img)
    except OSError:
        print(picture)
        img.close()
        return False
    word_eng,word_cn = 0,0
    for word in word_list:
        if (len(word['text']) > 2 and word['score'] > 0.68) or (len(word['text']) > 5 and word['score'] > 0.35):
            for word_1 in word['text']:
                if '!' <= word_1 <= 'z':
                    word_eng += 1
                else:
                    word_cn += 1
    if word_eng > 14 or word_cn > 4:
        img.close()
        return False
    img.close()
    shutil.move(folder_in + "/" + picture,folder_out)
    return True
if __name__ == '__main__':
    root = tkinter.Tk()
    var = tkinter.StringVar()
    root.resizable(0,0)
    root.geometry('600x400')
    root.title("色图分离程序")
    window_height,window_width = 400,600
    progressbarOne = tkinter.ttk.Progressbar(root,length=400)
    progressbarOne['value'] = 0
    def get_folders():
        global folder_in,folder_out,folder_now,var
        folder_now = os.getcwd()+"/"
        if os.path.exists(folder_now+"last.txt") == True:
            txt = open(folder_now+"last.txt",'r',encoding='utf-8')
            folder_in = txt.readline()
            folder_in = folder_in.strip("\n")
            folder_out = txt.readline()
            folder_out = folder_out.strip("\n")
        else:
            var.set("1")
            txt = open(folder_now+"last.txt",'w',encoding='utf-8')
            folder_in = folder_now
            folder_out = folder_now+"/Output"
            txt.write(folder_in+"\n"+folder_out+"\n")
        txt.close()
        return
    get_folders()
    global folder_in,folder_out,counting_variable
    counting_variable = 10
    Dectect_humanface,Dectect_face,Dectect_entropy = tkinter.BooleanVar(),tkinter.BooleanVar(),tkinter.BooleanVar()
    text_in,text_out,text_time,text_time2,text_setu = tkinter.StringVar(),tkinter.StringVar(),tkinter.StringVar(),tkinter.StringVar(),tkinter.StringVar()
    text_in.set("输入文件夹:\n"+folder_in)
    text_out.set("输出文件夹:\n"+folder_out)
    text_time.set("任务进度:0/"+str(len(os.listdir(folder_in))))
    text_time2.set("预计剩余时间:00:00:00")
    text_setu.set("已分离出"+str(setu_number)+"张非色图照片")
    lable_in = tkinter.Label(root,textvariable=text_in)
    lable_out = tkinter.Label(root,textvariable=text_out)
    lable_time = tkinter.Label(root,textvariable=text_time)
    lable_time2 = tkinter.Label(root,textvariable=text_time2)
    lable_setu = tkinter.Label(root,textvariable=text_setu)
    frame_checkbutton = tkinter.Frame(root)
    for i in range(n_cpu-4):
        n_cpu_list.append(str(i+1))
    combox_core = tkinter.OptionMenu(frame_checkbutton,var,*n_cpu_list)
    def choose_in():
        global folder_in,folder_out,folder_now
        folder_in = filedialog.askdirectory()
        text_in.set("输入文件夹:\n"+folder_in)
        text_time.set("任务进度:0/"+str(len(os.listdir(folder_in))))
        text_setu.set("已分离出"+str(setu_number)+"张非色图照片")
        txt = open(folder_now + "last.txt",'w',encoding='utf-8')
        txt.write(folder_in+"\n"+folder_out)
        txt.close()
        root.update()
        return
    def choose_out():
        global folder_in,folder_out,folder_now
        folder_out = filedialog.askdirectory()
        text_out.set("输出文件夹:\n"+folder_out)
        text_setu.set("已分离出"+str(setu_number)+"张非色图照片")
        txt = open(folder_now + "last.txt",'w',encoding='utf-8')
        txt.write(folder_in+"\n"+folder_out)
        txt.close()
        root.update()
        return
    def start_work():
        if folder_in == "" or folder_out == "" or folder_in == folder_out or  os.path.exists(folder_in) == False or os.path.exists(folder_out) == False:
            text_setu.set("文件夹路径设置错误,请重新设置")
            root.update()
            return
        disable_all()
        if os.path.exists(folder_out + "/History_PictorySeperate") == False:
            os.mkdir(folder_out + "/History_PictorySeperate")
        global setu_number
        setu_number = 0
        all_pictures = os.listdir(folder_in)
        progressbarOne['maximum'] = len(all_pictures)
        runtime = 0
        last_time = time.time()
        for picture in all_pictures:
            runtime += 1
            text_time.set("任务进度:"+str(runtime)+"/"+str(len(all_pictures)))
            if picture[-3:] in format_list:
                if seperate(picture) == False:
                    setu_number += 1
                    shutil.copy(folder_in + "/" + picture,folder_out + "/History_PictorySeperate")
                    os.remove(folder_in + "/" + picture)
                    text_setu.set("已分离出"+str(setu_number)+"张非色图照片")
            last_time = get_runtime(last_time,runtime,len(all_pictures),10)
            progressbarOne["value"] = runtime
            root.update()
        if Dectect_entropy.get() == True:
            start_entropy()
        able_all()
        return
    def start_entropy():
        if os.path.exists(folder_out + "/Entropy_low") == False:
            os.mkdir(folder_out + "/Entropy_low")
        runtime = 0
        all_pictures = os.listdir(folder_out)
        progressbarOne["value"] = 0
        progressbarOne["maximum"] = len(all_pictures)
        last_time = time.time()
        global setu_number
        while runtime < len(all_pictures):
            core_number = int(var.get())
            # quenelist = []
            if all_pictures[runtime][-3:] in format_list:
                if core_number == 1:
                    entropy(all_pictures[runtime],folder_out)
                    runtime += 1
                    last_time = get_runtime(last_time,runtime,len(all_pictures),10)
                else:
                    tasklist = []
                    if len(all_pictures)-runtime < core_number:
                        core_number = len(all_pictures)-runtime
                    for i in range(core_number):
                        if all_pictures[runtime][-3:] in format_list:
                            # quenelist.append(multiprocessing.Queue())
                            tasklist.append(multiprocessing.Process(target=entropy,args=(all_pictures[runtime],folder_out)))
                            runtime += 1
                    for p in tasklist:
                        p.start()
                    # for q in quenelist:
                    #     if q.get() != False:
                    #         print(q.get())
                    for p in tasklist:
                        p.join()
                    last_time = get_multiruntime(last_time,runtime,len(all_pictures),core_number)
            else:
                runtime += 1
            text_setu.set("已分离出"+str(setu_number)+"张非色图照片")
            text_time.set("任务进度:"+str(runtime)+"/"+str(len(all_pictures)))
            root.update()
            progressbarOne["value"] = runtime
        return
    def get_runtime(start_time,runtime,total,min):
        global counting_variable
        if runtime % counting_variable == 0:
            end_time = time.time()
            if (end_time-start_time) < 1 and counting_variable * 2 < (total//10):
                counting_variable *= 2
            if (end_time-start_time) > 3 and counting_variable > min:
                counting_variable //= 2
                if counting_variable < min:
                    counting_variable = min
            if abs(end_time-start_time) > 1:
                time1 = ((end_time-start_time)*((total-runtime)//counting_variable))//3600
                time2 = (((end_time-start_time)*((total-runtime)//counting_variable))-(time1*3600))//60
                time3 = ((end_time-start_time)*((total-runtime)//counting_variable))-(time1*3600)-(time2*60)
                text_time2.set("预计剩余时间:%02d"%time1+":"+"%02d"%time2+":"+"%02d"%time3)
            return end_time
        else:
            return start_time
    def get_multiruntime(start_time,runtime,total,core_number):
        end_time = time.time()
        time1 = ((end_time-start_time)*((total-runtime)//core_number))//3600
        time2 = (((end_time-start_time)*((total-runtime)//core_number))-(time1*3600))//60
        time3 = ((end_time-start_time)*((total-runtime)//core_number))-(time1*3600)-(time2*60)
        text_time2.set("预计剩余时间:%02d"%time1+":"+"%02d"%time2+":"+"%02d"%time3)
        return end_time

    class all_buttons():
        button_in = tkinter.Button(root,text="输入文件夹浏览",command=choose_in)
        button_out = tkinter.Button(root,text="输出文件夹浏览",command=choose_out)
        button_start = tkinter.Button(root,text="冲刺!",height=3,width=15,command=start_work)
        checkbutton_face = tkinter.Checkbutton(frame_checkbutton,text="启用人像识别",variable=Dectect_face,onvalue=True,offvalue=False)
        checkbutton_humanface = tkinter.Checkbutton(frame_checkbutton,text="排除真人的人脸",variable=Dectect_humanface,onvalue=True,offvalue=False)
        checkbutton_entropy = tkinter.Checkbutton(frame_checkbutton,text="启用图像熵识别",variable=Dectect_entropy,onvalue=True,offvalue=False)
    class formalize():
        all_buttons.checkbutton_entropy.pack()
        all_buttons.checkbutton_face.pack()
        all_buttons.checkbutton_humanface.pack()
        combox_core.pack()
        frame_checkbutton.pack(side="left")
        lable_in.pack(side="top")
        all_buttons.button_in.pack(side="top",pady=10)
        lable_out.pack(side="top",pady=0)
        all_buttons.button_out.pack(side="top",pady=0)
        lable_time.pack(side="top",pady=5)
        progressbarOne.pack(side="top",pady=0)
        lable_setu.pack(side="top",pady=5)
        lable_time2.pack(side="top",pady=0)
        all_buttons.button_start.pack(side="bottom")
    def disable_all():
        all_buttons.button_in["state"] = "disabled"
        all_buttons.button_out["state"] = "disabled"
        all_buttons.button_start["state"] = "disabled"
        all_buttons.checkbutton_face["state"] = "disabled"
        all_buttons.checkbutton_entropy["state"] = "disabled"
        all_buttons.checkbutton_humanface["state"] = "disabled"
        return
    def able_all():
        all_buttons.button_in["state"] = "normal"
        all_buttons.button_out["state"] = "normal"
        all_buttons.button_start["state"] = "normal"
        all_buttons.checkbutton_face["state"] = "normal"
        all_buttons.checkbutton_entropy["state"] = "normal"
        all_buttons.checkbutton_humanface["state"] = "normal"
        return
    formalize()
    root.mainloop()