from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askdirectory,askopenfilename
from tkinter import messagebox
import sys
from easydict import EasyDict as edict
from tkinter.scrolledtext import ScrolledText
path = '/home/wangyang/桌面/interface/SAR'
sys.path.append(path)
import CNN_Chen_Eval
###define global parameter
aa = 1000
model_config_1 = edict()
model_config_1.image_model = '/home/wangyang/桌面/interface/SAR/model_information/image_model1.jpg'
model_config_1.image_loss =''
model_config_1.image_lr = ''
model_config_1.model_checkpoint_path = '/home/wangyang/下载/SAR/model2/model.ckpt'

model_config_2 = edict()
model_config_2.image_model = '/home/wangyang/桌面/interface/SAR/model_information/image_model2.jpg'
model_config_2.image_loss =''
model_config_2.image_lr = ''
model_config_2.model_checkpoint_path = '/home/wangyang/下载/SAR/model2/model.ckpt'

model_config_3 = edict()
model_config_3.image_model = '/home/wangyang/桌面/interface/SAR/model_information/image_model1.jpg'
model_config_3.image_loss =''
model_config_3.image_lr = ''
model_config_3.model_checkpoint_path = '/home/wangyang/下载/SAR/model2/model.ckpt'

model_select = model_config_1
tk_model_select= 1

path_single = "/home/wangyang/桌面/interface/SAR/test/Image_88/5/Center_5_HB14932.JPG"
#path_model = '/home/wangyang/下载/SAR/model/model.ckpt'
#model_name="model"
root=Tk()
root.geometry("1040x600")
root.title("test system")
root.resizable(width=True,height=True)
frame1=Frame(root,width=1040,height=600)
frame2=Frame(root,width=1040,height=600)
frame3=Frame(root,width=1040,height=600)
frame4=Frame(root,width=1040,height=600)
frame1.pack()

def to_frame1():
	frame3.pack_forget()
	frame2.pack_forget()
	frame4.pack_forget()
	frame1.pack()
def to_frame2():
	frame3.pack_forget()
	frame1.pack_forget()
	frame4.pack_forget()
	frame2.pack()
def to_frame3():
	frame1.pack_forget()
	frame2.pack_forget()
	frame4.pack_forget()
	frame3.pack()
def to_frame4():
	frame1.pack_forget()
	frame2.pack_forget()
	frame3.pack_forget()
	frame4.pack()
def back_to_frame1():
	frame2.pack_forget()
	frame3.pack_forget()
	frame4.pack_forget()
	frame1.pack()

def select_mode(m):
	global model_select
	global tk_model_select
	if m==1:
		model_select = model_config_1
		image_model_select = Image.open(model_select.image_model)
		print(model_select.image_model)
		tk_model_select = ImageTk.PhotoImage(image_model_select)

	if m==2:
		model_select = model_config_2
		image_model_select = Image.open(model_select.image_model)
		print(model_select.image_model)
		tk_model_select = ImageTk.PhotoImage(image_model_select)


	if m==3:
		model_select = model_config_3
		image_model_select = Image.open(model_select.image_model)
		print(model_select.image_model)
		tk_model_select = ImageTk.PhotoImage(image_model_select)

def select_model_1():
	select_mode(1)

def select_model_2():
	select_mode(2)

def select_model_3():
	select_mode(3)

def change_image():
	l.configure(image=tk_model_select)

def show_mode2_image():
	label_image_origin.configure(image=tk_image_change)

def run_1():
	CNN_Chen_Eval.evaluation_single(model_select.model_checkpoint_path,path_single)
	image_change = Image.open(path_single)
	global tk_image_change
	tk_image_change = ImageTk.PhotoImage(image_change)
	show_mode2_image()
	global text_image_label
	global text_image_label_output
	path_split = path_single.split('/')
	text_image_label.set(path_split[-2])
	text_image_label_output.set(CNN_Chen_Eval.index_predict)
	text2 = Text(frame2,width=15,height=8)

	if text_image_label.get()==text_image_label_output.get():
		text2.insert(0.0,'检测正确\n请继续使用')
		text2.place(x=700,y=60)

	if text_image_label.get() != text_image_label_output.get():
		text2.delete(1.0,END)
		text2.insert(0.0,'检测错误')
		text2.place(x=700,y=60)
		messagebox.showwarning('警告','检测错误,请选择其他模型')


def run_2():
	CNN_Chen_Eval.evaluation(model_select.model_checkpoint_path,path_record)
	result_array = CNN_Chen_Eval.final_results
	result_00.set(int(result_array[0][0])), result_01.set(int(result_array[0][1])), result_02.set(int(result_array[0][2])), result_03.set(int(result_array[0][3])), result_04.set(int(result_array[0][4])), result_05.set(int(result_array[0][5])), result_06.set(int(result_array[0][6])), result_07.set(int(result_array[0][7])), result_08.set(int(result_array[0][8])), result_09.set(int(result_array[0][9]))
	result_10.set(int(result_array[1][0])), result_11.set(int(result_array[1][1])), result_12.set(int(result_array[1][2])), result_13.set(int(result_array[1][3])), result_14.set(int(result_array[1][4])), result_15.set(int(result_array[1][5])), result_16.set(int(result_array[1][6])), result_17.set(int(result_array[1][7])), result_18.set(int(result_array[1][8])), result_19.set(int(result_array[1][9]))
	result_20.set(int(result_array[2][0])), result_21.set(int(result_array[2][1])), result_22.set(int(result_array[2][2])), result_23.set(int(result_array[2][3])), result_24.set(int(result_array[2][4])), result_25.set(int(result_array[2][5])), result_26.set(int(result_array[2][6])), result_27.set(int(result_array[2][7])), result_28.set(int(result_array[2][8])), result_29.set(int(result_array[2][9]))
	result_30.set(int(result_array[3][0])), result_31.set(int(result_array[3][1])), result_32.set(int(result_array[3][2])), result_33.set(int(result_array[3][3])), result_34.set(int(result_array[3][4])), result_35.set(int(result_array[3][5])), result_36.set(int(result_array[3][6])), result_37.set(int(result_array[3][7])), result_38.set(int(result_array[3][8])), result_39.set(int(result_array[3][9]))
	result_40.set(int(result_array[4][0])), result_41.set(int(result_array[4][1])), result_42.set(int(result_array[4][2])), result_43.set(int(result_array[4][3])), result_44.set(int(result_array[4][4])), result_45.set(int(result_array[4][5])), result_46.set(int(result_array[4][6])), result_47.set(int(result_array[4][7])), result_48.set(int(result_array[4][8])), result_49.set(int(result_array[4][9]))
	result_50.set(int(result_array[5][0])), result_51.set(int(result_array[5][1])), result_52.set(int(result_array[5][2])), result_53.set(int(result_array[5][3])), result_54.set(int(result_array[5][4])), result_55.set(int(result_array[5][5])), result_56.set(int(result_array[5][6])), result_57.set(int(result_array[5][7])), result_58.set(int(result_array[5][8])), result_59.set(int(result_array[5][9]))
	result_60.set(int(result_array[6][0])), result_61.set(int(result_array[6][1])), result_62.set(int(result_array[6][2])), result_63.set(int(result_array[6][3])), result_64.set(int(result_array[6][4])), result_65.set(int(result_array[6][5])), result_66.set(int(result_array[6][6])), result_67.set(int(result_array[6][7])), result_68.set(int(result_array[6][8])), result_69.set(int(result_array[6][9]))
	result_70.set(int(result_array[7][0])), result_71.set(int(result_array[7][1])), result_72.set(int(result_array[7][2])), result_73.set(int(result_array[7][3])), result_74.set(int(result_array[7][4])), result_75.set(int(result_array[7][5])), result_76.set(int(result_array[7][6])), result_77.set(int(result_array[7][7])), result_78.set(int(result_array[7][8])), result_79.set(int(result_array[7][9]))
	result_80.set(int(result_array[8][0])), result_81.set(int(result_array[8][1])), result_82.set(int(result_array[8][2])), result_83.set(int(result_array[8][3])), result_84.set(int(result_array[8][4])), result_85.set(int(result_array[8][5])), result_86.set(int(result_array[8][6])), result_87.set(int(result_array[8][7])), result_88.set(int(result_array[8][8])), result_89.set(int(result_array[8][9]))
	result_90.set(int(result_array[9][0])), result_91.set(int(result_array[9][1])), result_92.set(int(result_array[9][2])), result_93.set(int(result_array[9][3])), result_94.set(int(result_array[9][4])), result_95.set(int(result_array[9][5])), result_96.set(int(result_array[9][6])), result_97.set(int(result_array[9][7])), result_98.set(int(result_array[9][8])), result_99.set(int(result_array[9][9]))
def change_feature_map():
	label_image_layer1_1.configure(image=image_layer1_1_tk_84)
	label_image_layer1_2.configure(image=image_layer1_2_tk_84)
	label_image_layer2_1.configure(image=image_layer2_1_tk_84)
	label_image_layer2_2.configure(image=image_layer2_2_tk_84)
	label_image_layer3_1.configure(image=image_layer3_1_tk_84)
	label_image_layer3_2.configure(image=image_layer3_2_tk_84)
	label_image_layer4_1.configure(image=image_layer4_1_tk_84)
	label_image_layer4_2.configure(image=image_layer4_2_tk_84)
	label_image_layer5_1.configure(image=image_layer5_1_tk_84)
	label_image_layer5_2.configure(image=image_layer5_2_tk_84)
def show_feature_map():
	global image_layer1_1_tk_84
	global image_layer1_2_tk_84
	global image_layer2_1_tk_84
	global image_layer2_2_tk_84
	global image_layer3_1_tk_84
	global image_layer3_2_tk_84
	global image_layer4_1_tk_84
	global image_layer4_2_tk_84
	global image_layer5_1_tk_84
	global image_layer5_2_tk_84
	image_layer1_1_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer1_1.jpg')
	image_layer1_1_tk_84 = ImageTk.PhotoImage(image_layer1_1_84)

	image_layer1_2_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer1_2.jpg')
	image_layer1_2_tk_84 = ImageTk.PhotoImage(image_layer1_2_84)

	image_layer2_1_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer2_1_84.jpg')
	image_layer2_1_tk_84 = ImageTk.PhotoImage(image_layer2_1_84)

	image_layer2_2_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer2_2_84.jpg')
	image_layer2_2_tk_84 = ImageTk.PhotoImage(image_layer2_2_84)

	image_layer3_1_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer3_1_84.jpg')
	image_layer3_1_tk_84 = ImageTk.PhotoImage(image_layer3_1_84)

	image_layer3_2_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer3_2_84.jpg')
	image_layer3_2_tk_84 = ImageTk.PhotoImage(image_layer3_2_84)

	image_layer4_1_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer4_1_84.jpg')
	image_layer4_1_tk_84 = ImageTk.PhotoImage(image_layer4_1_84)

	image_layer4_2_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer4_2_84.jpg')
	image_layer4_2_tk_84 = ImageTk.PhotoImage(image_layer4_2_84)

	image_layer5_1_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer5_1_84.jpg')
	image_layer5_1_tk_84 = ImageTk.PhotoImage(image_layer5_1_84)

	image_layer5_2_84 = Image.open('/home/wangyang/桌面/interface/feature_map/layer5_2_84.jpg')
	image_layer5_2_tk_84 = ImageTk.PhotoImage(image_layer5_2_84)
	change_feature_map()
def exit():
	root.destroy()
#### define frame1###
Button(frame1,text="button1").pack()
Button(frame1,text="button2",command=exit).pack()
image_background = Image.open('/home/wangyang/桌面/interface/background_resize.jpg')
tk_background = ImageTk.PhotoImage(image_background)
background = Label(frame1,image=tk_background)
background.pack()
#### define frame3###
image_model_1 = Image.open(model_config_1.image_model)
tk_model = ImageTk.PhotoImage(image_model_1)
l = Label(frame3,image=tk_model)
l.place(x=20,y=10)
button_refresh = Button(frame3,text="update",command=change_image)
button_refresh.place(x=800,y=100)
#### define frame2###
path = StringVar()

def selectPath():
	path_ = askopenfilename()
	path.set(path_)
	global path_single
	path_single=path_

	#print(path_single)

Label(frame2,text = "目标路径:",font=("Arial",10)).place(x=350,y=12)
entry1 = Entry(frame2, textvariable = path,font=("Arial",10)).place(x=430,y=12)
Button(frame2, text = "路径选择", command = selectPath,font=("Arial",10)).place(x=580,y=10)
text1 = Text(frame2,width=15,height=18)
#text1.insert(0.0,'成功')
#text1.place(x=700,y=60)
Button(frame2,text='返回',command=back_to_frame1).place(x=900,y=550)
Button(frame2, text = "运行", command = run_1,font=("Arial",10)).place(x=670,y=10)
Button(frame2, text = "显示特征图", command = show_feature_map,font=("Arial",10)).place(x=300,y=200)
image_origin = Image.open(path_single)
tk_image_origin = ImageTk.PhotoImage(image_origin)
label_image_origin = Label(frame2,image=tk_image_origin)
label_image_origin.place(x=300,y=60)

image_layer1_1 = Image.open('/home/wangyang/桌面/interface/feature_map/layer1_1.jpg')
image_layer1_1_tk = ImageTk.PhotoImage(image_layer1_1)
label_image_layer1_1 = Label(frame2,image=image_layer1_1_tk)
label_image_layer1_1.place(x=100,y=250)
print("label_image_layer1_1")
image_layer1_2 = Image.open('/home/wangyang/桌面/interface/feature_map/layer1_2.jpg')

image_layer1_2_tk = ImageTk.PhotoImage(image_layer1_2)
label_image_layer1_2 = Label(frame2,image=image_layer1_2_tk)
label_image_layer1_2.place(x=100,y=350)

image_layer2_1 = Image.open('/home/wangyang/桌面/interface/feature_map/layer2_1_84.jpg')
image_layer2_1_tk = ImageTk.PhotoImage(image_layer2_1)
label_image_layer2_1 = Label(frame2,image=image_layer2_1_tk)
label_image_layer2_1.place(x=200,y=250)

image_layer2_2 = Image.open('/home/wangyang/桌面/interface/feature_map/layer2_2_84.jpg')
image_layer2_2_tk = ImageTk.PhotoImage(image_layer2_2)
label_image_layer2_2 = Label(frame2,image=image_layer2_2_tk)
label_image_layer2_2.place(x=200,y=350)

image_layer3_1 = Image.open('/home/wangyang/桌面/interface/feature_map/layer3_1_84.jpg')
image_layer3_1_tk = ImageTk.PhotoImage(image_layer3_1)
label_image_layer3_1 = Label(frame2,image=image_layer3_1_tk)
label_image_layer3_1.place(x=300,y=250)

image_layer3_2 = Image.open('/home/wangyang/桌面/interface/feature_map/layer3_2_84.jpg')
image_layer3_2_tk = ImageTk.PhotoImage(image_layer3_2)
label_image_layer3_2 = Label(frame2,image=image_layer3_2_tk)
label_image_layer3_2.place(x=300,y=350)

image_layer4_1 = Image.open('/home/wangyang/桌面/interface/feature_map/layer4_1_84.jpg')
image_layer4_1_tk = ImageTk.PhotoImage(image_layer4_1)
label_image_layer4_1 = Label(frame2,image=image_layer4_1_tk)
label_image_layer4_1.place(x=400,y=250)

image_layer4_2 = Image.open('/home/wangyang/桌面/interface/feature_map/layer4_2_84.jpg')
image_layer4_2_tk = ImageTk.PhotoImage(image_layer4_2)
label_image_layer4_2 = Label(frame2,image=image_layer4_2_tk)
label_image_layer4_2.place(x=400,y=350)

image_layer5_1 = Image.open('/home/wangyang/桌面/interface/feature_map/layer5_1_84.jpg')
image_layer5_1_tk = ImageTk.PhotoImage(image_layer5_1)
label_image_layer5_1 = Label(frame2,image=image_layer5_1_tk)
label_image_layer5_1.place(x=500,y=250)

image_layer5_2 = Image.open('/home/wangyang/桌面/interface/feature_map/layer5_2_84.jpg')
image_layer5_2_tk = ImageTk.PhotoImage(image_layer5_2)
label_image_layer5_2 = Label(frame2,image=image_layer5_2_tk)
label_image_layer5_2.place(x=500,y=350)



text_image_label_output = StringVar()
text_image_label = StringVar()
path_split = path_single.split('/')
text_image_label.set(path_split[-2])
#label_predict = str(CNN_Chen_Eval.index_pos[0])
print(CNN_Chen_Eval.index_predict)
text_image_label_output.set(CNN_Chen_Eval.index_predict)
Label(frame2,text = "原始图像:",font=("Arial",11)).place(x=200,y=70)
Label(frame2,text = "图像类别:",font=("Arial",11)).place(x=450,y=70)
Label(frame2,textvariable=text_image_label,font=("Arial",11)).place(x=600,y=70)
Label(frame2,text = "图像类别:",font=("Arial",11)).place(x=450,y=120)
Label(frame2,textvariable=text_image_label_output,font=("Arial",11)).place(x=600,y=120)

#### define frame4###

path_frame4 = StringVar()

result_00 = StringVar()
result_01 = StringVar()
result_02 = StringVar()
result_03 = StringVar()
result_04 = StringVar()
result_05 = StringVar()
result_06 = StringVar()
result_07 = StringVar()
result_08 = StringVar()
result_09 = StringVar()
result_10 = StringVar()
result_11 = StringVar()
result_12 = StringVar()
result_13 = StringVar()
result_14 = StringVar()
result_15 = StringVar()
result_16 = StringVar()
result_17 = StringVar()
result_18 = StringVar()
result_19 = StringVar()
result_20 = StringVar()
result_21 = StringVar()
result_22 = StringVar()
result_23 = StringVar()
result_24 = StringVar()
result_25 = StringVar()
result_26 = StringVar()
result_27 = StringVar()
result_28 = StringVar()
result_29 = StringVar()
result_30 = StringVar()
result_31 = StringVar()
result_32 = StringVar()
result_33 = StringVar()
result_34 = StringVar()
result_35 = StringVar()
result_36 = StringVar()
result_37 = StringVar()
result_38 = StringVar()
result_39 = StringVar()
result_40 = StringVar()
result_41 = StringVar()
result_42 = StringVar()
result_43 = StringVar()
result_44 = StringVar()
result_45 = StringVar()
result_46 = StringVar()
result_47 = StringVar()
result_48 = StringVar()
result_49 = StringVar()
result_50 = StringVar()
result_51 = StringVar()
result_52 = StringVar()
result_53 = StringVar()
result_54 = StringVar()
result_55 = StringVar()
result_56 = StringVar()
result_57 = StringVar()
result_58 = StringVar()
result_59 = StringVar()
result_60 = StringVar()
result_61 = StringVar()
result_62 = StringVar()
result_63 = StringVar()
result_64 = StringVar()
result_65 = StringVar()
result_66 = StringVar()
result_67 = StringVar()
result_68 = StringVar()
result_69 = StringVar()
result_70 = StringVar()
result_71 = StringVar()
result_72 = StringVar()
result_73 = StringVar()
result_74 = StringVar()
result_75 = StringVar()
result_76 = StringVar()
result_77 = StringVar()
result_78 = StringVar()
result_79 = StringVar()
result_80 = StringVar()
result_81 = StringVar()
result_82 = StringVar()
result_83 = StringVar()
result_84 = StringVar()
result_85 = StringVar()
result_86 = StringVar()
result_87 = StringVar()
result_88 = StringVar()
result_89 = StringVar()
result_90 = StringVar()
result_91 = StringVar()
result_92 = StringVar()
result_93 = StringVar()
result_94 = StringVar()
result_95 = StringVar()
result_96 = StringVar()
result_97 = StringVar()
result_98 = StringVar()
result_99 = StringVar()
Label(frame4,textvariable=result_00,font=("Arial",11)).place(x=300,y=95),Label(frame4,textvariable=result_01,font=("Arial",11)).place(x=350,y=95)
Label(frame4,textvariable=result_02,font=("Arial",11)).place(x=400,y=95),Label(frame4,textvariable=result_03,font=("Arial",11)).place(x=450,y=95)
Label(frame4,textvariable=result_04,font=("Arial",11)).place(x=500,y=95),Label(frame4,textvariable=result_05,font=("Arial",11)).place(x=550,y=95)
Label(frame4,textvariable=result_06,font=("Arial",11)).place(x=600,y=95),Label(frame4,textvariable=result_07,font=("Arial",11)).place(x=650,y=95)
Label(frame4,textvariable=result_08,font=("Arial",11)).place(x=700,y=95),Label(frame4,textvariable=result_09,font=("Arial",11)).place(x=750,y=95)

Label(frame4,textvariable=result_10,font=("Arial",11)).place(x=300,y=120),Label(frame4,textvariable=result_11,font=("Arial",11)).place(x=350,y=120)
Label(frame4,textvariable=result_12,font=("Arial",11)).place(x=400,y=120),Label(frame4,textvariable=result_13,font=("Arial",11)).place(x=450,y=120)
Label(frame4,textvariable=result_14,font=("Arial",11)).place(x=500,y=120),Label(frame4,textvariable=result_15,font=("Arial",11)).place(x=550,y=120)
Label(frame4,textvariable=result_16,font=("Arial",11)).place(x=600,y=120),Label(frame4,textvariable=result_17,font=("Arial",11)).place(x=650,y=120)
Label(frame4,textvariable=result_18,font=("Arial",11)).place(x=700,y=120),Label(frame4,textvariable=result_19,font=("Arial",11)).place(x=750,y=120)

Label(frame4,textvariable=result_20,font=("Arial",11)).place(x=300,y=145),Label(frame4,textvariable=result_21,font=("Arial",11)).place(x=350,y=145)
Label(frame4,textvariable=result_22,font=("Arial",11)).place(x=400,y=145),Label(frame4,textvariable=result_23,font=("Arial",11)).place(x=450,y=145)
Label(frame4,textvariable=result_24,font=("Arial",11)).place(x=500,y=145),Label(frame4,textvariable=result_25,font=("Arial",11)).place(x=550,y=145)
Label(frame4,textvariable=result_26,font=("Arial",11)).place(x=600,y=145),Label(frame4,textvariable=result_27,font=("Arial",11)).place(x=650,y=145)
Label(frame4,textvariable=result_28,font=("Arial",11)).place(x=700,y=145),Label(frame4,textvariable=result_29,font=("Arial",11)).place(x=750,y=145)

Label(frame4,textvariable=result_30,font=("Arial",11)).place(x=300,y=170),Label(frame4,textvariable=result_31,font=("Arial",11)).place(x=350,y=170)
Label(frame4,textvariable=result_32,font=("Arial",11)).place(x=400,y=170),Label(frame4,textvariable=result_33,font=("Arial",11)).place(x=450,y=170)
Label(frame4,textvariable=result_34,font=("Arial",11)).place(x=500,y=170),Label(frame4,textvariable=result_35,font=("Arial",11)).place(x=550,y=170)
Label(frame4,textvariable=result_36,font=("Arial",11)).place(x=600,y=170),Label(frame4,textvariable=result_37,font=("Arial",11)).place(x=650,y=170)
Label(frame4,textvariable=result_38,font=("Arial",11)).place(x=700,y=170),Label(frame4,textvariable=result_39,font=("Arial",11)).place(x=750,y=170)

Label(frame4,textvariable=result_40,font=("Arial",11)).place(x=300,y=195),Label(frame4,textvariable=result_41,font=("Arial",11)).place(x=350,y=195)
Label(frame4,textvariable=result_42,font=("Arial",11)).place(x=400,y=195),Label(frame4,textvariable=result_43,font=("Arial",11)).place(x=450,y=195)
Label(frame4,textvariable=result_44,font=("Arial",11)).place(x=500,y=195),Label(frame4,textvariable=result_45,font=("Arial",11)).place(x=550,y=195)
Label(frame4,textvariable=result_46,font=("Arial",11)).place(x=600,y=195),Label(frame4,textvariable=result_47,font=("Arial",11)).place(x=650,y=195)
Label(frame4,textvariable=result_48,font=("Arial",11)).place(x=700,y=195),Label(frame4,textvariable=result_49,font=("Arial",11)).place(x=750,y=195)

Label(frame4,textvariable=result_50,font=("Arial",11)).place(x=300,y=220),Label(frame4,textvariable=result_51,font=("Arial",11)).place(x=350,y=220)
Label(frame4,textvariable=result_52,font=("Arial",11)).place(x=400,y=220),Label(frame4,textvariable=result_53,font=("Arial",11)).place(x=450,y=220)
Label(frame4,textvariable=result_54,font=("Arial",11)).place(x=500,y=220),Label(frame4,textvariable=result_55,font=("Arial",11)).place(x=550,y=220)
Label(frame4,textvariable=result_56,font=("Arial",11)).place(x=600,y=220),Label(frame4,textvariable=result_57,font=("Arial",11)).place(x=650,y=220)
Label(frame4,textvariable=result_58,font=("Arial",11)).place(x=700,y=220),Label(frame4,textvariable=result_59,font=("Arial",11)).place(x=750,y=220)

Label(frame4,textvariable=result_60,font=("Arial",11)).place(x=300,y=245),Label(frame4,textvariable=result_61,font=("Arial",11)).place(x=350,y=245)
Label(frame4,textvariable=result_62,font=("Arial",11)).place(x=400,y=245),Label(frame4,textvariable=result_63,font=("Arial",11)).place(x=450,y=245)
Label(frame4,textvariable=result_64,font=("Arial",11)).place(x=500,y=245),Label(frame4,textvariable=result_65,font=("Arial",11)).place(x=550,y=245)
Label(frame4,textvariable=result_66,font=("Arial",11)).place(x=600,y=245),Label(frame4,textvariable=result_67,font=("Arial",11)).place(x=650,y=245)
Label(frame4,textvariable=result_68,font=("Arial",11)).place(x=700,y=245),Label(frame4,textvariable=result_69,font=("Arial",11)).place(x=750,y=245)

Label(frame4,textvariable=result_70,font=("Arial",11)).place(x=300,y=270),Label(frame4,textvariable=result_71,font=("Arial",11)).place(x=350,y=270)
Label(frame4,textvariable=result_72,font=("Arial",11)).place(x=400,y=270),Label(frame4,textvariable=result_73,font=("Arial",11)).place(x=450,y=270)
Label(frame4,textvariable=result_74,font=("Arial",11)).place(x=500,y=270),Label(frame4,textvariable=result_75,font=("Arial",11)).place(x=550,y=270)
Label(frame4,textvariable=result_76,font=("Arial",11)).place(x=600,y=270),Label(frame4,textvariable=result_77,font=("Arial",11)).place(x=650,y=270)
Label(frame4,textvariable=result_78,font=("Arial",11)).place(x=700,y=270),Label(frame4,textvariable=result_79,font=("Arial",11)).place(x=750,y=270)

Label(frame4,textvariable=result_80,font=("Arial",11)).place(x=300,y=295),Label(frame4,textvariable=result_81,font=("Arial",11)).place(x=350,y=295)
Label(frame4,textvariable=result_82,font=("Arial",11)).place(x=400,y=295),Label(frame4,textvariable=result_83,font=("Arial",11)).place(x=450,y=295)
Label(frame4,textvariable=result_84,font=("Arial",11)).place(x=500,y=295),Label(frame4,textvariable=result_85,font=("Arial",11)).place(x=550,y=295)
Label(frame4,textvariable=result_86,font=("Arial",11)).place(x=600,y=295),Label(frame4,textvariable=result_87,font=("Arial",11)).place(x=650,y=295)
Label(frame4,textvariable=result_88,font=("Arial",11)).place(x=700,y=295),Label(frame4,textvariable=result_89,font=("Arial",11)).place(x=750,y=295)

Label(frame4,textvariable=result_90,font=("Arial",11)).place(x=300,y=320),Label(frame4,textvariable=result_91,font=("Arial",11)).place(x=350,y=320)
Label(frame4,textvariable=result_92,font=("Arial",11)).place(x=400,y=320),Label(frame4,textvariable=result_93,font=("Arial",11)).place(x=450,y=320)
Label(frame4,textvariable=result_94,font=("Arial",11)).place(x=500,y=320),Label(frame4,textvariable=result_95,font=("Arial",11)).place(x=550,y=320)
Label(frame4,textvariable=result_96,font=("Arial",11)).place(x=600,y=320),Label(frame4,textvariable=result_97,font=("Arial",11)).place(x=650,y=320)
Label(frame4,textvariable=result_98,font=("Arial",11)).place(x=700,y=320),Label(frame4,textvariable=result_99,font=("Arial",11)).place(x=750,y=320)

def selectPath_frame4():
	path_ = askopenfilename()
	path.set(path_)
	global path_record
	path_record=path_
	print(path_record)

Label(frame4,text = "目标路径:",font=("Arial",10)).place(x=350,y=12)
entry1 = Entry(frame4, textvariable = path,font=("Arial",10)).place(x=430,y=12)
Button(frame4, text = "路径选择", command = selectPath_frame4,font=("Arial",10)).place(x=580,y=10)
Button(frame4, text = "运行", command = run_2,font=("Arial",10)).place(x=670,y=10)
Label(frame4,text = "原始类别 :",font=("Arial",11)).place(x=200,y=70)
Label(frame4,text = "第0类(274)",font=("Arial",11)).place(x=200,y=95)
Label(frame4,text = "第1类(195)",font=("Arial",11)).place(x=200,y=120)
Label(frame4,text = "第2类(274)",font=("Arial",11)).place(x=200,y=145)
Label(frame4,text = "第3类(195)",font=("Arial",11)).place(x=200,y=170)
Label(frame4,text = "第4类(196)",font=("Arial",11)).place(x=200,y=195)
Label(frame4,text = "第5类(274)",font=("Arial",11)).place(x=200,y=220)
Label(frame4,text = "第6类(273)",font=("Arial",11)).place(x=200,y=245)
Label(frame4,text = "第7类(196)",font=("Arial",11)).place(x=200,y=270)
Label(frame4,text = "第8类(274)",font=("Arial",11)).place(x=200,y=295)
Label(frame4,text = "第9类(274)",font=("Arial",11)).place(x=200,y=320)
Label(frame4,text = "预测类别:",font=("Arial",11)).place(x=220,y=50)
Label(frame4,text = "第0类 ",font=("Arial",11)).place(x=300,y=50)
Label(frame4,text = "第1类 ",font=("Arial",11)).place(x=350,y=50)
Label(frame4,text = "第2类 ",font=("Arial",11)).place(x=400,y=50)
Label(frame4,text = "第3类 ",font=("Arial",11)).place(x=450,y=50)
Label(frame4,text = "第4类 ",font=("Arial",11)).place(x=500,y=50)
Label(frame4,text = "第5类 ",font=("Arial",11)).place(x=550,y=50)
Label(frame4,text = "第6类 ",font=("Arial",11)).place(x=600,y=50)
Label(frame4,text = "第7类 ",font=("Arial",11)).place(x=650,y=50)
Label(frame4,text = "第8类 ",font=("Arial",11)).place(x=700,y=50)
Label(frame4,text = "第9类 ",font=("Arial",11)).place(x=750,y=50)


#[274, 195, 274, 195, 196, 274, 273, 196, 274, 274]
menubar=Menu(root)
function_select_menu = Menu(menubar,tearoff=0)
function_select_menu.add_command(label="home",command=back_to_frame1)
function_select_menu.add_command(label="model information",command=to_frame3)
function_select_menu.add_command(label="test on single image",command=to_frame2)
function_select_menu.add_command(label="test on dataset",command=to_frame4)
function_select_model = Menu(menubar,tearoff=0)
function_select_model.add_command(label="model1",command=select_model_1)
function_select_model.add_command(label="model2",command=select_model_2)
function_select_model.add_command(label="model3",command=select_model_3)
menubar.add_cascade(label="select",menu=function_select_menu)
menubar.add_cascade(label="model",menu=function_select_model)
root.config(menu=menubar)

root.mainloop()
