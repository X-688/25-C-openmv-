# 25C - By: 688 - Wed Jul 30 2025
# 基于单目视觉的目标物测量装置
# 使用硬件：openmv H7
# 通过串口与单片机通信，并根据串口接收的内容执行不同的功能函数
# 可实现除了重叠矩形外的所有要求，精度在0.01mm
# 除了数字识别外，不使用深度学习
import sensor
import time
import pyb
import math
import tf
import image
# ========初始化==========
uart = pyb.UART(3,115200,timeout_char = 1000)  #P4-TXD，P5-RXD
clock = time.clock()
led_red = pyb.LED(1)
led_green = pyb.LED(2)
# ========全局变量区=======
BK_TH = 120
WT_TH = 80
BLACK_TH = (0,BK_TH)   #影响目标物边长识别
WHITE_TH = (WT_TH,255) #影响数字提取
ROI = (220,160)  #VGA 640*480
ROI_CENTER = (80,110)
MIN_SIZE = 60    #单位mm
MAX_SIZE = 160   #单位mm
FOCUS_MM = 653.7 #单位mm
A4W_MM = 170     #单位mm
MIN_BLOB_AREA = 300            # 最小黑色矩形像素面积 实测306
MAX_BLOB_AREA = 10000          # 最大黑色矩形像素面积 实测9801
rotateRate = 1   #现实/预期 若旋转则<1
Distance_mm = 0  #单位mm
A4_Range = None
DEBUG_MODE = 0
OFFSET = 0
# =======================
def init():
    led_red.off()
    led_green.off()
    sensor.reset()
    sensor.set_pixformat(sensor.GRAYSCALE)
    sensor.set_framesize(sensor.VGA) # 480*640
    sensor.set_auto_gain(False)
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False)
    sensor.set_transpose(True)
    sensor.set_vflip(True)
    sensor.set_windowing(ROI) #裁剪到 #area 120*150
    sensor.set_contrast(1)
    sensor.skip_frames(20)

def Cal_D(A4_height):
    global OFFSET
    if(A4_height):
        OFFSET = A4W_MM*0.5*math.sqrt(1-rotateRate**2)
        return 168000/A4_height+ OFFSET#旋转修正
    else:
        return 0

def getLength(line):
    if (type(line)!= tuple) or (len(line)!=4):
        return 0
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def find_min(blobs,min_length_limit,max_length_limit):
    #在大于min_length的边长（用最小区域矩形的最大边长）里找最短，可兼容所有情况
    min_blob = None
    min_length = 10000
    if(rotateRate != 1):#根据A4的旋转情况修正了min_length_limit的值，并修正测量的边长
        min_length_limit = min_length_limit*rotateRate
        for blob in blobs:
            inner_rect_max = blob.major_axis_line()
            blob_length = getLength(inner_rect_max)
            offset = (Distance_mm-OFFSET)/Distance_mm
            blob_length *= offset
            if blob_length < min_length_limit: #长度、高度小于最小值（像素）
                continue
            if blob.h()/blob.w() < 0.8: #高宽比 比等边三角形还小，滤掉
                continue
            if blob_length < min_length: #用最小区域矩形的最大边长做比较
                min_blob = blob
                min_length = blob_length #取最小区域矩形的最大边长
        data = min_length
        if min_blob:
            print("产生了修正后的矩形")
            return data

    for blob in blobs:
        inner_rect_max = blob.major_axis_line()
        blob_length = getLength(inner_rect_max)
        if blob_length < min_length_limit: #长度、高度小于最小值（像素）
            continue
        if blob.h()/blob.w() < 0.8: #高宽比 比等边三角形还小，滤掉
            continue
        if blob_length < min_length: #用最小区域矩形的最大边长做比较
            min_blob = blob
            min_length = blob_length #取最小区域矩形的最大边长
    data = min_length
    if min_blob:
        return data
    return -1

def detectA4rotate(rect,):
    global rotateRate
    tolerance_rate = 0.05
    tolerance = 10
    if rect:
        # expect_width = rect.h()/1.414 #A4比例
        expect_width = rect.h()/1.5 #A4内边框比例
        actual_width = rect.w()
        if actual_width > expect_width + tolerance:
            print("这是哪门子A4？？")
            led_red.off()
            led_green.off()
            return 0
        elif actual_width < expect_width*0.5 -tolerance: #这个好像基本不触发
            print("转过头了 滤掉吧")
            led_red.off()
            led_green.off()
            return 0
        elif expect_width*(0.5-tolerance_rate)< actual_width < expect_width*(0.866+tolerance_rate):
            rotateRate = actual_width/expect_width
            led_green.on()
            led_red.off()
            return 1
        else:
            rotateRate = 1
            led_green.off()
            led_red.on()
            return 1
    else:
        return 0

def find_center_min_blob(blobs,min_length_limit=50):
    # 找中间最小的色块（白色）
    blob = None
    min_area = 100000
    for b in blobs:
        if abs(b.cx()-ROI_CENTER[0])+ abs(b.cy()-ROI_CENTER[1]) > 40:
            continue
        if b.area() > min_area:
            continue
        if b.w() < min_length_limit:
            continue
        blob = b
        min_area = b.area()
    return blob

def Find_Frame():
    global Distance_mm
    img = sensor.snapshot()
    blobs = img.find_blobs([WHITE_TH])
    img.draw_circle(ROI_CENTER[0],ROI_CENTER[1],3,color=255)
    frame_blob = find_center_min_blob(blobs) #找到A4中间的大白块
    if not frame_blob:
        print("NO FRAME")
        led_red.off() #没找到或被滤掉，则关灯
        return None
    img.draw_circle(frame_blob.cx(),frame_blob.cy(),3,color=128)
    if(detectA4rotate(frame_blob)): #对检测到的白色框架进行长宽比检测
        if DEBUG_MODE:
            print("h:",frame_blob.h(),"w:",frame_blob.w())
            print("Distance_mm:",Distance_mm,"mm rotateRate:",rotateRate)
        Distance_mm = Cal_D(frame_blob.h()) #取高，防一手旋转
        img.draw_rectangle(frame_blob.rect())
        return frame_blob.rect() #检测通过
    return None

def pack_data(data_list):
    packet = bytearray([0xAA])
    for d in data_list:
        packet.append(d)
    return packet

def process_edge(img):
    # 获取图像尺寸
    width = img.width()
    height = img.height()
    rag = 0.05
    # 计算边缘区域尺寸（至少1像素）
    edge_width = max(1, int(width * rag))  # 左右边缘宽度
    edge_height = max(1, int(height * rag))  # 上下边缘高度
    # 创建处理后的图像副本
    result = img.copy()
    # 填充左侧边缘
    result.draw_rectangle(0, 0, edge_width, height, color=255, fill=True)
    # 填充右侧边缘
    result.draw_rectangle(width - edge_width, 0, edge_width, height, color=255, fill=True)
    # 填充顶部边缘
    result.draw_rectangle(0, 0, width, edge_height, color=255, fill=True)
    # 填充底部边缘
    result.draw_rectangle(0, height - edge_height, width, edge_height, color=255, fill=True)
    return result

def preprocess_digit(img, roi):
    """
    从黑色矩形区域提取数字并预处理
    :param img: 原始图像
    :param roi: 黑色矩形区域 (x, y, w, h)
    :return: 预处理后的数字图像
    """
    x, y, w, h = roi
    # 提取黑色矩形区域
    digit_region = img.copy(roi=roi)
    # 二值化提取白色数字
    digit_binary = digit_region.binary([WHITE_TH])
    # 创建纯黑背景
    black_bg = image.Image(w, h, sensor.GRAYSCALE)
    black_bg.clear()
    # 将白色数字绘制到黑背景上
    black_bg.draw_image(digit_binary, 0, 0)
    return black_bg

def recognize_digit(img, roi):
    """
    识别黑色矩形中的数字
    :param img: 原始图像
    :param roi: 黑色矩形区域 (x, y, w, h)
    :return: 识别出的数字 (0-9) 或 -1 (识别失败)
    """
    # 预处理数字区域
    digit_img = preprocess_digit(img, roi)
    # 识别数字
    for obj in tf.classify("trained.tflite", digit_img,
                         min_scale=1.0, scale_mul=0.5,
                         x_overlap=0.0, y_overlap=0.0):
        predictions = obj.output()
        # 获取置信度最高的数字
        max_confidence = max(predictions)
        if max_confidence > 0.7:  # 设置置信度阈值
            return predictions.index(max_confidence)
    return -1

def find_black_rectangles(img, roi):
    """
    在指定区域内查找黑色矩形
    :param img: 原始图像
    :param roi: 搜索区域 (x, y, w, h)
    :return: 黑色矩形列表 [blob]
    """
    # 寻找黑色矩形
    blobs = img.find_blobs([BLACK_TH],area_threshold=MIN_BLOB_AREA,merge=False)
    # 过滤过大和过小的区域
    filtered_blobs = []
    for blob in blobs:
        if MIN_BLOB_AREA <= blob.area() <= MAX_BLOB_AREA:
            filtered_blobs.append(blob)
            img.draw_rectangle(blob.rect())#debug
    return filtered_blobs

def find_target_rectangle(img, target_digit):
    """
    查找包含目标数字的黑色矩形
    :param img: 原始图像
    :param target_digit: 目标数字
    :return: 包含目标数字的blob对象 (找不到返回None)
    """
    # 整个图像都是白色背景区域
    white_frame = (0, 0, img.width(), img.height())
    # 查找所有黑色矩形
    black_rects = find_black_rectangles(img, white_frame)
    # 遍历所有黑色矩形
    for rect in black_rects:
        # 识别矩形中的数字
        digit = recognize_digit(img, rect.rect())
        # 检查是否为目标数字
        if digit == target_digit:
            return rect
    return None

#=======主要功能函数========
def digital_function(digit):
    if(type(digit)!= int)or(digit<0)or(digit>9):
        print("invalid digit, return..")
        return
    print("==digital function==",digit)
    retry = 0
    while True:
        if(A4_Range == None):
            print("invalid:A4_range")
            return
        img = sensor.snapshot().scale(roi = A4_Range)
        target_rect = find_target_rectangle(img, digit)
        if target_rect:
            min_length = getLength(target_rect.major_axis_line())
            # 绘制目标矩形
            # img.draw_rectangle(target_rect.rect(), color=255, thickness=2)
            # 标记数字
            # img.draw_string(target_rect.x(), target_rect.y()-10,f"Target: {digit}",color=255, scale=1,x_spacing=-4)
            # 打印处理结果
            Distance_mm_full = round(Distance_mm*100) # 精度 0000.01mm
            R_full = round(Distance_mm_full*min_length/FOCUS_MM) #0.01mm
            print("R:",R_full/100,"mm","Distance_mm:",Distance_mm_full/100,"mm")
            message = pack_data([int(Distance_mm_full//10000),int(Distance_mm_full//100%100),int(Distance_mm_full%100),int(R_full//10000),int(R_full//100%100),int(R_full%100)])
            uart.write(message)
            return
        else:
            retry += 1
            if retry == 50:
                print("timeout not found")
                return
    return

def find_min_function():
    global Distance_mm
    div_count = 0
    div_sum = 0
    print("==find min function==")
    size_min_pixel = MIN_SIZE*FOCUS_MM/Distance_mm-5 #单位pixel 增加了一些裕度
    size_max_pixel = MAX_SIZE*FOCUS_MM/Distance_mm+5 #单位pixel
    while True:
        if((type(A4_Range) != tuple) and (type(A4_Range) != list)):
            print("invalid:A4_range")
            return
        img = sensor.snapshot().scale(roi = A4_Range)
        img = process_edge(img)
        blobs = img.find_blobs([BLACK_TH])
        if blobs:
            min_length = find_min(blobs,size_min_pixel,size_max_pixel)
            if min_length != -1:
                div_count += 1
                div_sum += min_length
                if(div_count >= 20):
                    Distance_mm_full = round(Distance_mm*100) # 精度 0000.01mm
                    div_sum = div_sum/20
                    R_full = round(Distance_mm_full*div_sum/FOCUS_MM) #0.01mm
                    print("R:",R_full/100,"mm","Distance_mm:",Distance_mm_full/100,"mm","rotate offset:",A4W_MM*0.5*((1-rotateRate)**2)*10,"mm")
                    message = pack_data([int(Distance_mm_full//10000),int(Distance_mm_full//100%100),int(Distance_mm_full%100),int(R_full//10000),int(R_full//100%100),int(R_full%100)])
                    uart.write(message)
                    return
            else:
                print("min length not found")
        else:
            print("未发现图形（from find min）")
    return
#====program start here====
while True:
    init()
    while True: #校准模式
        A4_Range = Find_Frame()
        if(A4_Range != None):
            if(uart.any()):
                receive_message = uart.readline()
                print("receive:",receive_message[0])
                led_red.off()
                led_green.off()
                break
    if receive_message[0] == 0xff: #默认模式
        find_min_function()
    elif receive_message[0] == 0xa0: #调试模式 黑色范围减少
        BK_TH -= 5
        print("BLACK-")
    elif receive_message[0] == 0xa1: #调试模式 黑色范围增加
        BK_TH += 5
        print("BLACK+")
    elif receive_message[0] == 0xb1: #调试模式 白色范围增加
        WT_TH -= 5
        print("WHITE+")
    elif receive_message[0] == 0xb0: #调试模式 白色范围减少
        WT_TH += 5
        print("WHITE-")
    else: # 数字识别模式
        digital_function(receive_message[0])
#========================
