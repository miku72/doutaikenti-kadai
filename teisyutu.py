import cv2
import numpy as np
import os

# 動体検知に使用するアルゴリズムを選択
model = cv2.bgsegm.createBackgroundSubtractorMOG()
# テキストファイル
k = open("ekuseru.txt","w")
# 画像各種設定
# 連番画像のパス
#img_path = r"images"
image_folder = "images"
# ファイル名
template = "{:06d}.bmp"
# フレーム数指定
start_frame = 1
end_frame = 1030
# 変数
detection_count = 0
now = 0
framecount = 0

# 処理
for frame_number in range(start_frame, end_frame + 1):
    # 画像のファイル名を取得します
    filename = template.format(frame_number)
    filepath = os.path.join(image_folder, filename)
    # 画像の読み込み
    frame = cv2.imread(filepath)

    # 動体検知
    mask = model.apply(frame)


    # 輪郭抽出
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    bboxes = list(map(lambda x: cv2.boundingRect(x), contours))

    # 動体が検知された場合
    if len(contours) > 0:
        detection_count += 1
        print(str(framecount)+"  1")
        k.write("1\n")


        # 検知した物を赤枠で囲む
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    else:
        k.write("0\n")
        print(str(framecount)+"  0")

    # 画像の表示
    cv2.imshow('frame', frame)

    framecount+=1
    #kakikomu = ("framecount "+str(framecount),"  hito  "+str(now))

    # qを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    '''
    with open('detection_count.txt', 'w') as f:
        f.write(str(detection_count)+"\n")
with open('now.txt', 'a') as f:
        f.write(str(detection_count)+"\n")
        '''
k.close()

# 後始末
cv2.destroyAllWindows()
