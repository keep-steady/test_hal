import numpy as np
import cv2 as cv
import serial
import datetime, time

# arduino = serial.Serial('COM3',9600)



# 0:딸기, 1:바나나, 2:라임, 3:플럼

def get_lu(i):      #ColorPicker에서 측정한 각 과일의 Hue와 Saturation, Value 범위 세 가지를 리턴해주는 함수
    h = [2,19,38,155]
    s = [50,50,50,20]
    v = [50,50,50,20]

    hue = h[i]
    sl = s[i]
    su = 255
    vl = v[i]
    vu = 255
    range = 10

    if hue < 10:
        l1 = np.array([hue-range+180, sl, vl])
        u1 = np.array([180, su, vu])
        l2 = np.array([0, sl, vl])
        u2 = np.array([hue, su, vu])
        l3 = np.array([hue, sl, vl])
        u3 = np.array([hue+range, su, vu])

    elif hue > 170:
        l1 = np.array([hue, sl, vl])
        u1 = np.array([180, su, vu])
        l2 = np.array([0, sl, vl])
        u2 = np.array([hue+range-180, su, vu])
        l3 = np.array([hue-range, sl, vl])
        u3 = np.array([hue, su, vu])

    else:
        l1 = np.array([hue, sl, vl])
        u1 = np.array([hue+range, su, vu])
        l2 = np.array([hue-range, sl, vl])
        u2 = np.array([hue+range-180, su, vu])
        l3 = np.array([hue-range, sl, vl])
        u3 = np.array([hue, su, vu])

    return l1,l2,l3,u1,u2,u3

def get_mask(i):        #각 과일을 detection한 mask를 리턴해주는 함수
    l1 = get_lu(i)[0]
    l2 = get_lu(i)[1]
    l3 = get_lu(i)[2]
    u1 = get_lu(i)[3]
    u2 = get_lu(i)[4]
    u3 = get_lu(i)[5]

    mask1 = cv.inRange(card_hsv, l1, u1)
    mask2 = cv.inRange(card_hsv, l2, u2)
    mask3 = cv.inRange(card_hsv, l3, u3)
    mask = mask1 | mask2 | mask3

    kernel = np.ones((4,4), np.uint8)       #커널이 크면 오픈/클로즈 때 가까운 과일 끼리 붙을 수 있다! 이미지 크기에 따라 조절해야함
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask

def labeled_components(labels_set):     #카드 내 과일 덩어리 labels 외곽선 궁금해서 그려주는 함수
    final_labeled_area =  np.zeros((height, width, 3), np.uint8)

    for i in range(0,4):
        labels =  labels_set[i]
        hue = 90
        labeled_hue = np.zeros((height, width), np.uint8)
        labeled_hue[labels != 0] = hue
        blank_ch = 255 * np.ones_like(labeled_hue)
        labeled_area = cv.merge([labeled_hue, blank_ch, blank_ch])
        labeled_area = cv.cvtColor(labeled_area, cv.COLOR_HSV2BGR)
        labeled_area[labeled_hue == 0] = 0
        final_labeled_area += labeled_area

    return final_labeled_area




#이상 함수 정의
#이하 레알 이미지 처리
cap = cv.VideoCapture(1+cv.CAP_DSHOW)
# cap = cv.VideoCapture('img/sample_original.mp4')
cap.set(cv.CAP_PROP_FPS,30)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)

fourcc = cv.VideoWriter_fourcc(*'XVID')
writer = cv.VideoWriter('img/output.avi', fourcc, 30, (1080, 1080))
print(cap.get(cv.CAP_PROP_FPS))
print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))



last_data = b'\n'
prevTime = 0

# 해상도에 따른 카드/과일 넓이 범위 자동 설정
# 1080 : 3
#  900 : 4
#  800 : 5
#  700 : 6
#  600 : 7
#  500 : 10
#  400 : 12
#  300 : 13
set_height = set_width = 300  # 250~1080은 잘 잡음
card_min_area = int(np.square(175  / 1080 * set_height))
card_max_area = int(np.square(240  / 1080 * set_height))
fruit_min_area= int(np.square(41.2 / 1080 * set_height))
fruit_max_area= int(np.square(63.25/ 1080 * set_height))

print(card_min_area, card_max_area, fruit_min_area, fruit_max_area)

while(True):

    # 680x680 : 20 -> 당첨!
    # 691x691 : 19
    # 702x702 : 10
    # 756x756 : 6
    # #사진 사용할 경우
    # source = cv.imread('img/sample_1.png')
    # source = source[:,420:1500]  # 1080x1080으로 자르기
    # source = cv.resize(source, (set_height, set_width), interpolation=cv.INTER_AREA)  # 700*700이면 충분할듯

    
    #웹캠 사용할 경우
    ret, source = cap.read()
    if ret == False:
        continue

    
    source2 = source.copy()
    height = source.shape[0]
    width = source.shape[1]

    ########### fps 계산 ##################
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime  #이전 시간을 현재시간으로 다시 저장시킴
    fps = 1 / (sec)
    # 프레임 수를 문자열에 저장
    fps_str = "FPS : %.0f | shape : %d|%d" % (fps, height, width)
    cv.putText(source, fps_str, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 표시
    ###################################    
    
    source_gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    ret, source_binary = cv.threshold(source_gray, 200, 255, 0)
    # cv.imshow('source_binary', source_binary)

    contours, hierarchy = cv.findContours(source_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cards_mask = np.zeros((height, width), np.uint8)

    fruits = ['Strawberry', 'Banana', 'Lime', 'Plum']
    counts = [0, 0, 0, 0]  # Total count. 따로 빼야할듯!


    # 카드 하나당 알고리즘
    for contour in contours:
        count = 0
        layer = np.zeros((height, width), np.uint8)
        x, y, w, h = cv.boundingRect(contour)       #카드를 둘러싸는 상자의 수치

        if (card_min_area < w * h < card_max_area) and (0.5 < w/h < 1.7):
            # print('w=',w,'h=',h,'사각형의 넓이=',w*h, 'w/h=',w/h)     #for 타겟 범위 좁히기 용도
            cv.rectangle(source, (x, y), (x + w, y + h), (0, 255, 0), 2)

            layer[y:y+h, x:x+w] = 255       # ROI 설정
            cards_mask[y:y+h, x:x+w] = 255      # 화면 출력용 카드 마스크 레이어에도 추가

            card = cv.bitwise_and(source2, source2, mask=np.uint8(layer))
            card_hsv = cv.cvtColor(card, cv.COLOR_BGR2HSV)

            # labels_set = [0,0,0,0]        # for labeled_components


            #카드 한장 안에서 과일 디텍션 시작
            for i, fruit in enumerate(fruits):

                number, labels, stats, centroids = cv.connectedComponentsWithStats(get_mask(i))  # 덩어리들을 detect
                # labels_set[i] = labels        # for labeled_components

                #이 과일이 아닌가벼...
                if number <= 1:     
                    continue

                #덩어리가 1개 이상 포착될 경우
                else:       
                    # 각 덩어리 j에 대해서
                    for j, centroid in enumerate(centroids):
                        if stats[j][0] == 0 and stats[j][1] == 0:       #배경 버리고
                            continue

                        if np.any(np.isnan(centroid)):      #잡것 버리고
                            continue

                        x2, y2, w2, h2, A2 = stats[j]  # 덩어리의 수치
                        cx, cy = int(centroid[0]), int(centroid[1])
                        # print('덩어리의 넓이=',A2)

                        if fruit_min_area < A2 < fruit_max_area:  # 과일 크기에 맞는 적절한 size의 덩어리들만 골라내서 count (해상도 변경에 따라 magnification^2 곱해줌)
                            # 근데 Card의 content가 아닌 경계부에서 덩어리가 발견? =딸기 비슷한 손가락 끝! X치고 배제!
                            s = 1  # sensor의 넓이
                            p = 2  # pixel #와 array #의 차이로 인해 생긴 오차 수정
                            if layer[y2 - s - p, x2 - s - p] == 0 or layer[y2 - s - p, x2 + w2 + s - p] == 0 or layer[y2 + h2 + s - p, x2 - s - p] == 0 or layer[y2 + h2 + s - p, x2 + w2 + s - p] == 0:
                                cv.drawMarker(source, (x2, y2), (0, 0, 255), cv.MARKER_TILTED_CROSS, markerSize=20,
                                            thickness=2)

                            else:       #손가락 아니면 과일로 인지용~
                                count += 1
                                cv.circle(source, (cx, cy), 10, (255, 0, 0), 2)

                        else:
                            continue

                    if count == 0:
                        continue

                    m = 5       #중심점 캐치용 마진
                    q = 2       #대칭 보정용 마진

                    if count == 3:
                        if labels[round(y+h/2)-m:round(y+h/2)+m+q,round(x+w/2)-m:round(x+w/2)+m+q].any():
                            count = 3
                        else:
                            count = 4

                    elif count == 4:
                        if labels[round(y+h/2)-m:round(y+h/2)+m+q,round(x+w/2)-m:round(x+w/2)+m+q].any():
                            count = 5
                        else:
                            count = 4


                    counts[i] += count

                    break


        else:
            continue           #카드 사이즈 안맞으면(=다른 물체) 다음 카드로


    print('counts(딸,바,라,플)=', counts, fps_str)
    text_color = [(65,65,204),(51,194,234),(69,219,189),(155,54,170)]
    for i, fruit in enumerate(fruits):
        text = fruit +':'+ str(counts[i])
        cv.putText(source,text,(5,30*(3+i)),cv.FONT_HERSHEY_SIMPLEX,0.5,text_color[i],2)

    # cv.imshow('labeled_area',labeled_components(labels_set))

    led_color = [b'r\n', b'y\n', b'g\n', b'b\n']

    if 5 in counts:      #5개 짜리 과일이 있는 경우, BELL!!!!!
        data = led_color[counts.index(5)]

        if data == last_data:
            pass
        else:
            # arduino.write(data)
            print(data)
            last_data = data

        print("BELL!")
        cv.putText(source,"BELL!",(int(width/2)-100,int(height/2)+20), cv.FONT_HERSHEY_COMPLEX, 3, (0,0,255),5)

    else:
        data = b'd\n'

        if data == last_data:
            pass
        else:
            # arduino.write(data)
            print(data)
            last_data = data

    cv.imshow('cards_mask',cards_mask)
    cv.imshow('source',source)

    # writer.write(source)      ##녹화 사용할 경우 켜기

    if cv.waitKey(1) & 0xFF == 27:
        break


cap.release()
writer.release()
cv.destroyAllWindows()