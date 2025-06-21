import numpy as np 
import cv2 

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def auto_scan_image_via_webcam():
    try:
        cap = cv2.VideoCapture(0)
    except:
        print('Cannot load Camera!')
        return

    while True:
        ret, frame = cap.read()
        if ret == True:
            k = cv2.waitKey(10)
            if k == 27:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            edged = cv2.Canny(gray, 75, 200)

            (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

            screenCnt = []
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    contourSize = cv2.contourArea(approx)
                    camSize = frame.shape[0] * frame.shape[1]
                    ratio = contourSize / camSize
                    if ratio > 0.01:
                        screenCnt = approx
                        break

            if len(screenCnt) == 0:
                cv2.imshow("WebCam", frame)
                continue
            else:
                cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
                cv2.imshow("WebCam", frame)

                rect = order_points(screenCnt.reshape(4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = rect

                w1 = abs(bottomRight[0] - bottomLeft[0])
                w2 = abs(topRight[0] - topLeft[0])
                h1 = abs(topRight[1] - bottomRight[1])
                h2 = abs(topLeft[1] - bottomLeft[1])
                maxWidth = max([w1, w2])
                maxHeight = max([h1, h2])

                dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
                cv2.imshow("Scanned", warped)
        else:
            print('Cannot load Camera!')
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == '__main__':
    auto_scan_image_via_webcam()
