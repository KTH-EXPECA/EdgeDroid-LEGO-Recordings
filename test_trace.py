import cv2
import numpy as np

trace = './square00.npz'


def play_trace():
    trace_imgs = np.load(trace)
    cv2.namedWindow('Trace')

    for name, img in trace_imgs.items():
        cv2.putText(
            img=img,
            text=name,
            color=(0, 0, 255),
            org=(img.shape[0] // 10, img.shape[1] // 10),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            thickness=2
        )
        cv2.imshow(
            'Trace',
            img,
        )
        cv2.waitKey(50)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    play_trace()
