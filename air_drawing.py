import cv2
import mediapipe as mp
import numpy as np
from collections import deque

#  KONFIGURASI
COLORS = {
    "RED":    (0,   0,   255),
    "GREEN":  (0,   255, 0),
    "BLUE":   (255, 0,   0),
    "YELLOW": (0,   255, 255),
    "WHITE":  (255, 255, 255),
}
COLOR_KEYS = list(COLORS.keys())

# Smoothing: 0.0 = no smooth, 0.7 = sangat halus
SMOOTH_FACTOR = 0.55

# Stabilizer: berapa frame mode harus konsisten sebelum berganti
MODE_STABLE_FRAMES = 4

#  STATE GLOBAL
# =============================================
current_color_idx = 4          # default WHITE
draw_color        = COLORS[COLOR_KEYS[current_color_idx]]
line_thickness    = 5
eraser_size       = 60
mode              = "IDLE"

smooth_x, smooth_y = None, None
prev_x,   prev_y   = None, None
mode_buffer        = deque(maxlen=MODE_STABLE_FRAMES)

#  MEDIAPIPE
# =============================================
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 1,
    model_complexity         = 1,
    min_detection_confidence = 0.75,
    min_tracking_confidence  = 0.60,
)

#  FUNGSI
# =============================================
def count_fingers(lm, handedness="Right"):
    tips   = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    count  = 0

    # Ibu jari
    if handedness == "Right":
        if lm[tips[0]].x < lm[joints[0]].x - 0.02:
            count += 1
    else:
        if lm[tips[0]].x > lm[joints[0]].x + 0.02:
            count += 1

    # 4 jari lain
    for i in range(1, 5):
        if lm[tips[i]].y < lm[joints[i]].y - 0.02:
            count += 1

    return count


def stabilize_mode(raw_mode):
    mode_buffer.append(raw_mode)
    return max(set(mode_buffer), key=mode_buffer.count)


def smooth_position(nx, ny):
    global smooth_x, smooth_y
    if smooth_x is None:
        smooth_x, smooth_y = nx, ny
    else:
        smooth_x = int(smooth_x * SMOOTH_FACTOR + nx * (1 - SMOOTH_FACTOR))
        smooth_y = int(smooth_y * SMOOTH_FACTOR + ny * (1 - SMOOTH_FACTOR))
    return smooth_x, smooth_y


def draw_ui(frame, mode, color_name, line_thickness, eraser_size, fps):
    w = frame.shape[1]
    bar_h = 60

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    mode_color = {
        "DRAW":  (0, 255, 80),
        "ERASE": (0, 80, 255),
        "IDLE":  (160, 160, 160),
    }.get(mode, (255, 255, 255))

    cv2.rectangle(frame, (8, 8), (155, bar_h - 8), (50, 50, 50), -1)
    cv2.rectangle(frame, (8, 8), (155, bar_h - 8), mode_color, 1)
    cv2.putText(frame, mode, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

    col_bgr = COLORS[color_name]
    cv2.rectangle(frame, (165, 8), (310, bar_h - 8), (50, 50, 50), -1)
    cv2.rectangle(frame, (165, 8), (310, bar_h - 8), col_bgr, 1)
    cv2.putText(frame, color_name, (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col_bgr, 2)

    cv2.rectangle(frame, (320, 8), (435, bar_h - 8), (50, 50, 50), -1)
    cv2.putText(frame, f"LINE:{line_thickness}", (328, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (445, 8), (590, bar_h - 8), (50, 50, 50), -1)
    cv2.putText(frame, f"ERASE:{eraser_size}", (453, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    fps_color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS:{int(fps)}", (600, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

    cv2.putText(frame, "ESC:Exit S:Save C:Clear | 1-5:Color +/-:Line []:Erase",
                (8, bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)

#  MAIN
# =============================================
def main():
    global draw_color, line_thickness, eraser_size, mode
    global prev_x, prev_y, smooth_x, smooth_y, current_color_idx

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # kurangi latency!

    ret, frame = cap.read()
    if not ret:
        print("Kamera tidak ditemukan!")
        return

    h, w   = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    save_counter = 0
    prev_time    = cv2.getTickCount()

    print("=" * 50)
    print("  AIR DRAWING SYSTEM v2.0 - SIAP!")
    print("  1 jari=DRAW | 2 jari=ERASE | kepal=IDLE")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            curr_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_time - prev_time)
            prev_time = curr_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            detected = False

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                    handedness = hand_info.classification[0].label

                    mp_drawing.draw_landmarks(
                        frame, hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    fingers  = count_fingers(hand_lm.landmark, handedness)
                    raw_mode = "DRAW" if fingers == 1 else "ERASE" if fingers == 2 else "IDLE"
                    mode     = stabilize_mode(raw_mode)

                    raw_x = int(hand_lm.landmark[8].x * w)
                    raw_y = int(hand_lm.landmark[8].y * h)
                    ix, iy = smooth_position(raw_x, raw_y)

                    tip_color = (0, 255, 80) if mode == "DRAW" else (0, 80, 255) if mode == "ERASE" else (200, 200, 200)
                    cv2.circle(frame, (ix, iy), 8,  tip_color,     -1)
                    cv2.circle(frame, (ix, iy), 11, (255,255,255),  1)

                    if iy > 80:
                        if mode == "DRAW":
                            if prev_x is not None:
                                dist  = np.hypot(ix - prev_x, iy - prev_y)
                                steps = max(int(dist / 2), 1)
                                for s in range(steps + 1):
                                    t  = s / steps
                                    mx = int(prev_x + t * (ix - prev_x))
                                    my = int(prev_y + t * (iy - prev_y))
                                    cv2.circle(canvas, (mx, my), line_thickness, draw_color, -1)
                            prev_x, prev_y = ix, iy

                        elif mode == "ERASE":
                            cv2.circle(canvas, (ix, iy), eraser_size, (0, 0, 0), -1)
                            cv2.circle(frame,  (ix, iy), eraser_size, (100, 100, 100), 2)
                            cv2.circle(frame,  (ix, iy), eraser_size, (220, 220, 220), 1)
                            prev_x, prev_y = None, None
                        else:
                            prev_x, prev_y = None, None
                    else:
                        prev_x, prev_y = None, None

                    detected = True
                    break

            if not detected:
                mode = stabilize_mode("IDLE")
                prev_x = prev_y = None
                smooth_x = smooth_y = None

            output = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
            color_name = COLOR_KEYS[current_color_idx]
            draw_ui(output, mode, color_name, line_thickness, eraser_size, fps)
            cv2.imshow("AIR - WRITING v2", output)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key in (ord('c'), ord('C')):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                print("Canvas dibersihkan")
            elif key in (ord('s'), ord('S')):
                save_counter += 1
                fn = f"drawing_{save_counter:03d}.png"
                cv2.imwrite(fn, canvas)
                print(f"Disimpan: {fn}")
            elif key in (ord('+'), ord('=')):
                line_thickness = min(line_thickness + 1, 40)
            elif key == ord('-'):
                line_thickness = max(line_thickness - 1, 1)
            elif key == ord(']'):
                eraser_size = min(eraser_size + 5, 200)
            elif key == ord('['):
                eraser_size = max(eraser_size - 5, 10)
            elif key == ord('1'):
                current_color_idx = 0; draw_color = COLORS[COLOR_KEYS[0]]
            elif key == ord('2'):
                current_color_idx = 1; draw_color = COLORS[COLOR_KEYS[1]]
            elif key == ord('3'):
                current_color_idx = 2; draw_color = COLORS[COLOR_KEYS[2]]
            elif key == ord('4'):
                current_color_idx = 3; draw_color = COLORS[COLOR_KEYS[3]]
            elif key == ord('5'):
                current_color_idx = 4; draw_color = COLORS[COLOR_KEYS[4]]

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("AIR DRAWING SYSTEM ditutup.")


if __name__ == "__main__":
    main()
