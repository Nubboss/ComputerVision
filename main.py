import argparse
import time
import cv2
from ultralytics import YOLO
import logging
from pandas import DataFrame, Timestamp
import pandas as pd

MODEL_NAME = "yolov8n.pt"

STATE_DEBOUNCE = 0.5               # минимальное время для подтверждения смены состояния (сек)
APPROACH_MIN_EMPTY_DURATION = 2.0  # если стол был пуст >= этого — появление считается подходом

df = DataFrame(columns=['event', 'time'])

parser = argparse.ArgumentParser(description='Welcome')
parser.add_argument(
        '--video',
        help='Укажите путь до видео',
        required=True
    )
my_namespace = parser.parse_args()
video_path = my_namespace.video  # путь к файлу

state = "unknown"
last_change_time = time.time()
empty_start_time = None

model = YOLO(MODEL_NAME, verbose=False)
logging.getLogger("ultralytics").setLevel(logging.ERROR)   # отключаем логи
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {video_path}")

# подготовка VideoWriter (ДОБАВЛЕНО)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None or fps != fps:  # проверка NaN
    fps = 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Прочитать первый кадр и показать для выбора ROI
ret, frame = cap.read()
if not ret or frame is None:
    writer.release()
    cap.release()
    raise RuntimeError("Не удалось прочитать первый кадр")

cv2.imshow("tmp", frame)
cv2.waitKey(1)
x, y, w, h = cv2.selectROI("tmp", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("tmp")

if w == 0 or h == 0:
    print("No ROI selected")
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

# Координаты ROI
roi_x1, roi_y1 = int(x), int(y)
roi_x2, roi_y2 = int(x + w), int(y + h)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Прогон через модель
    results = list(model(frame, imgsz=640))
    res = results[0]
    boxes = getattr(res, "boxes", [])

    people_in_roi = 0

    for box in boxes:
        # получить класс и координаты
        cls = int(box.cls[0])
        name = model.names.get(cls, str(cls))

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Проверка пересечения с ROI (простая AABB проверка)
        inter_x1 = max(x1, roi_x1)
        inter_y1 = max(y1, roi_y1)
        inter_x2 = min(x2, roi_x2)
        inter_y2 = min(y2, roi_y2)

        if name != "person":
            # можно рисовать детекции не-person если нужно, но не обязательно
            continue

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            people_in_roi += 1
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    now = time.time()
    if people_in_roi > 0:
        new_state = "occupied"
    else:
        new_state = "empty"

    # подтверждаем смену состояния, чтобы не реагировать на шум
    if new_state != state and (now - last_change_time) >= STATE_DEBOUNCE:
        # переход состояния
        if new_state == "empty":
            df.loc[len(df)] = ['table_empty', Timestamp.now()]
            empty_start_time = now
        else:  # new_state == "occupied"
            # проверяем, был ли длительный период пустоты -> approach
            approached = False
            if empty_start_time is not None and (now - empty_start_time) >= APPROACH_MIN_EMPTY_DURATION:
                df.loc[len(df)] = ['approach', Timestamp.now()]
                approached = True
            df.loc[len(df)] = ['table_occupied', Timestamp.now()]
            empty_start_time = None

        state = new_state
        last_change_time = now

    # отрисовка ROI: красный если занято, зелёный если пусто
    roi_color = (0, 0, 255) if state == "occupied" else (0, 255, 0)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)

    # показать и записать кадр (запись ДО показа или после — не важно)
    writer.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# список времён, когда было table_empty
empty_times = df.loc[df['event'] == 'table_empty', 'time'].reset_index(drop=True)

def mean_delta_seconds(target):
    deltas = []
    for t in empty_times:
        future = df[(df['time'] > t) & (df['event'] == target)]
        if not future.empty:
            deltas.append((future['time'].iloc[0] - t).total_seconds())
    return None if not deltas else sum(deltas) / len(deltas)

mean_empty_to_occupied = mean_delta_seconds('table_occupied')
mean_empty_to_approach = mean_delta_seconds('approach')

print('среднее пусто ->занято  (s):', mean_empty_to_occupied)
print('среднее  пусто ->подошли  (s):', mean_empty_to_approach)

# сохранить простой отчет в файл (ДОБАВЛЕНО)
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write(f"среднее пусто ->занято  (s):{mean_empty_to_occupied}\n")
    f.write(f"среднее  пусто ->подошли  (s): {mean_empty_to_approach}\n")
    f.write("\nEvents:\n")
    f.write(df.to_csv(index=False))

# корректное освобождение ресурсов (writer уже использовали)
writer.release()
cap.release()
cv2.destroyAllWindows()
