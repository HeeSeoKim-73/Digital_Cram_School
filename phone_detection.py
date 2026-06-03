import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime, time
from ultralytics import YOLO

phone_model = YOLO("C:/Users/k8254/phone_detection/runs/detect/phone_detection_model-3/weights/best.pt")
person_model = YOLO("yolov8n.pt")

PHONE_CLASS_NAME = "phone"
PERSON_CLASS_NAME = "person"

CONFIDENCE_LIMIT = 0.5
PERSON_CONFIDENCE_LIMIT = 0.5
ABSENCE_LIMIT_SECONDS = 10

SCHEDULE_IMAGE_PATH = "schedule.jpg"
TIMER_IMAGE_PATH = "timer.jpg"
TODO_IMAGE_PATH = "todo.jpg"

DISPLAY_HEIGHT = 540
WEBCAM_WIDTH = 720
TIMER_SCALE = 1 / 6

TEXT_SMALL = 0.3
TEXT_TINY = 0.25
TEXT_THICKNESS = 1

timer_total_seconds = 0
timer_remaining_seconds = 0
timer_running = False
timer_last_update = None

absence_start_time = None

todo_items = []

left_panel_width = 0
timer_panel_height = 0
todo_x = 0
todo_y = 0
todo_width = 0
todo_height = 0

STUDY_PERIODS = [
    (time(8, 40), time(10, 0), "08:40 - 10:00"),
    (time(10, 20), time(12, 0), "10:20 - 12:00"),
    (time(13, 10), time(14, 30), "13:10 - 14:30"),
    (time(14, 40), time(16, 10), "14:40 - 16:10"),
    (time(16, 20), time(17, 40), "16:20 - 17:40"),
    (time(18, 40), time(20, 20), "18:40 - 20:20"),
    (time(20, 30), time(22, 0), "20:30 - 22:00")
]

BREAK_PERIODS = [
    (time(10, 0), time(10, 20), "BREAK"),
    (time(12, 0), time(13, 10), "LUNCH"),
    (time(14, 30), time(14, 40), "BREAK"),
    (time(16, 10), time(16, 20), "BREAK"),
    (time(17, 40), time(18, 40), "DINNER"),
    (time(20, 20), time(20, 30), "BREAK")
]


def is_in_period(current_time, periods):
    for start, end, label in periods:
        if start <= current_time < end:
            return True, label, start, end
    return False, None, None, None


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_remaining_time(delta):
    return format_seconds(delta.total_seconds())


def get_next_break_remaining(now, study_end_time):
    study_end_datetime = datetime.combine(now.date(), study_end_time)
    return format_remaining_time(study_end_datetime - now)


def get_next_session_text(now):
    current_time = now.time()
    all_periods = STUDY_PERIODS + BREAK_PERIODS
    sorted_periods = sorted(all_periods, key=lambda x: x[0])

    for start, end, label in sorted_periods:
        if current_time < start:
            return f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"

    return "END OF DAY"


def resize_keep_ratio_by_height(image, target_height):
    h, w = image.shape[:2]
    ratio = target_height / h
    return cv2.resize(image, (int(w * ratio), target_height))


def resize_keep_ratio_by_width(image, target_width):
    h, w = image.shape[:2]
    ratio = target_width / w
    return cv2.resize(image, (target_width, int(h * ratio)))


def resize_timer_by_scale(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def resize_webcam(frame):
    return cv2.resize(frame, (WEBCAM_WIDTH, DISPLAY_HEIGHT))


def draw_border(frame, color, thickness=18):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


def draw_warning_overlay(frame, message, color):
    h, w, _ = frame.shape
    draw_border(frame, color)

    center_x = w // 2
    center_y = h // 2
    triangle_size = 90

    triangle_points = np.array([
        (center_x, center_y - triangle_size),
        (center_x - triangle_size, center_y + triangle_size),
        (center_x + triangle_size, center_y + triangle_size)
    ], np.int32)

    cv2.polylines(frame, [triangle_points], True, color, 8)
    cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 45), color, 12)
    cv2.circle(frame, (center_x, center_y + 80), 8, color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)

    text_x = center_x - text_size[0] // 2
    text_y = center_y + 170 + text_size[1] // 2

    cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)


def draw_status_panel(frame, now, study_mode, timer_mode, current_start, current_end, absence_seconds):
    if timer_mode:
        color = (255, 0, 255)
        status = "MOCK EXAM MODE"
        current_session = "TIMER ACTIVE"
        time_title = "TIMER LEFT"
        time_value = format_seconds(timer_remaining_seconds)

    elif study_mode:
        color = (0, 0, 255)
        status = "STUDY TIME"
        current_session = f"{current_start.strftime('%H:%M')} - {current_end.strftime('%H:%M')}"
        time_title = "NEXT BREAK"
        time_value = get_next_break_remaining(now, current_end)

    else:
        color = (0, 255, 0)
        status = "BREAK TIME"

        if current_start is not None and current_end is not None:
            current_session = f"{current_start.strftime('%H:%M')} - {current_end.strftime('%H:%M')}"
        else:
            current_session = "OUT OF SCHEDULE"

        time_title = "NEXT SESSION"
        time_value = get_next_session_text(now)

    panel_x = 20

    cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (panel_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SMALL, color, TEXT_THICKNESS)

    cv2.putText(frame, f"STATUS : {status}", (panel_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SMALL, color, TEXT_THICKNESS)

    cv2.putText(frame, "CURRENT SESSION", (panel_x, 85),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_TINY, (255, 255, 255), TEXT_THICKNESS)

    cv2.putText(frame, current_session, (panel_x, 105),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SMALL, color, TEXT_THICKNESS)

    cv2.putText(frame, time_title, (panel_x, 140),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_TINY, (255, 255, 255), TEXT_THICKNESS)

    cv2.putText(frame, time_value, (panel_x, 160),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SMALL, color, TEXT_THICKNESS)

    if study_mode or timer_mode:
        cv2.putText(frame, f"ABSENCE : {absence_seconds:.1f}s", (panel_x, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, TEXT_SMALL, color, TEXT_THICKNESS)


def draw_timer_text_center(image, text, center_x_ratio, center_y_ratio, font_scale, color, thickness):
    h, w = image.shape[:2]
    center_x = int(w * center_x_ratio)
    center_y = int(h * center_y_ratio)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2

    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)


def draw_todo_text(todo_area):
    start_x = 35
    start_y = 75
    line_gap = 32

    for i, item in enumerate(todo_items):
        y = start_y + i * line_gap

        if y > todo_area.shape[0] - 30:
            break

        checkbox_x = 18
        checkbox_y = y - 15

        cv2.rectangle(
            todo_area,
            (checkbox_x, checkbox_y),
            (checkbox_x + 16, checkbox_y + 16),
            (0, 0, 0),
            1
        )

        if item["done"]:
            cv2.line(
                todo_area,
                (checkbox_x + 3, checkbox_y + 8),
                (checkbox_x + 7, checkbox_y + 13),
                (0, 0, 255),
                2
            )
            cv2.line(
                todo_area,
                (checkbox_x + 7, checkbox_y + 13),
                (checkbox_x + 14, checkbox_y + 3),
                (0, 0, 255),
                2
            )

        text_x = start_x
        text_y = y

        cv2.putText(
            todo_area,
            item["text"],
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1
        )

        if item["done"]:
            cv2.line(
                todo_area,
                (text_x, text_y - 6),
                (todo_area.shape[1] - 25, text_y - 6),
                (0, 0, 255),
                2
            )


def make_left_panel(timer_img, todo_img):
    global left_panel_width, timer_panel_height, todo_x, todo_y, todo_width, todo_height

    timer_resized = resize_timer_by_scale(timer_img, TIMER_SCALE)
    timer_h, timer_w = timer_resized.shape[:2]

    todo_resized = resize_keep_ratio_by_width(todo_img, timer_w)
    todo_h, todo_w = todo_resized.shape[:2]

    left_panel_width = timer_w
    timer_panel_height = timer_h

    panel = np.ones((DISPLAY_HEIGHT, left_panel_width, 3), dtype=np.uint8) * 255

    panel[0:timer_h, 0:timer_w] = timer_resized

    draw_timer_text_center(
        panel[0:timer_h, 0:timer_w],
        format_seconds(timer_remaining_seconds),
        0.50,
        0.27,
        0.7,
        (0, 0, 0),
        2
    )

    todo_x = 0
    todo_y = timer_h + 10
    todo_width = todo_w
    todo_height = min(todo_h, DISPLAY_HEIGHT - todo_y)

    visible_todo = todo_resized[0:todo_height, 0:todo_width].copy()
    draw_todo_text(visible_todo)

    panel[todo_y:todo_y + todo_height, todo_x:todo_x + todo_width] = visible_todo

    return panel


def make_combined_screen(timer_img, todo_img, frame, schedule_img):
    left_panel = make_left_panel(timer_img, todo_img)
    frame_resized = resize_webcam(frame)
    schedule_resized = resize_keep_ratio_by_height(schedule_img, DISPLAY_HEIGHT)

    return cv2.hconcat([left_panel, frame_resized, schedule_resized])


def update_timer(now):
    global timer_remaining_seconds, timer_running, timer_last_update

    if not timer_running:
        timer_last_update = now
        return

    if timer_last_update is None:
        timer_last_update = now
        return

    elapsed = (now - timer_last_update).total_seconds()
    timer_last_update = now
    timer_remaining_seconds -= elapsed

    if timer_remaining_seconds <= 0:
        timer_remaining_seconds = 0
        timer_running = False


def add_todo_text():
    root = tk.Tk()
    root.withdraw()

    new_text = simpledialog.askstring("To Do List", "Enter study task:")

    root.destroy()

    if new_text is not None and new_text.strip():
        todo_items.append({
            "text": new_text.strip(),
            "done": False
        })


def on_mouse(event, x, y, flags, param):
    global timer_total_seconds, timer_remaining_seconds, timer_running, timer_last_update

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if x < 0 or x > left_panel_width:
        return

    if 0 <= y <= timer_panel_height:
        local_x = x / left_panel_width
        local_y = y / timer_panel_height

        minute_button = (0.29, 0.73, 0.12)
        reset_button = (0.50, 0.73, 0.12)
        start_button = (0.72, 0.73, 0.12)

        def clicked_button(button):
            bx, by, radius = button
            distance = ((local_x - bx) ** 2 + (local_y - by) ** 2) ** 0.5
            return distance <= radius

        if clicked_button(minute_button):
            if not timer_running:
                timer_total_seconds += 60
                timer_remaining_seconds += 60
                print("+1 minute")

        elif clicked_button(reset_button):
            timer_running = False
            timer_total_seconds = 0
            timer_remaining_seconds = 0
            timer_last_update = None
            print("Timer reset")

        elif clicked_button(start_button):
            if timer_remaining_seconds > 0:
                timer_running = not timer_running
                timer_last_update = datetime.now()
                print("Timer start/stop")

    elif todo_y <= y <= todo_y + todo_height:
        local_y = y - todo_y
        local_x = x - todo_x

        start_y = 75
        line_gap = 32

        index = (local_y - start_y + 15) // line_gap

        if 0 <= index < len(todo_items):
            item_y = start_y + index * line_gap
            checkbox_x1 = 18
            checkbox_y1 = item_y - 15
            checkbox_x2 = checkbox_x1 + 16
            checkbox_y2 = checkbox_y1 + 16

            if checkbox_x1 <= local_x <= checkbox_x2 and checkbox_y1 <= local_y <= checkbox_y2:
                todo_items[index]["done"] = not todo_items[index]["done"]
                return

        add_todo_text()


schedule_img = cv2.imread(SCHEDULE_IMAGE_PATH)
timer_img = cv2.imread(TIMER_IMAGE_PATH)
todo_img = cv2.imread(TODO_IMAGE_PATH)

if schedule_img is None:
    print("Cannot load schedule.jpg.")
    exit()

if timer_img is None:
    print("Cannot load timer.jpg.")
    exit()

if todo_img is None:
    print("Cannot load todo.jpg.")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

window_name = "Computer Academy Phone Control System"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, on_mouse)

print("Computer Academy System Started")
print("Timer left button  : +1 minute")
print("Timer middle button: reset")
print("Timer right button : start / stop")
print("Click memo area to add study task")
print("Press q or Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    now = datetime.now()
    update_timer(now)

    current_time = now.time()
    timer_mode = timer_remaining_seconds > 0

    study_mode, study_label, study_start, study_end = is_in_period(current_time, STUDY_PERIODS)
    break_mode, break_label, break_start, break_end = is_in_period(current_time, BREAK_PERIODS)

    if study_mode:
        current_start = study_start
        current_end = study_end
    elif break_mode:
        current_start = break_start
        current_end = break_end
    else:
        current_start = None
        current_end = None

    restriction_mode = study_mode or timer_mode

    phone_results = phone_model(frame, verbose=False)
    phone_detected = False

    for result in phone_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = phone_model.names[cls_id]

            if class_name == PHONE_CLASS_NAME and confidence >= CONFIDENCE_LIMIT:
                phone_detected = True

    person_results = person_model(frame, verbose=False)
    person_detected = False

    for result in person_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = person_model.names[cls_id]

            if class_name == PERSON_CLASS_NAME and confidence >= PERSON_CONFIDENCE_LIMIT:
                person_detected = True

    absence_seconds = 0

    if restriction_mode:
        if not person_detected:
            if absence_start_time is None:
                absence_start_time = now

            absence_seconds = (now - absence_start_time).total_seconds()
        else:
            absence_start_time = None
            absence_seconds = 0
    else:
        absence_start_time = None
        absence_seconds = 0

    if restriction_mode and phone_detected:
        draw_warning_overlay(frame, "PHONE DETECTED", (0, 0, 255))

    elif restriction_mode and absence_seconds >= ABSENCE_LIMIT_SECONDS:
        draw_warning_overlay(frame, "SEAT DOWN", (255, 0, 0))

    elif not restriction_mode:
        draw_border(frame, (0, 255, 0))

    draw_status_panel(
        frame,
        now,
        study_mode,
        timer_mode,
        current_start,
        current_end,
        absence_seconds
    )

    combined_screen = make_combined_screen(timer_img, todo_img, frame, schedule_img)

    cv2.imshow(window_name, combined_screen)

    key = cv2.waitKey(10) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()