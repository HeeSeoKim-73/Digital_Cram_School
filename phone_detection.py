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

DISPLAY_HEIGHT = 540
LEFT_WIDTH = 330
WEBCAM_WIDTH = 720
RIGHT_WIDTH = 360

DARK_BG = (18, 22, 28)
CARD_BG = (30, 36, 46)
CARD_BORDER = (70, 85, 105)
WHITE = (245, 245, 245)
GRAY = (160, 165, 175)
RED = (0, 0, 255)
GREEN = (0, 220, 120)
BLUE = (255, 120, 0)
PURPLE = (255, 0, 255)
BLACK = (0, 0, 0)

timer_total_seconds = 0
timer_remaining_seconds = 0
timer_running = False
timer_last_update = None

absence_start_time = None
todo_items = []

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

SCHEDULE_ROWS = [
    ("08:40 - 10:00", "STUDY TIME"),
    ("10:00 - 10:20", "BREAK"),
    ("10:20 - 12:00", "STUDY TIME"),
    ("12:00 - 13:10", "LUNCH"),
    ("13:10 - 14:30", "STUDY TIME"),
    ("14:30 - 14:40", "BREAK"),
    ("14:40 - 16:10", "STUDY TIME"),
    ("16:10 - 16:20", "BREAK"),
    ("16:20 - 17:40", "STUDY TIME"),
    ("17:40 - 18:40", "DINNER"),
    ("18:40 - 20:20", "STUDY TIME"),
    ("20:20 - 20:30", "BREAK"),
    ("20:30 - 22:00", "STUDY TIME")
]


def is_in_period(current_time, periods):
    for start, end, label in periods:
        if start <= current_time < end:
            return True, label, start, end
    return False, None, None, None


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_next_break_remaining(now, study_end_time):
    end_dt = datetime.combine(now.date(), study_end_time)
    return format_seconds((end_dt - now).total_seconds())


def get_next_session_text(now):
    current_time = now.time()
    all_periods = STUDY_PERIODS + BREAK_PERIODS
    sorted_periods = sorted(all_periods, key=lambda x: x[0])

    for start, end, label in sorted_periods:
        if current_time < start:
            return f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"

    return "END OF DAY"


def resize_webcam(frame):
    return cv2.resize(frame, (WEBCAM_WIDTH, DISPLAY_HEIGHT))


def draw_card(img, x1, y1, x2, y2, title):
    cv2.rectangle(img, (x1, y1), (x2, y2), CARD_BG, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), CARD_BORDER, 1)

    cv2.putText(
        img,
        title,
        (x1 + 18, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1
    )


def draw_badge(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    w = text_size[0] + 22
    h = 28

    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

    cv2.putText(
        img,
        text,
        (x + 11, y + 19),
        font,
        font_scale,
        WHITE,
        thickness
    )


def draw_border(frame, color, thickness=8):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


def draw_warning_overlay(frame, message, color):
    h, w = frame.shape[:2]

    draw_border(frame, color, 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
    frame[:] = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    center_x = w // 2
    center_y = h // 2

    triangle_size = 65

    points = np.array([
        (center_x, center_y - triangle_size),
        (center_x - triangle_size, center_y + triangle_size),
        (center_x + triangle_size, center_y + triangle_size)
    ], np.int32)

    cv2.polylines(frame, [points], True, color, 7)
    cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 35), color, 10)
    cv2.circle(frame, (center_x, center_y + 65), 7, color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.05
    thickness = 3

    text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
    text_x = center_x - text_size[0] // 2
    text_y = center_y + 145

    cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)


def draw_status_panel(left_panel, now, study_mode, timer_mode, current_start, current_end):
    x1, y1, x2, y2 = 12, 12, LEFT_WIDTH - 12, 150

    draw_card(left_panel, x1, y1, x2, y2, "ACADEMY STATUS")

    if timer_mode:
        color = PURPLE
        status = "MOCK EXAM"
        session_text = "TIMER ACTIVE"
    elif study_mode:
        color = RED
        status = "STUDY TIME"
        session_text = f"{current_start.strftime('%H:%M')} - {current_end.strftime('%H:%M')}"
    else:
        color = GREEN
        status = "BREAK TIME"

        if current_start is not None and current_end is not None:
            session_text = f"{current_start.strftime('%H:%M')} - {current_end.strftime('%H:%M')}"
        else:
            session_text = "OUT OF SCHEDULE"

    draw_badge(left_panel, status, x1 + 18, y1 + 45, color)

    cv2.putText(
        left_panel,
        now.strftime("%Y-%m-%d  %H:%M:%S"),
        (x1 + 18, y1 + 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        GRAY,
        1
    )

    cv2.putText(
        left_panel,
        "CURRENT SESSION",
        (x1 + 18, y1 + 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        GRAY,
        1
    )

    cv2.putText(
        left_panel,
        session_text,
        (x1 + 18, y1 + 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1
    )


def draw_monitoring_panel(left_panel, now, study_mode, timer_mode, current_end, absence_seconds):
    x1, y1, x2, y2 = 12, 160, LEFT_WIDTH - 12, 245

    draw_card(left_panel, x1, y1, x2, y2, "MONITORING")

    if timer_mode:
        time_title = "TIMER LEFT"
        time_value = format_seconds(timer_remaining_seconds)
        color = PURPLE
    elif study_mode:
        time_title = "NEXT BREAK"
        time_value = get_next_break_remaining(now, current_end)
        color = RED
    else:
        time_title = "NEXT SESSION"
        time_value = get_next_session_text(now)
        color = GREEN

    cv2.putText(
        left_panel,
        time_title,
        (x1 + 18, y1 + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        GRAY,
        1
    )

    cv2.putText(
        left_panel,
        time_value,
        (x1 + 18, y1 + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2
    )

    cv2.putText(
        left_panel,
        "ABSENCE",
        (x1 + 185, y1 + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        GRAY,
        1
    )

    cv2.putText(
        left_panel,
        f"{absence_seconds:.1f}s",
        (x1 + 185, y1 + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        BLUE,
        2
    )


def draw_timer_panel(left_panel):
    x1, y1, x2, y2 = 12, 255, LEFT_WIDTH - 12, 360

    draw_card(left_panel, x1, y1, x2, y2, "MOCK EXAM TIMER")

    cx = (x1 + x2) // 2

    text = format_seconds(timer_remaining_seconds)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)

    cv2.putText(
        left_panel,
        text,
        (cx - text_size[0] // 2, y1 + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        WHITE,
        2
    )

    button_y = y1 + 82
    button_radius = 13

    buttons = [
        ("+1", x1 + 65, RED),
        ("RESET", cx, GRAY),
        ("START", x2 - 65, GREEN)
    ]

    for text, bx, color in buttons:
        cv2.circle(left_panel, (bx, button_y), button_radius, color, -1)
        cv2.circle(left_panel, (bx, button_y), button_radius, CARD_BORDER, 1)

        font_scale = 0.22 if text != "RESET" else 0.18
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        cv2.putText(
            left_panel,
            text,
            (bx - text_size[0] // 2, button_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            WHITE,
            1
        )


def draw_todo_panel(left_panel):
    x1, y1, x2, y2 = 12, 370, LEFT_WIDTH - 12, DISPLAY_HEIGHT - 12

    draw_card(left_panel, x1, y1, x2, y2, "TO DO LIST")

    start_y = y1 + 55
    line_gap = 27

    for i, item in enumerate(todo_items):
        y = start_y + i * line_gap

        if y > y2 - 38:
            break

        checkbox_x = x1 + 18
        checkbox_y = y - 13

        cv2.rectangle(left_panel, (checkbox_x, checkbox_y), (checkbox_x + 14, checkbox_y + 14), GRAY, 1)

        if item["done"]:
            cv2.line(left_panel, (checkbox_x + 3, checkbox_y + 7), (checkbox_x + 6, checkbox_y + 11), RED, 2)
            cv2.line(left_panel, (checkbox_x + 6, checkbox_y + 11), (checkbox_x + 13, checkbox_y + 3), RED, 2)

        text_x = checkbox_x + 25
        text_y = y

        cv2.putText(
            left_panel,
            item["text"],
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            WHITE,
            1
        )

        if item["done"]:
            cv2.line(left_panel, (text_x, text_y - 5), (x2 - 20, text_y - 5), RED, 2)

    cv2.rectangle(left_panel, (x1 + 18, y2 - 35), (x2 - 18, y2 - 10), (55, 65, 80), -1)
    cv2.putText(
        left_panel,
        "+ ADD TASK",
        (x1 + 30, y2 - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        GRAY,
        1
    )


def draw_schedule_panel(right_panel, now):
    x1, y1, x2, y2 = 12, 12, RIGHT_WIDTH - 12, DISPLAY_HEIGHT - 12

    draw_card(right_panel, x1, y1, x2, y2, "TODAY'S STUDY SCHEDULE")

    current_time = now.time()

    table_x1 = x1 + 18
    table_y1 = y1 + 55
    table_x2 = x2 - 18
    row_h = 33

    for i, (period, label) in enumerate(SCHEDULE_ROWS):
        y = table_y1 + i * row_h

        start_text, end_text = period.split(" - ")
        start_h, start_m = map(int, start_text.split(":"))
        end_h, end_m = map(int, end_text.split(":"))

        start_t = time(start_h, start_m)
        end_t = time(end_h, end_m)

        active = start_t <= current_time < end_t
        row_color = (60, 25, 35) if active else CARD_BG

        cv2.rectangle(right_panel, (table_x1, y), (table_x2, y + row_h), row_color, -1)
        cv2.rectangle(right_panel, (table_x1, y), (table_x2, y + row_h), CARD_BORDER, 1)

        if active:
            cv2.fillPoly(
                right_panel,
                [np.array([
                    (table_x1 - 12, y + row_h // 2),
                    (table_x1 - 3, y + 8),
                    (table_x1 - 3, y + row_h - 8)
                ], np.int32)],
                RED
            )

        cv2.putText(
            right_panel,
            period,
            (table_x1 + 18, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE,
            1
        )

        cv2.putText(
            right_panel,
            label,
            (table_x1 + 180, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE if label == "STUDY TIME" else GRAY,
            1
        )


def make_left_panel(now, study_mode, timer_mode, current_start, current_end, absence_seconds):
    left_panel = np.zeros((DISPLAY_HEIGHT, LEFT_WIDTH, 3), dtype=np.uint8)
    left_panel[:] = DARK_BG

    draw_status_panel(left_panel, now, study_mode, timer_mode, current_start, current_end)
    draw_monitoring_panel(left_panel, now, study_mode, timer_mode, current_end, absence_seconds)
    draw_timer_panel(left_panel)
    draw_todo_panel(left_panel)

    return left_panel


def make_right_panel(now):
    right_panel = np.zeros((DISPLAY_HEIGHT, RIGHT_WIDTH, 3), dtype=np.uint8)
    right_panel[:] = DARK_BG

    draw_schedule_panel(right_panel, now)

    return right_panel


def make_combined_screen(left_panel, frame, right_panel):
    frame_resized = resize_webcam(frame)

    separator = np.zeros((DISPLAY_HEIGHT, 8, 3), dtype=np.uint8)
    separator[:] = DARK_BG

    return cv2.hconcat([left_panel, separator, frame_resized, separator, right_panel])


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

    if x > LEFT_WIDTH:
        return

    timer_y1 = 255
    timer_y2 = 360

    if timer_y1 <= y <= timer_y2:
        if 52 <= x <= 78:
            if not timer_running:
                timer_total_seconds += 60
                timer_remaining_seconds += 60

        elif 152 <= x <= 178:
            timer_running = False
            timer_total_seconds = 0
            timer_remaining_seconds = 0
            timer_last_update = None

        elif 252 <= x <= 278:
            if timer_remaining_seconds > 0:
                timer_running = not timer_running
                timer_last_update = datetime.now()

    todo_y1 = 370
    todo_y2 = DISPLAY_HEIGHT - 12

    if todo_y1 <= y <= todo_y2:
        add_y1 = todo_y2 - 35
        add_y2 = todo_y2 - 10

        if add_y1 <= y <= add_y2:
            add_todo_text()
            return

        start_y = todo_y1 + 55
        line_gap = 27

        index = (y - start_y + 13) // line_gap

        if 0 <= index < len(todo_items):
            checkbox_x1 = 30
            checkbox_x2 = 44

            item_y = start_y + index * line_gap
            checkbox_y1 = item_y - 13
            checkbox_y2 = checkbox_y1 + 14

            if checkbox_x1 <= x <= checkbox_x2 and checkbox_y1 <= y <= checkbox_y2:
                todo_items[index]["done"] = not todo_items[index]["done"]


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

window_name = "Computer Academy Phone Control System"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, on_mouse)

print("Computer Academy System Started")
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
        draw_warning_overlay(frame, "PHONE DETECTED", RED)

    elif restriction_mode and absence_seconds >= ABSENCE_LIMIT_SECONDS:
        draw_warning_overlay(frame, "SEAT DOWN", BLUE)

    elif not restriction_mode:
        draw_border(frame, GREEN, 8)

    left_panel = make_left_panel(
        now,
        study_mode,
        timer_mode,
        current_start,
        current_end,
        absence_seconds
    )

    right_panel = make_right_panel(now)

    combined_screen = make_combined_screen(left_panel, frame, right_panel)

    cv2.imshow(window_name, combined_screen)

    key = cv2.waitKey(10) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()