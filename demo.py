import face_recognition
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import sys
from dotenv import load_dotenv

import smtplib
from email.message import EmailMessage

# Send email
load_dotenv()
from_email = os.getenv("EMAIL_USER")
from_password = os.getenv("EMAIL_PASS")

def send_email_with_attachment(to_email, subject, body, attachment_path):
    from_email = os.getenv("EMAIL_USER")
    from_password = os.getenv("EMAIL_PASS")

    msg = EmailMessage()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content(body)

    # Read the image file and add as attachment
    with open(attachment_path, 'rb') as img_file:
        img_data = img_file.read()
        img_name = os.path.basename(attachment_path)
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=img_name)

    # Connect to SMTP server and send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(from_email, from_password)
        smtp.send_message(msg)

    print(f"📧 Email sent with attachment {img_name}")


#Format names (first last)
def format_name(name):
    result = name[0] 
    for char in name[1:]:
        if char.isupper():
            result += ' ' + char
        else:
            result += char
    return result

# Set Paths
path = "ImagesAttendance"
encodings_file = "encodings.pkl"
attendance_file = "Attendance.csv"
unknown_faces_path = "UnknownFaces"
os.makedirs(unknown_faces_path, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.6

if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name, Date, Time\n")

#Load Images
images = []
class_names = []

if os.path.exists(path):
    my_list = os.listdir(path)
else:
    my_list = []

for cl in my_list:
    cur_img = cv2.imread(f'{path}/{cl}')
    if cur_img is not None:
        images.append(cur_img)
        class_names.append(os.path.splitext(cl)[0])
    else:
        print(f"⚠️ Skipping file {cl} — not a valid image")

#Encode
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = face_recognition.face_encodings(img)
        if face:
            encode_list.append(face[0])
        else:
            print("⚠️ No face found in one image — skipping.")
    return encode_list

# Save and Load
def save_encodings(encodings, names):
    with open(encodings_file, "wb") as f:
            pickle.dump((encodings, names), f)

def load_encodings():
    with open(encodings_file, "rb") as f:
        return pickle.load(f)

#Write to CSV
def mark_attendance(name, face_img=None):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')

    # Format name for display
    display_name = format_name(name) if name != "Unknown" else "Unknown"

    # Load existing CSV entries
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            entries = [line.strip().split(',') for line in f.readlines()]
    else:
        entries = []

    # For known people, log once per day
    if name != "Unknown":
        already_logged = any(e[0] == display_name and e[1] == date_string for e in entries)
        if not already_logged:
            with open(attendance_file, 'a') as f:
                f.writelines(f'\n{display_name},{date_string},{time_string}')
            print(f"✅ Attendance logged: {display_name}")

    # For unknown faces, log every time (or could add a cooldown)
    else:
        # Optional: avoid spamming CSV with multiple frames of same unknown
        last_unknown_time = None
        if entries:
            last_entry = entries[-1]
            if last_entry[0] == "Unknown":
                last_unknown_time = datetime.strptime(last_entry[1] + ' ' + last_entry[2], '%Y-%m-%d %H:%M:%S')
        if not last_unknown_time or (now - last_unknown_time).seconds > 30:
            # Save snapshot
            if face_img is not None:
                filename = f"{unknown_faces_path}/Unknown_{date_string}_{time_string.replace(':','-')}.jpg"
                cv2.imwrite(filename, face_img)
                print(f"📷 Saved unknown face snapshot: {filename}")

                # Send email alert
                send_email_with_attachment(
                    to_email="mbala@umich.edu",
                    subject="Unknown Face Detected",
                    body=f"An unknown face was detected at {date_string} {time_string}. See the attached image.",
                    attachment_path=filename
                )

            # Log in CSV
            with open(attendance_file, 'a') as f:
                f.writelines(f'\nUnknown,{date_string},{time_string}')
            print(f"⚠️ Unknown logged at {time_string}")

# Load/Compute Encodings
if os.path.exists(encodings_file):
    encodeListKnown, classNames = load_encodings()
    print("✅ Loaded encodings from file")
else:
    encodeListKnown = find_encodings(images)
    save_encodings(encodeListKnown, class_names)
    print("✅ Encodings computed and saved")

print("Encoding Complete")

#Video
video_path = "ImagesAttendance/test.mp4"  # path to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
out = cv2.VideoWriter('output_demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

unknown_in_frame = False

# --- Parameters ---
CONFIDENCE_THRESHOLD = 0.6
UNKNOWN_FRAMES_THRESHOLD = 5  # Number of consecutive frames to confirm unknown

# --- Track last seen faces ---
tracked_faces = []  # list of dicts: { 'name': str, 'bbox': (y1,x2,y2,x1), 'frames_seen': int, 'frames_unknown': int }

# --- Video processing ---
while True:
    success, img = cap.read()
    if not success:
        print("✅ Video processing complete")
        break

    imgS = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodedCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # For this frame
    current_tracked = []

    for encodedFace, faceLoc in zip(encodedCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodedFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodedFace)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = [v*4 for v in faceLoc]  # scale back to original size

        if faceDis[matchIndex] < CONFIDENCE_THRESHOLD and matches[matchIndex]:
            name = classNames[matchIndex]
            color = (0, 255, 0)
            frames_unknown = 0  # reset unknown counter
            mark_attendance(name)
        else:
            name = "Unknown"
            color = (0, 0, 255)
            frames_unknown = 1  # start counting unknown

        # Check if this face was tracked last frame
        matched_face = None
        for f in tracked_faces:
            # simple overlap check
            x_overlap = max(0, min(f['bbox'][1], x2) - max(f['bbox'][3], x1))
            y_overlap = max(0, min(f['bbox'][2], y2) - max(f['bbox'][0], y1))
            if x_overlap * y_overlap > 0.25*(x2-x1)*(y2-y1):  # >25% overlap
                matched_face = f
                break

        if matched_face:
            if name == "Unknown":
                matched_face['frames_unknown'] += 1
                if matched_face['frames_unknown'] >= UNKNOWN_FRAMES_THRESHOLD and not matched_face.get('flagged'):
                    # Only mark unknown if seen unknown for enough frames
                    y1_clip, y2_clip = max(0,y1), min(img.shape[0], y2)
                    x1_clip, x2_clip = max(0,x1), min(img.shape[1], x2)
                    face_crop = img[y1_clip:y2_clip, x1_clip:x2_clip]
                    mark_attendance("Unknown", face_crop)
                    matched_face['flagged'] = True
            else:
                matched_face['frames_unknown'] = 0
                matched_face['flagged'] = False
            matched_face['bbox'] = (y1, x2, y2, x1)
            matched_face['name'] = name
            current_tracked.append(matched_face)
        else:
            current_tracked.append({'name': name, 'bbox': (y1, x2, y2, x1),
                                    'frames_seen': 1, 'frames_unknown': frames_unknown, 'flagged': False})

        # Draw box and name
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), color, cv2.FILLED)
        display_name = format_name(name) if name != "Unknown" else "Unknown"
        cv2.putText(img, display_name, (x1+6, y2-6),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    tracked_faces = current_tracked  # update tracked faces
    out.write(img)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()