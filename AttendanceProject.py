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
    today = datetime.now().strftime('%Y-%m-%d')
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')

    # Known people: log once per day
    if name != "Unknown":
        display_name = format_name(name)

        with open(attendance_file, 'r+') as f:
            DataList = f.readlines()
            for line in DataList:
                entry = line.strip().split(',')
                if entry[0] == display_name and entry[1] == today:
                    return
            f.writelines(f'\n{display_name},{date_string},{time_string}')
    
    else:
            with open(attendance_file, 'a') as f:
                f.writelines(f'\nUnknown,{date_string},{time_string}')
            
            # Unknown-- Save snapshot and send email

            if face_img is not None:
                filename = f"{unknown_faces_path}/Unknown_{date_string}_{time_string.replace(':', '-')}.jpg"
                cv2.imwrite(filename, face_img)
                print(f"📷 Saved unknown face snapshot: {filename}")

                send_email_with_attachment(
                    to_email="mbala@umich.edu",
                    subject="Unknown Face Detected",
                    body=f"An unknown face was detected at {date_string} {time_string}. See the attached image.",
                    attachment_path=filename
                )
            
            last_unknown_time = now.timestamp()

# Load/Compute Encodings
if os.path.exists(encodings_file):
    encodeListKnown, classNames = load_encodings()
    print("✅ Loaded encodings from file")
else:
    encodeListKnown = find_encodings(images)
    save_encodings(encodeListKnown, class_names)
    print("✅ Encodings computed and saved")

print("Encoding Complete")

#Webcam
cap = cv2.VideoCapture(0)

unknown_in_frame = False

while True:
    success, img = cap.read()

    if not success:
        print("⚠️ Webcam not accessible")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit("Exiting program due to webcam error.")

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodedCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    unknown_detected_this_frame = False 

    for encodedFace, faceLoc in zip(encodedCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodedFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodedFace)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if faceDis[matchIndex] < CONFIDENCE_THRESHOLD and matches[matchIndex]:
            name = classNames[matchIndex]
            color = (0, 255, 0)
            mark_attendance(name)
        else:
            name = "Unknown"
            color = (0, 0, 255)
            unknown_detected_this_frame = True

            y1, y2 = max(0, y1), min(img.shape[0], y2)
            x1, x2 = max(0, x1), min(img.shape[1], x2)
            face_crop = img[y1:y2, x1:x2]

            if not unknown_in_frame:
                mark_attendance(name, face_crop)
                unknown_in_frame = True

      
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        
        display_name = format_name(name)  
        cv2.putText(img, display_name, (x1 + 6, y2 - 6),
            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    if not unknown_detected_this_frame:
        unknown_in_frame = False

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()