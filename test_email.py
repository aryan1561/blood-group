import smtplib
from email.mime.text import MIMEText

EMAIL = "nagraluaryan@gmail.com"
APP_PASSWORD = "qbbq iqjm jsaq xvks"

msg = MIMEText("This is a test message from Python.")
msg["Subject"] = "Test Email"
msg["From"] = EMAIL
msg["To"] = EMAIL

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL, APP_PASSWORD)
        server.send_message(msg)
        print("✅ Email sent successfully.")
except Exception as e:
    print("❌ Email failed:", e)
