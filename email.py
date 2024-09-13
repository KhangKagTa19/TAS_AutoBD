from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import streamlit as st
import os

def send_email(to_email, subject, content):
    message = Mail(
        from_email='tagiakhang19@gmail.com',
        to_emails=to_email,
        subject=subject,
        html_content=content)
    try:
        # Prompt the user to input the SendGrid API key
        # Retrieve the SendGrid API key from environment variables
        sg = SendGridAPIClient('SEND_GRID_API_KEY')
        response = sg.send(message)
        return True, f"Email sent successfully. Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)


def add_email_manually(email_list):
    new_email = st.text_input("Add a new email manually:")
    if st.button("Add Email"):
        if new_email and new_email not in email_list:
            email_list.append(new_email)
            st.success(f"Email {new_email} added successfully!")
        elif new_email in email_list:
            st.warning("This email is already in the list.")
        else:
            st.warning("Please enter a valid email.")
    return email_list

