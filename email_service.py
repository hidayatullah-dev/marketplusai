import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='email_logs.log'
)
logger = logging.getLogger('email_service')

class EmailService:
    def __init__(self, smtp_server="smtp.gmail.com", port=465):
        """Initialize email service with server config"""
        self.smtp_server = smtp_server
        self.port = port
        self.sender_email = os.environ.get("EMAIL_USER", "hidayatullah2269@gmail.com")
        self.sender_password = os.environ.get("EMAIL_PASSWORD", "")  # Set this through environment variables for security
        
    def send_email(self, recipient, subject, body, attachments=None):
        """Send email with optional attachments"""
        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient
            message["Subject"] = subject
            
            # Add body to email
            message.attach(MIMEText(body, "html"))
            
            # Add attachments if any
            if attachments:
                for file_path in attachments:
                    try:
                        with open(file_path, "rb") as file:
                            part = MIMEApplication(file.read(), Name=os.path.basename(file_path))
                            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                            message.attach(part)
                    except Exception as e:
                        logger.error(f"Failed to attach file {file_path}: {str(e)}")
            
            # Connect to SMTP server
            with smtplib.SMTP_SSL(self.smtp_server, self.port) as server:
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            logger.info(f"Email sent successfully to {recipient}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {str(e)}")
            return False
    
    def send_contact_confirmation(self, name, email, message):
        """Send confirmation email for contact form submission"""
        subject = "We received your message - MarketPulse AI"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px; background-color: #f9f9f9;">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1 style="color: #9333EA;">MarketPulse AI</h1>
                </div>
                <h2 style="color: #444;">Thank you for contacting us!</h2>
                <p>Hello {name},</p>
                <p>We have received your message and will get back to you as soon as possible.</p>
                <p>Your message:</p>
                <blockquote style="background-color: #f0f0f0; padding: 15px; border-left: 4px solid #9333EA; margin: 20px 0;">
                    {message}
                </blockquote>
                <p>Best regards,<br>MarketPulse AI Team</p>
            </div>
        </body>
        </html>
        """
        return self.send_email(email, subject, body)
    
    def notify_admin_contact(self, name, email, message):
        """Notify admin about new contact form submission"""
        admin_email = "hidayatullah2269@gmail.com"  # Admin email
        subject = f"New Contact Form Submission from {name}"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px; background-color: #f9f9f9;">
                <h2 style="color: #9333EA;">New Contact Form Submission</h2>
                <p><strong>Name:</strong> {name}</p>
                <p><strong>Email:</strong> {email}</p>
                <p><strong>Message:</strong></p>
                <blockquote style="background-color: #f0f0f0; padding: 15px; border-left: 4px solid #9333EA; margin: 20px 0;">
                    {message}
                </blockquote>
                <p>Respond to this inquiry by replying directly to this email.</p>
            </div>
        </body>
        </html>
        """
        return self.send_email(admin_email, subject, body)
    
    def send_subscription_confirmation(self, email, plan_type, plan_price, expiration_date):
        """Send subscription confirmation email"""
        subject = "Your MarketPulse AI PRO Subscription"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px; background-color: #f9f9f9;">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1 style="color: #9333EA;">MarketPulse AI PRO</h1>
                </div>
                <h2 style="color: #444;">Subscription Confirmed!</h2>
                <p>Hello,</p>
                <p>Your subscription has been activated successfully.</p>
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>Plan:</strong> {plan_type.capitalize()}</p>
                    <p><strong>Price:</strong> ${plan_price}</p>
                    <p><strong>Valid until:</strong> {expiration_date.strftime('%Y-%m-%d')}</p>
                </div>
                <p>You now have access to all premium features including:</p>
                <ul>
                    <li>Advanced technical indicators</li>
                    <li>Machine learning predictions</li>
                    <li>Multiple visualization options</li>
                    <li>Priority customer support</li>
                </ul>
                <p>Enjoy your premium features!</p>
                <p>Best regards,<br>MarketPulse AI Team</p>
            </div>
        </body>
        </html>
        """
        return self.send_email(email, subject, body)
    
    def send_forecast_report(self, email, asset_name, forecast_data, chart_image=None):
        """Send forecast report email with optional chart attachment"""
        subject = f"MarketPulse AI: {asset_name} Forecast Report"
        
        # Format forecast data as HTML table
        forecast_table = "<table border='1' style='border-collapse: collapse; width: 100%; margin: 20px 0;'>"
        forecast_table += "<tr style='background-color: #f0f0f0;'><th>Date</th><th>Predicted Price</th><th>Lower Bound</th><th>Upper Bound</th></tr>"
        
        for date, row in forecast_data.iterrows():
            forecast_table += f"<tr><td>{date.strftime('%Y-%m-%d')}</td><td>${row['Close']:.2f}</td><td>${row['Lower']:.2f}</td><td>${row['Upper']:.2f}</td></tr>"
        
        forecast_table += "</table>"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px; background-color: #f9f9f9;">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1 style="color: #9333EA;">MarketPulse AI</h1>
                </div>
                <h2 style="color: #444;">{asset_name} Forecast Report</h2>
                <p>Hello,</p>
                <p>Here is your requested forecast report for {asset_name}:</p>
                
                {forecast_table}
                
                <p>This forecast was generated using advanced statistical models and historical data analysis.</p>
                <p>Please note that market forecasts are subject to uncertainty and should be used as one of many tools in your investment decision process.</p>
                <p>Best regards,<br>MarketPulse AI Team</p>
            </div>
        </body>
        </html>
        """
        
        # Prepare attachments
        attachments = []
        if chart_image:
            attachments.append(chart_image)
            
        return self.send_email(email, subject, body, attachments)

# Create a singleton instance
email_service = EmailService() 