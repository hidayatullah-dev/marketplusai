# Deployment Instructions for MarketPulse AI

## Option 1: Deploy to Streamlit Cloud (Recommended)

Streamlit Cloud offers free hosting for Streamlit apps directly from GitHub repositories.

### Steps:

1. **Create a GitHub repository**
   - Go to [GitHub](https://github.com) and sign in or create an account
   - Click the "+" icon in the top right and select "New repository"
   - Name your repository (e.g., "marketpulse-ai")
   - Make it Public
   - Click "Create repository"

2. **Upload your code to GitHub**
   - You can use GitHub Desktop (easiest for beginners):
     - [Download GitHub Desktop](https://desktop.github.com/)
     - Install and sign in
     - Choose "Add an Existing Repository from your Hard Drive"
     - Browse to `C:\Users\AT\Desktop\FinancialForecast`
     - Commit all changes and click "Publish repository"
     - Select the repository you created and click "Publish"

   - Alternatively, if you have Git installed, you can use these commands:
     ```
     cd C:\Users\AT\Desktop\FinancialForecast
     git init
     git add .
     git commit -m "Initial commit"
     git branch -M main
     git remote add origin https://github.com/YOUR_USERNAME/marketpulse-ai.git
     git push -u origin main
     ```

3. **Deploy to Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch (main), and main file path (app.py)
   - Click "Deploy"
   - Wait for the deployment to complete

4. **Access Your App**
   - Once deployed, Streamlit Cloud will provide a URL for your app
   - Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`

## Option 2: Deploy to Render.com (Alternative)

Render offers a free tier for web services and works well with Streamlit apps.

### Steps:

1. **Push your code to GitHub** (follow steps from Option 1)

2. **Create Render account**
   - Go to [Render](https://render.com) and sign up

3. **Deploy your app**
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Fill in the details:
     - Name: MarketPulse-AI
     - Environment: Python 3
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - Choose the free plan
   - Click "Create Web Service"

## Option 3: Deploy to Heroku (Alternative)

You'll need a free Heroku account:

1. **Create a `Procfile` in your project root**
   - Add this line: `web: streamlit run app.py --server.port=$PORT`

2. **Install Heroku CLI and deploy**
   - Download [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
   - Run these commands:
     ```
     heroku login
     cd C:\Users\AT\Desktop\FinancialForecast
     heroku create marketpulse-ai
     git add .
     git commit -m "Heroku deployment"
     git push heroku main
     ```

## IMPORTANT NOTE

For any of these deployment options, be aware of the following limitations:

1. **Database**: The SQLite database will be ephemeral (temporary) on these free hosting platforms. For persistent data storage, consider connecting to a cloud database service like Supabase, MongoDB Atlas, or Firebase (all have free tiers).

2. **Email functionality**: You'll need to set environment secrets/variables for email credentials in your hosting platform settings.

3. **API Limits**: Free deployments may have limitations on how often you can fetch finance data from external APIs. 