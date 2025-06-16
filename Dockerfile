# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container 
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest 
COPY . .

# Make port 5000 
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]