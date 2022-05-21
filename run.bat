docker build -m 2g -t training -f "TrainingDocker" . & docker run --gpus all training
docker build -m 2g -t chatbot -f "ChatbotDocker" . & docker run --gpus all -p 5000:5000 chatbot