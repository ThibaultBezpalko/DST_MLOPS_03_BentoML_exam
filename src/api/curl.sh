#!/bin/bash

# Get the JWT token
token=$(curl -s -X POST http://127.0.0.1:3000/login \
     -H "Content-Type: application/json" \
     -d '{"username": "user123", "password": "password123"}' | jq -r '.token')

# Display the token
echo "Token JWT obtenu: $token"

# Use this token to send a POST request to the prediction endpoint
response=$(curl -s -X POST http://127.0.0.1:3000/v1/models/rf_regressor/predict \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $token" \
     -d '{
          "serial_nb": 362,
          "gre": 334,
          "toefl": 116,
          "university_rating": 4,
          "sop": 4.0,
          "lor" : 3.5,
          "cgpa": 9.54,
          "research": 1
     }')

# Display the prediction
echo "Réponse de l'API de prédiction: $response"