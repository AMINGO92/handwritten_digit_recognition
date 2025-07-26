CREATE DATABASE digit_db;

USE digit_db;

CREATE TABLE digit_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255),
    predicted_digit INT,
    prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP
);

