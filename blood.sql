CREATE DATABASE blood_transfusion;

CREATE TABLE blood(
	recency INT,
    frequency INT,
    monetary INT,
    months INT,
    donate INT);
USE blood_transfusion;

UPDATE blood;
SET donate = CASE 
                WHEN donate = 2 THEN 1
                ELSE 0
             END;

SELECT recency, frequency, monetary, months, donate 
FROM blood;