-- Smart Hospital: database upgrade for booking approval and doctor leave.
-- The web app runs these steps by itself when it starts (python app.py), so you
-- normally do NOT need this file. Use it only if the app says it could not
-- upgrade the database. Run it ONCE in MySQL Workbench.

USE smart_hospital;

-- 1. Bookings can now wait for the doctor's acceptance ('pending')
ALTER TABLE appointments
  MODIFY COLUMN status ENUM('scheduled','completed','cancelled','no_show','pending')
  NOT NULL DEFAULT 'scheduled';

-- 2. Remember who cancelled an appointment and why
ALTER TABLE appointments
  ADD COLUMN cancelled_by ENUM('patient','doctor') NULL,
  ADD COLUMN cancel_reason VARCHAR(255) NULL;

-- 3. Days a doctor is on leave
CREATE TABLE IF NOT EXISTS doctor_leaves (
    leave_id    INT AUTO_INCREMENT PRIMARY KEY,
    doctor_id   INT NOT NULL,
    leave_date  DATE NOT NULL,
    reason      VARCHAR(255) NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id),
    UNIQUE (doctor_id, leave_date)
);
