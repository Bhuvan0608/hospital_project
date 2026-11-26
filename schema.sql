-- Smart Hospital Management and Appointment Prediction System
-- MySQL 8 schema

CREATE DATABASE IF NOT EXISTS smart_hospital;
USE smart_hospital;

-- Login accounts for all three roles
CREATE TABLE users (
    user_id        INT AUTO_INCREMENT PRIMARY KEY,
    email          VARCHAR(120) NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    role           ENUM('patient', 'doctor', 'admin') NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE departments (
    department_id  INT AUTO_INCREMENT PRIMARY KEY,
    name           VARCHAR(80) NOT NULL UNIQUE,
    description    VARCHAR(255)
);

CREATE TABLE patients (
    patient_id     INT AUTO_INCREMENT PRIMARY KEY,
    user_id        INT NOT NULL UNIQUE,
    full_name      VARCHAR(100) NOT NULL,
    date_of_birth  DATE NOT NULL,
    gender         ENUM('M', 'F', 'Other') NOT NULL,
    phone          VARCHAR(15) NOT NULL,
    address        VARCHAR(255),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE doctors (
    doctor_id        INT AUTO_INCREMENT PRIMARY KEY,
    user_id          INT NOT NULL UNIQUE,
    department_id    INT NOT NULL,
    full_name        VARCHAR(100) NOT NULL,
    specialization   VARCHAR(100),
    qualification    VARCHAR(100),
    consultation_min INT NOT NULL DEFAULT 15,   -- average minutes per patient
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

-- Weekly availability of each doctor
CREATE TABLE doctor_schedules (
    schedule_id    INT AUTO_INCREMENT PRIMARY KEY,
    doctor_id      INT NOT NULL,
    day_of_week    TINYINT NOT NULL,            -- 0 = Monday ... 6 = Sunday
    start_time     TIME NOT NULL,
    end_time       TIME NOT NULL,
    max_patients   INT NOT NULL DEFAULT 20,
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id),
    UNIQUE (doctor_id, day_of_week, start_time)
);

-- Core table; also the main source of ML training data
CREATE TABLE appointments (
    appointment_id    INT AUTO_INCREMENT PRIMARY KEY,
    patient_id        INT NOT NULL,
    doctor_id         INT NOT NULL,
    booked_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    appointment_date  DATE NOT NULL,
    slot_time         TIME NOT NULL,
    status            ENUM('scheduled', 'completed', 'cancelled', 'no_show', 'pending')
                      NOT NULL DEFAULT 'scheduled',   -- 'pending' = waiting for the doctor to accept
    reason            VARCHAR(255),
    sms_reminder_sent BOOLEAN NOT NULL DEFAULT FALSE,
    actual_start_time TIME NULL,                -- filled in after the visit
    waiting_minutes   INT NULL,                 -- target for waiting-time model
    cancelled_by      ENUM('patient', 'doctor') NULL,
    cancel_reason     VARCHAR(255) NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id),
    UNIQUE (doctor_id, appointment_date, slot_time)   -- prevents double booking
);

CREATE TABLE consultations (
    consultation_id  INT AUTO_INCREMENT PRIMARY KEY,
    appointment_id   INT NOT NULL UNIQUE,
    notes            TEXT,
    prescription     TEXT,
    follow_up_date   DATE NULL,
    FOREIGN KEY (appointment_id) REFERENCES appointments(appointment_id)
);

-- Stored model outputs, so the dashboard does not recompute them
CREATE TABLE predictions (
    prediction_id        INT AUTO_INCREMENT PRIMARY KEY,
    appointment_id       INT NOT NULL UNIQUE,
    no_show_probability  DECIMAL(5,4) NOT NULL,   -- 0.0000 to 1.0000
    predicted_wait_min   INT NULL,
    model_version        VARCHAR(30) NOT NULL,
    predicted_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (appointment_id) REFERENCES appointments(appointment_id)
);

-- Days a doctor is on leave (no bookings are possible on these dates)
CREATE TABLE doctor_leaves (
    leave_id    INT AUTO_INCREMENT PRIMARY KEY,
    doctor_id   INT NOT NULL,
    leave_date  DATE NOT NULL,
    reason      VARCHAR(255) NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id),
    UNIQUE (doctor_id, leave_date)
);

CREATE INDEX idx_appt_date   ON appointments (appointment_date);
CREATE INDEX idx_appt_status ON appointments (status);
CREATE INDEX idx_appt_patient ON appointments (patient_id);
