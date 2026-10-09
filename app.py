import os
import time
import sqlite3
from flask import Flask, render_template_string, jsonify, request
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from ncps.torch import CfC

torch.set_num_threads(1)

# ==========================================
# 1. DATABASE INITIALIZATION
# ==========================================
DB_FILE = 'thermocast.db'
app = Flask(__name__)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            heat_index REAL NOT NULL,
            forecast_3hr_hi REAL NOT NULL,
            pagasa_advisory TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

latest_data = {
    "timestamp": "--:--:--",
    "temperature": "--",
    "humidity": "--",
    "current_hi": "--",
    "forecast_hi": "--",
    "cur_category": "WAITING...",
    "cur_action": "Awaiting wireless ESP32 payload...",
    "cur_color": "#64748b",
    "fore_category": "WAITING...",
    "fore_action": "Awaiting wireless ESP32 payload...",
    "fore_color": "#64748b",
    "status": "Listening for Wi-Fi Data"
}

# ==========================================
# 2. LOAD LNN MODEL
# ==========================================
class HeatIndexLNN(nn.Module):
    def __init__(self, input_size, hidden_units):
        super().__init__()
        self.lnn = CfC(input_size, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)
        
    def forward(self, x):
        out, _ = self.lnn(x)
        return self.fc(out[:, -1, :])

model = None
try:
    model = HeatIndexLNN(input_size=3, hidden_units=32)
    model.load_state_dict(torch.load("lnn_heat_index_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("LNN Model Loaded Successfully!")
except Exception as e:
    print(f"Model load warning: {e}")

# ==========================================
# 3. ROTHFUSZ & PAGASA LOGIC
# ==========================================
def calculate_heat_index(temp_c, rh):
    T = temp_c * 1.8 + 32
    hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (rh * 0.094))
    if hi >= 80:
        hi = (-42.379 + 2.04901523 * T + 10.14333127 * rh - 0.22475541 * T * rh 
              - 0.00683783 * T**2 - 0.05481717 * rh**2 + 0.00122874 * (T**2) * rh 
              + 0.00085282 * T * (rh**2) - 0.00000199 * (T**2) * (rh**2))
    return (hi - 32) / 1.8

def get_pagasa_advisory(hi_val):
    if hi_val < 27:
        return "NOT HAZARDOUS", "Conditions normal. Stay hydrated.", "#28a745"
    elif 27 <= hi_val <= 32:
        return "CAUTION", "Fatigue possible with prolonged exposure.", "#ffc107"
    elif 33 <= hi_val <= 41:
        return "EXTREME CAUTION", "Heat cramps and heat exhaustion possible.", "#fd7e14"
    elif 42 <= hi_val <= 51:
        return "DANGER", "Heat stroke likely with continued activity.", "#dc3545"
    else:
        return "EXTREME DANGER", "Heat stroke highly imminent!", "#721c24"

sequence_history = []

# ==========================================
# 4. TELEMETRY INGESTION (ESP32 POST)
# ==========================================
@app.route('/api/telemetry', methods=['POST'])
def receive_telemetry():
    global latest_data, sequence_history
    try:
        payload = request.get_json(force=True)
        temp = float(payload['temperature'])
        hum = float(payload['humidity'])
        current_hi = calculate_heat_index(temp, hum)
        
        sequence_history.append([temp, hum, current_hi])
        if len(sequence_history) > 24:
            sequence_history.pop(0)
        
        padded_seq = sequence_history.copy()
        while len(padded_seq) < 24:
            padded_seq.insert(0, [temp, hum, current_hi])
        
        input_tensor = torch.tensor([padded_seq], dtype=torch.float32)
        
        if model is not None:
            with torch.no_grad():
                pred_raw = model(input_tensor).item()
            predicted_3hr_hi = current_hi + (pred_raw * 0.2)
        else:
            predicted_3hr_hi = current_hi
        
        cur_category, cur_action, cur_color = get_pagasa_advisory(current_hi)
        fore_category, fore_action, fore_color = get_pagasa_advisory(predicted_3hr_hi)
        timestamp_time = time.strftime("%H:%M:%S")
        full_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save to SQLite Database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sensor_readings (timestamp, temperature, humidity, heat_index, forecast_3hr_hi, pagasa_advisory)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (full_timestamp, temp, hum, round(current_hi, 2), round(predicted_3hr_hi, 2), fore_category))
        conn.commit()
        conn.close()

        latest_data.update({
            "timestamp": timestamp_time,
            "temperature": f"{temp:.1f}",
            "humidity": f"{hum:.1f}",
            "current_hi": f"{current_hi:.1f}",
            "forecast_hi": f"{predicted_3hr_hi:.1f}",
            "cur_category": cur_category,
            "cur_action": cur_action,
            "cur_color": cur_color,
            "fore_category": fore_category,
            "fore_action": fore_action,
            "fore_color": fore_color,
            "status": "Wi-Fi Telemetry Active (Stored in DB)"
        })
        return jsonify({"status": "success", "db_logged": True}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ==========================================
# 5. REST API ENDPOINTS FOR RETRIEVAL
# ==========================================
@app.route('/api/v1/latest', methods=['GET'])
def get_latest_api():
    return jsonify(latest_data)

@app.route('/api/v1/history', methods=['GET'])
def get_history_api():
    limit = request.args.get('limit', default=20, type=int)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, temperature, humidity, heat_index, forecast_3hr_hi, pagasa_advisory FROM sensor_readings ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    data = []
    for r in rows:
        data.append({
            "id": r[0],
            "timestamp": r[1],
            "temperature_c": r[2],
            "humidity_pct": r[3],
            "heat_index_c": r[4],
            "forecast_3hr_c": r[5],
            "pagasa_advisory": r[6]
        })
    return jsonify({"count": len(data), "readings": data})

# ==========================================
# 6. WEB DASHBOARD UI (HOMEPAGE ROUTE)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThermoCast AI - Heat Index Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            background: linear-gradient(180deg, #bde0fe 0%, #dbeafe 30%, #e8f4f8 60%, #f0f7fa 100%);
            color: #1e293b;
            min-height: 100vh;
        }

        /* -- Navbar -- */
        .navbar {
            background: #1e293b;
            padding: 14px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        }
        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 800;
            font-size: 1.25em;
            color: #fff;
            letter-spacing: -0.5px;
        }
        .navbar-brand .logo-icon {
            width: 32px; height: 32px;
            background: linear-gradient(135deg, #f59e0b, #ef4444);
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; color: #fff; font-weight: 900;
        }
        .nav-links {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .nav-link {
            background: transparent;
            color: #cbd5e1;
            border: none;
            padding: 8px 18px;
            border-radius: 24px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.85em;
            font-weight: 500;
            transition: all 0.25s ease;
            text-decoration: none;
        }
        .nav-link:hover { background: #334155; color: #fff; }
        .nav-link.active-link {
            background: #f59e0b;
            color: #1e293b;
            font-weight: 700;
        }
        .nav-status {
            display: flex; align-items: center; gap: 6px;
            color: #94a3b8; font-size: 0.8em;
        }
        .nav-status .dot {
            width: 8px; height: 8px;
            background: #22c55e;
            border-radius: 50%;
            animation: pulse-dot 2s ease-in-out infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.4); }
        }

        /* -- Main Content -- */
        .main { max-width: 1100px; margin: 0 auto; padding: 36px 28px 28px; }

        /* -- Hero Section -- */
        .hero {
            display: grid;
            grid-template-columns: 1.1fr 1fr;
            gap: 32px;
            align-items: start;
            margin-bottom: 28px;
        }

        .hero-left .location-label {
            font-size: 0.72em;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #6b7280;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .hero-left .location-name {
            font-size: 1.6em;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 16px;
            letter-spacing: -0.5px;
        }
        .hero-left .reading-label {
            font-size: 0.92em;
            color: #6b7280;
            margin-bottom: 8px;
        }
        .hero-left .reading-label span { color: #2563eb; font-weight: 600; }

        .temp-display {
            display: flex;
            align-items: flex-start;
            gap: 6px;
            margin-bottom: 16px;
        }
        .temp-value {
            font-size: 5.5em;
            font-weight: 900;
            color: #1e293b;
            line-height: 1;
            letter-spacing: -3px;
        }
        .temp-unit {
            font-size: 1.8em;
            font-weight: 300;
            color: #64748b;
            margin-top: 8px;
        }

        .advisory-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 20px;
            border-radius: 24px;
            font-weight: 700;
            font-size: 0.85em;
            color: #fff;
            background: #64748b;
            margin-bottom: 14px;
            transition: background-color 0.4s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }
        .advisory-badge .badge-icon { font-size: 1.1em; }

        .advisory-text {
            font-size: 0.88em;
            color: #6b7280;
            line-height: 1.5;
            max-width: 380px;
        }

        /* -- Metric Cards Grid -- */
        .cards-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            padding: 18px 20px;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        }
        .metric-card .card-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }
        .metric-card .card-icon {
            width: 28px; height: 28px;
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px;
        }
        .metric-card .card-title {
            font-size: 0.82em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .metric-card .card-value {
            font-size: 2em;
            font-weight: 800;
            color: #1e293b;
            line-height: 1.1;
        }
        .metric-card .card-value .card-unit {
            font-size: 0.45em;
            font-weight: 500;
            color: #94a3b8;
            margin-left: 2px;
        }
        .metric-card .card-sub {
            font-size: 0.78em;
            color: #6b7280;
            margin-top: 4px;
            font-weight: 500;
        }

        /* Card accent colors */
        .card-heat .card-header { border-bottom-color: #f59e0b; }
        .card-heat .card-icon { background: #fef3c7; color: #d97706; }
        .card-heat .card-title { color: #d97706; }

        .card-humidity .card-header { border-bottom-color: #3b82f6; }
        .card-humidity .card-icon { background: #dbeafe; color: #2563eb; }
        .card-humidity .card-title { color: #2563eb; }

        .card-temp .card-header { border-bottom-color: #10b981; }
        .card-temp .card-icon { background: #d1fae5; color: #059669; }
        .card-temp .card-title { color: #059669; }

        .card-forecast .card-header { border-bottom-color: #ef4444; }
        .card-forecast .card-icon { background: #fee2e2; color: #dc2626; }
        .card-forecast .card-title { color: #dc2626; }

        /* -- Advisory Panel -- */
        .advisory-panel {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 28px;
        }
        .adv-card {
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.8);
            border-radius: 16px;
            padding: 22px 24px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        }
        .adv-card h4 {
            font-size: 0.72em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #94a3b8;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .adv-badge {
            display: inline-block;
            padding: 8px 22px;
            border-radius: 24px;
            font-weight: 700;
            font-size: 0.9em;
            color: #fff;
            background: #64748b;
            margin-bottom: 10px;
            transition: background-color 0.4s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .adv-action {
            font-size: 0.82em;
            color: #6b7280;
            line-height: 1.5;
        }

        /* -- Chart Section -- */
        .chart-section {
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        }
        .chart-section .chart-title {
            font-size: 0.78em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 16px;
        }

        /* -- Footer -- */
        .footer {
            text-align: center;
            font-size: 0.78em;
            color: #94a3b8;
            padding: 8px 0 24px;
        }
        .footer code {
            background: rgba(255,255,255,0.5);
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.95em;
        }

        /* -- Weather SVG icon -- */
        .weather-svg { width: 72px; height: 72px; margin-left: 12px; margin-top: 8px; }

        /* -- Responsive -- */
        @media (max-width: 768px) {
            .hero { grid-template-columns: 1fr; gap: 20px; }
            .cards-grid { grid-template-columns: 1fr 1fr; }
            .advisory-panel { grid-template-columns: 1fr; }
            .temp-value { font-size: 4em; }
            .main { padding: 20px 16px; }
        }
    </style>
</head>
<body>

    <!-- Navbar -->
    <nav class="navbar">
        <div class="navbar-brand">
            <div class="logo-icon">TC</div>
            <span>Thermo<span style="font-weight:400; color:#f59e0b;">Cast</span></span>
        </div>
        <div class="nav-links">
            <a href="/" class="nav-link active-link">Dashboard</a>
            <a href="/logs" class="nav-link">Data Logs</a>
            <div class="nav-status">
                <div class="dot"></div>
                <span id="nav-status-text">Listening</span>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="main">

        <!-- Hero: Large temp + Cards -->
        <div class="hero">
            <div class="hero-left">
                <div class="location-label">IoT Sensor Node</div>
                <div class="location-name">ThermoCast AI</div>
                <div class="reading-label">Heat Index reading at <span id="reading-time">--:--:--</span></div>

                <div class="temp-display">
                    <span class="temp-value" id="hero-temp">--</span>
                    <span class="temp-unit">&deg;C</span>
                    <!-- Sun/cloud SVG -->
                    <svg class="weather-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="55" cy="40" r="26" fill="#f59e0b"/>
                        <circle cx="55" cy="40" r="22" fill="#fbbf24"/>
                        <ellipse cx="42" cy="65" rx="26" ry="16" fill="#e2e8f0"/>
                        <ellipse cx="56" cy="60" rx="20" ry="14" fill="#f1f5f9"/>
                        <ellipse cx="35" cy="62" rx="14" ry="10" fill="#fff"/>
                    </svg>
                </div>

                <div class="advisory-badge" id="hero-badge">
                    <span class="badge-icon">&#9650;</span>
                    <span id="hero-badge-text">WAITING...</span>
                </div>
                <div class="advisory-text" id="hero-advisory-text">Awaiting wireless ESP32 payload...</div>
            </div>

            <div class="cards-grid">
                <!-- Heat Index Card -->
                <div class="metric-card card-heat">
                    <div class="card-header">
                        <div class="card-icon">&#9728;</div>
                        <span class="card-title">Heat Index</span>
                    </div>
                    <div class="card-value" id="card-hi">--<span class="card-unit">&deg;C</span></div>
                    <div class="card-sub" id="card-hi-sub">Awaiting data</div>
                </div>
                <!-- Humidity Card -->
                <div class="metric-card card-humidity">
                    <div class="card-header">
                        <div class="card-icon">&#128167;</div>
                        <span class="card-title">Humidity</span>
                    </div>
                    <div class="card-value" id="card-hum">--<span class="card-unit">%</span></div>
                    <div class="card-sub" id="card-hum-sub">Awaiting data</div>
                </div>
                <!-- Temperature Card -->
                <div class="metric-card card-temp">
                    <div class="card-header">
                        <div class="card-icon">&#127777;</div>
                        <span class="card-title">Temperature</span>
                    </div>
                    <div class="card-value" id="card-temp">--<span class="card-unit">&deg;C</span></div>
                    <div class="card-sub">Ambient reading</div>
                </div>
                <!-- Forecast Card -->
                <div class="metric-card card-forecast">
                    <div class="card-header">
                        <div class="card-icon">&#9203;</div>
                        <span class="card-title">3-Hr Forecast</span>
                    </div>
                    <div class="card-value" id="card-fore">--<span class="card-unit">&deg;C</span></div>
                    <div class="card-sub" id="card-fore-sub">LNN prediction</div>
                </div>
            </div>
        </div>

        <!-- Advisory Panel -->
        <div class="advisory-panel">
            <div class="adv-card">
                <h4>Current PAGASA Advisory</h4>
                <div class="adv-badge" id="cur_badge">WAITING...</div>
                <div class="adv-action" id="cur_action">Awaiting telemetry payload...</div>
            </div>
            <div class="adv-card">
                <h4>3-Hour Forecasted Advisory</h4>
                <div class="adv-badge" id="fore_badge">WAITING...</div>
                <div class="adv-action" id="fore_action">Awaiting telemetry payload...</div>
            </div>
        </div>

        <!-- Chart -->
        <div class="chart-section">
            <div class="chart-title">Real-Time Trend Curves</div>
            <canvas id="trendChart" height="100"></canvas>
        </div>

        <div class="footer">
            Status: <span id="status">Listening for Wi-Fi Data</span> &nbsp;|&nbsp; Database: <code>thermocast.db</code>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('trendChart').getContext('2d');
        const maxPoints = 20;

        const trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Temp (\\u00b0C)', borderColor: '#10b981', backgroundColor: '#10b98118', data: [], tension: 0.4, fill: true, pointRadius: 3, pointBackgroundColor: '#10b981', borderWidth: 2 },
                    { label: 'Humidity (%)', borderColor: '#3b82f6', backgroundColor: '#3b82f618', data: [], tension: 0.4, fill: true, pointRadius: 3, pointBackgroundColor: '#3b82f6', borderWidth: 2 },
                    { label: 'Current HI (\\u00b0C)', borderColor: '#f59e0b', backgroundColor: '#f59e0b18', data: [], tension: 0.4, fill: true, pointRadius: 3, pointBackgroundColor: '#f59e0b', borderWidth: 2 },
                    { label: '3-Hr Forecast (\\u00b0C)', borderColor: '#ef4444', backgroundColor: '#ef444418', borderDash: [6, 4], data: [], tension: 0.4, fill: false, pointRadius: 3, pointBackgroundColor: '#ef4444', borderWidth: 2 }
                ]
            },
            options: {
                animation: false,
                responsive: true,
                interaction: { intersect: false, mode: 'index' },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: '#e5e7eb' } },
                    y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: '#e5e7eb' } }
                },
                plugins: {
                    legend: { labels: { color: '#64748b', usePointStyle: true, pointStyle: 'circle', padding: 16, font: { size: 12 } } },
                    tooltip: { backgroundColor: '#1e293b', titleColor: '#f8fafc', bodyColor: '#cbd5e1', cornerRadius: 10, padding: 12 }
                }
            }
        });

        function getHumidityLabel(h) {
            if (h < 40) return 'Dry';
            if (h < 60) return 'Comfortable';
            if (h < 80) return 'Humid';
            return 'Very Humid';
        }

        function getHILabel(hi) {
            if (hi < 27) return 'Normal';
            if (hi <= 32) return 'Rest Often';
            if (hi <= 41) return 'Limit Exposure';
            if (hi <= 51) return 'Avoid Outdoors';
            return 'Stay Indoors';
        }

        async function updateDashboard() {
            try {
                const res = await fetch('/api/v1/latest');
                const data = await res.json();
                if (data.temperature === "--") return;

                const temp = parseFloat(data.temperature);
                const hum = parseFloat(data.humidity);
                const curHI = parseFloat(data.current_hi);
                const foreHI = parseFloat(data.forecast_hi);

                // Hero section
                document.getElementById('hero-temp').innerText = data.current_hi;
                document.getElementById('reading-time').innerText = data.timestamp;

                const heroBadge = document.getElementById('hero-badge');
                heroBadge.style.backgroundColor = data.cur_color || '#64748b';
                document.getElementById('hero-badge-text').innerText = data.cur_category || 'WAITING...';
                document.getElementById('hero-advisory-text').innerText = data.cur_action || '';

                // Metric cards
                document.getElementById('card-hi').innerHTML = data.current_hi + '<span class="card-unit">&deg;C</span>';
                document.getElementById('card-hi-sub').innerText = getHILabel(curHI);
                document.getElementById('card-hum').innerHTML = data.humidity + '<span class="card-unit">%</span>';
                document.getElementById('card-hum-sub').innerText = getHumidityLabel(hum);
                document.getElementById('card-temp').innerHTML = data.temperature + '<span class="card-unit">&deg;C</span>';
                document.getElementById('card-fore').innerHTML = data.forecast_hi + '<span class="card-unit">&deg;C</span>';
                document.getElementById('card-fore-sub').innerText = getHILabel(foreHI);

                // Advisory panels
                const curBadge = document.getElementById('cur_badge');
                curBadge.innerText = data.cur_category || 'WAITING...';
                curBadge.style.backgroundColor = data.cur_color || '#64748b';
                document.getElementById('cur_action').innerText = data.cur_action || 'Awaiting telemetry...';

                const foreBadge = document.getElementById('fore_badge');
                foreBadge.innerText = data.fore_category || 'WAITING...';
                foreBadge.style.backgroundColor = data.fore_color || '#64748b';
                document.getElementById('fore_action').innerText = data.fore_action || 'Awaiting telemetry...';

                // Status
                document.getElementById('status').innerText = data.status;
                document.getElementById('nav-status-text').innerText = 'Live';

                // Chart
                const timeLabel = data.timestamp;
                if (trendChart.data.labels.length === 0 || trendChart.data.labels[trendChart.data.labels.length - 1] !== timeLabel) {
                    trendChart.data.labels.push(timeLabel);
                    trendChart.data.datasets[0].data.push(temp);
                    trendChart.data.datasets[1].data.push(hum);
                    trendChart.data.datasets[2].data.push(curHI);
                    trendChart.data.datasets[3].data.push(foreHI);

                    if (trendChart.data.labels.length > maxPoints) {
                        trendChart.data.labels.shift();
                        trendChart.data.datasets.forEach(ds => ds.data.shift());
                    }
                    trendChart.update('none');
                }
            } catch (e) {
                console.error("Fetch error:", e);
            }
        }

        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# ==========================================
# 7. STORED SENSOR DATA TABLE PAGE
# ==========================================
@app.route('/logs')
def view_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, temperature, humidity, heat_index, forecast_3hr_hi, pagasa_advisory FROM sensor_readings ORDER BY id DESC LIMIT 100')
    rows = cursor.fetchall()
    conn.close()

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>ThermoCast - Stored Data Logs</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', 'Segoe UI', sans-serif; background: linear-gradient(180deg, #bde0fe 0%, #e8f4f8 100%); color: #1e293b; padding: 0; margin: 0; min-height: 100vh; }
            .navbar { background: #1e293b; padding: 14px 32px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 12px rgba(0,0,0,0.15); }
            .navbar-brand { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.25em; color: #fff; }
            .navbar-brand .logo-icon { width: 32px; height: 32px; background: linear-gradient(135deg, #f59e0b, #ef4444); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 14px; color: #fff; font-weight: 900; }
            .nav-links { display: flex; gap: 8px; }
            .nav-link { color: #cbd5e1; padding: 8px 18px; border-radius: 24px; text-decoration: none; font-size: 0.85em; font-weight: 500; transition: all 0.25s; }
            .nav-link:hover { background: #334155; color: #fff; }
            .nav-link.active-link { background: #f59e0b; color: #1e293b; font-weight: 700; }
            .container { max-width: 1100px; margin: auto; padding: 32px 28px; }
            h1 { color: #1e293b; margin-bottom: 5px; font-size: 1.5em; }
            p { color: #6b7280; font-size: 0.9em; margin-bottom: 20px; }
            a { color: #2563eb; text-decoration: none; font-weight: 600; }
            a:hover { text-decoration: underline; }
            table { width: 100%; border-collapse: collapse; margin-top: 8px; background: rgba(255,255,255,0.8); backdrop-filter: blur(12px); border-radius: 16px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.04); }
            th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #e5e7eb; font-size: 0.88em; }
            th { background: #1e293b; color: white; text-transform: uppercase; font-size: 0.72em; letter-spacing: 1px; font-weight: 600; }
            tr:hover { background: rgba(59,130,246,0.06); }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="navbar-brand">
                <div class="logo-icon">TC</div>
                <span>Thermo<span style="font-weight:400; color:#f59e0b;">Cast</span></span>
            </div>
            <div class="nav-links">
                <a href="/" class="nav-link">Dashboard</a>
                <a href="/logs" class="nav-link active-link">Data Logs</a>
            </div>
        </nav>
        <div class="container">
            <h1>Stored Sensor Data Logs</h1>
            <p><a href="/">&larr; Back to Live Dashboard</a> &nbsp;|&nbsp; Displaying last 100 records from <code>thermocast.db</code></p>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>Temp (&deg;C)</th>
                        <th>Humidity (%)</th>
                        <th>Heat Index (&deg;C)</th>
                        <th>3-Hr Forecast (&deg;C)</th>
                        <th>PAGASA Advisory</th>
                    </tr>
                </thead>
                <tbody>
    """
    for r in rows:
        html += f"""
                    <tr>
                        <td>{r[0]}</td>
                        <td>{r[1]}</td>
                        <td>{r[2]}</td>
                        <td>{r[3]}</td>
                        <td>{r[4]}</td>
                        <td>{r[5]}</td>
                        <td><strong>{r[6]}</strong></td>
                    </tr>
        """
    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"ThermoCast AI Server running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)