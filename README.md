# GridMind — Smart Energy Microgrid & Demand Forecasting Engine

**HackSphere 3.0: Smart Energy Microgrid & Demand Forecasting Engine**  
*Full-Stack Energy Intelligence Platform for Zoho Catalyst AppSail & QuickML*

---

## 1. Project Overview

**GridMind** is an end-to-end intelligent energy microgrid decision-support, 24-hour demand forecasting, battery simulation, and machine learning research platform built for **HackSphere 3.0**. 

The system accepts hourly weather and telemetry inputs, forecasts electricity demand over a 24-hour horizon, deterministically evaluates grid dispatch operating modes according to official HackSphere 3.0 rules, simulates battery electrochemical storage (BESS), and generates critical overload alerts.

> **Operational Scope & Decision Support Notice**: GridMind is a software-level decision-support and simulation engine. It does not directly command physical switchgear or high-voltage electrical actuators.

---

## 2. Core Architecture & Technologies

- **Frontend**: React 19, TypeScript, Vite 8, Tailwind CSS, Lucide icons, Motion.
- **Backend API**: Express 4 running under Node.js 22 with ESM / TypeScript type-stripping support, Vite development middleware in dev mode, and static asset serving in production.
- **Machine Learning Engine**:
  - **Production Prediction Service**: Zoho Catalyst QuickML Tabular Regression REST endpoint client (`src/engine/quickmlClient.ts`) with zero client secrets, SSRF-protected server proxy, standard OAuth headers, and transparent fallback.
  - **Unified Production Fallback**: Identical L2-regularized Ridge Regression model implemented identically in Python (`predict.py`) and TypeScript (`src/engine/forecastingEngine.ts`). Standardized feature extraction with exact pre-fit coefficients ($\alpha = 1.0$), eliminating model divergence.
  - **ML Research Lab**: Pure TypeScript implementations of Gradient Descent, Single Perceptron binary load classification, and Deep Multilayer Perceptron (MLP) with analytical backpropagation and finite-difference gradient verification.
  - **Model Explainability**: Breiman-Fisher out-of-sample Permutation Feature Importance ($\Delta$RMSE).
  - **Uncertainty Quantification**: Calibrated distribution-free Conformal Prediction intervals (80%, 90%, 95%).
  - **Battery Scenario Simulation**: Multi-scenario comparison of BESS capacities (0 kWh, 50 kWh, 100 kWh, 200 kWh) with peak-shaving and overload mitigation impact.
- **Data Persistence**: File-backed atomic JSON persistence (`data/persisted_telemetry.json`, `data/persisted_forecast.json`, `data/persisted_events.json`, `data/persisted_settings.json`) surviving server restarts.
- **Evaluation Contract**: Standalone dual-interface `predict.py` supporting Python starter kit function `predict(df)` and CLI pipeline.

---

## 3. Official HackSphere 3.0 Controller Rules & Threshold Parity

The energy transition controller deterministically evaluates every forecast hour $t$:

1. **Renewable Storage Mode**:
   $$\text{Mode} = \text{Renewable Storage Mode} \iff \text{Demand}_t > 160.0\text{ kW} \quad\lor\quad \text{Hour}_t \in [18, 19, 20, 21, 22]$$
2. **Main Grid Mode**:
   $$\text{Mode} = \text{Main Grid Mode} \iff \text{Demand}_t \le 160.0\text{ kW} \quad\land\quad \text{Hour}_t \notin [18, 19, 20, 21, 22]$$
3. **Critical Overload Alert**:
   $$\text{Alert} = \text{True} \iff \text{Demand}_t \ge 200.0\text{ kW}$$

> **Threshold Parity Guarantee**: All sub-engines, including `peakIntelligenceEngine.ts`, strictly use `kw > 160.0` (correcting legacy `>= 160.0` discrepancies). Exactly at 160.00 kW outside evening hours, the system remains in **Main Grid Mode**. At 160.01 kW, it transitions to **Renewable Storage Mode**.

---

## 4. Telemetry Schema, Datasets & Validation

### Telemetry Schema:
Four required columns with strict RFC 4180 parsing, NaN/Inf rejection, and range checks:
- `timestamp`: ISO 8601 string (e.g., `2026-10-20T00:00:00` or `2026-10-20 00:00:00`).
- `ambient_temp_c`: Ambient temperature in Celsius (-40.0°C to 70.0°C).
- `humidity_pct`: Relative humidity percentage (0.0% to 100.0%).
- `solar_irradiance_wm2`: Solar irradiance in W/m² (0.0 to 1500.0 W/m²).
- `target_demand_kw` *(training only)*: Electrical load in kW (0.0 to 1000.0 kW).

### Official Datasets:
- **`train_telemetry.csv`**: Exactly 72 continuous hourly rows from `2026-10-20T00:00:00` (Tuesday) to `2026-10-22T23:00:00` (Thursday). Zero missing rows, zero weekend observations.
- **`sample_test_input.csv`**: Exactly 24 continuous hourly rows for testing (`2026-10-27T00:00:00` to `2026-10-27T23:00:00`).
- **`sample_predictions.csv`**: 24-hour test forecasts produced by `predict.py`.

### RFC 4180 Parser Features (`src/engine/telemetryValidator.ts`):
- Handles CRLF (`\r\n`) and LF (`\n`).
- Strips UTF-8 Byte Order Mark (`\uFEFF`).
- Supports quoted fields containing commas, spaces, and escaped double quotes (`""`).
- Rejects malformed quotes, missing columns, duplicate column headers, and NaN/Infinity values.
- Rejects negative irradiance, out-of-range humidity, and physically impossible temperatures with HTTP 400.

---

## 5. Machine Learning Evaluation & Reproducibility

### Exact Model Replication:
The unified forecasting model uses standardized features:
1. `hour_sin` = $\sin(2\pi \cdot \text{hour} / 24)$
2. `hour_cos` = $\cos(2\pi \cdot \text{hour} / 24)$
3. `solar_norm` = $\text{solar} / 1000$
4. `temp_c`
5. `humidity_pct`
6. `temp_squared` = $\text{temp}^2$
7. `cooling_degree` = $\max(0, \text{temp} - 22)$
8. `solar_temp_interaction` = $\text{temp} \cdot (\text{solar} / 1000)$

### Benchmark Evaluation Metrics on Official Telemetry (`train_telemetry.csv`):
- **Overall Training Fit (72 observations)**:
  - **RMSE**: `10.03 kW`
  - **MAE**: `7.56 kW`
  - **$R^2$**: `0.943`
- **Chronological TimeSeriesSplit Cross-Validation (Walk-Forward)**:
  - **Average Held-Out Test RMSE**: `10.45 kW`
  - **Average Held-Out Test MAE**: `7.79 kW`
  - **Average Held-Out Test $R^2$**: `0.938`

### Small-Dataset Limitation & Heuristic Removal Disclosure:
1. **72-Row Horizon**: The dataset covers only 3 consecutive autumn weekdays (Tuesday through Thursday). It does not contain multi-month seasonal variation or weekend data.
2. **Weekend Heuristic Removed**: Prior iterations applied an arbitrary 8% weekend multiplier (`pred *= 0.92`). Because `train_telemetry.csv` contains **zero weekend observations**, this multiplier was mathematically ungrounded and has been completely removed to maintain empirical integrity.

---

## 6. Authoritative Battery Configuration Validator (`src/engine/batteryValidator.ts`)

Battery parameters are strictly validated server-side and client-side before updating state:
- `battery_capacity_kwh`: $\ge 0.0$ and finite. Explicitly supports `0 kWh` (representing a grid without energy storage).
- `battery_max_power_kw`: $\ge 0.0$ and finite.
- `battery_c_rate`: $> 0.0$ and $\le 5.0$.
- `battery_efficiency_pct`: $> 0.0$ and $\le 100.0$.
- `reserve_margin_pct`: $\ge 0.0$ and $\le 100.0$.
- `min_soc_pct`: $\ge 0.0$ and $< 100.0$.
- `max_soc_pct`: $> \text{min\_soc\_pct}$ and $\le 100.0$.
- `initial_soc_pct`: Must be bounded between `min_soc_pct` and `max_soc_pct`.
- `low_soc_warning_pct`: Must be bounded between `min_soc_pct` and `max_soc_pct`.

---

## 7. Security & Secret Management

1. **Zero Client Secrets**: No API keys, OAuth tokens, or client credentials are ever sent to the browser or stored in frontend local storage.
2. **Server-Side Proxy**: QuickML integration requests are routed exclusively through `/api/quickml/predict` and `/api/quickml/test-connection`.
3. **SSRF Protection**: Connection test routes ignore client-supplied URLs and strictly test the server's configured environment endpoint.
4. **Header Cleanliness**: Only standard `Authorization: Zoho-oauthtoken ...` and `X-Catalyst-Project-Id` headers are transmitted. Unsupported headers (`X-Catalyst-Endpoint-Key`) have been removed.

---

## 8. Dual Python Prediction Contracts (`predict.py`)

### Interface A — Starter Kit Function Contract:
```python
from predict import predict
import pandas as pd

df = pd.read_csv('sample_test_input.csv')
predictions = predict(df) # Returns list or numpy array of floats
```

### Interface B — CLI Pipeline Contract:
```bash
python3 predict.py --input sample_test_input.csv --output sample_predictions.csv
```

### Evaluation Verification:
```bash
python3 test_eval.py sample_test_input.csv sample_predictions.csv
```
*Result: 3 / 3 test suites passed (Interface A, Interface B, and error handling).*

---

## 9. Application Lifecycle & API Reference

GridMind tracks an explicit lifecycle readiness state: `initializing` $\to$ `ready` (or `degraded` if running on local Ridge fallback when QuickML credentials are unconfigured).

| Endpoint | Method | Status | Description |
| :--- | :--- | :--- | :--- |
| `/api/health` | GET | 200 | Truthful lifecycle status, readiness, dataset stats, persistence health, and QuickML status. |
| `/api/grid/status` | GET | 200 | Current operating mode, peak load, overload status, and active battery SoC. |
| `/api/forecast/latest` | GET | 200 | Returns the latest persisted 24-hour forecast with battery simulation and grid decisions. |
| `/api/forecast` | POST | 200/400 | Validates incoming telemetry and computes 24-hour demand forecast. Rejects invalid data with HTTP 400. |
| `/api/telemetry/upload` | POST | 200/400 | Ingests CSV telemetry with RFC 4180 parsing, range validation, and persistence. |
| `/api/battery/config` | POST | 200/400 | Validates and persists battery capacity, power limits, and reserve thresholds. |
| `/api/evaluate` | POST | 200 | Executes TimeSeriesSplit chronological walk-forward cross-validation. |
| `/api/quickml/test-connection` | POST | 200 | Tests QuickML endpoint reachability with server-side SSRF protection. |
| `/api/quickml/config` | GET | 200 | Returns safe QuickML integration status (no secrets disclosed). |
| `/api/ml/explainability` | GET | 200 | Computes Permutation Feature Importance ($\Delta$RMSE). |
| `/api/ml/uncertainty` | GET | 200 | Computes distribution-free Conformal Prediction intervals (80%, 90%, 95%). |
| `/api/ml/peak-intelligence` | GET | 200 | Evaluates diurnal peak, base load, load factor, and ramp rates with threshold parity. |
| `/api/ml/what-if` | POST | 200 | Simulates side-by-side battery capacity scenarios. |

---

## 10. Local Development & Testing Instructions

### 1. Prerequisites:
- Node.js $\ge 22.0.0$ (supporting native TypeScript type stripping)
- Python $\ge 3.10$

### 2. Install Dependencies:
```bash
npm install
```

### 3. Run Automated Unit Tests (14 test suites, 62 unit tests):
```bash
npm test
```

### 4. Run TypeScript Type Checking / Linting:
```bash
npm run lint
```

### 5. Build for Production:
```bash
npm run build
```

### 6. Start Production Server:
```bash
npm start
# Equivalent to: node server.ts
```

### 7. Run Python Evaluation Suite:
```bash
python3 test_eval.py sample_test_input.csv sample_predictions.csv
```

---

## 11. Zoho Catalyst Deployment Architecture

GridMind can be deployed to Zoho Catalyst in two distinct modes depending on requirements:

### Mode A: Full-Stack Deployment via Catalyst AppSail (Recommended)
AppSail is Zoho Catalyst's standalone PaaS service for containerized Node.js applications. It runs the Express server (`server.ts`), serves all API routes (`/api/*`), handles server-side QuickML communication, and serves the built Vite frontend (`dist/`):

1. **Configuration**: Configured in `catalyst.json` under `"appsail"`:
   - Build command: `npm run build`
   - Run command: `npm start` (`node server.ts`)
   - Environment variables: `PORT=3000`, `NODE_ENV=production`
2. **Deploy via Catalyst CLI**:
   ```bash
   # Install CLI & Login
   npm install -g zcatalyst-cli
   catalyst login

   # Deploy full-stack service to AppSail
   catalyst deploy --only appsail
   ```

### Mode B: Static Frontend Hosting via Catalyst Web Client
If deploying through the Catalyst Web Client Console ("React + Vite" framework):
- **Console Settings**:
  - Runtime: Node 22
  - Install Command: `npm install`
  - Build Command: `npm run build`
  - Build Path: `dist`
- **Notice on Static Hosting**: Catalyst Web Client Hosting serves only the static assets in `dist/`. In this static mode, the Express server process is not started. The frontend operates with client-side simulation and fallback telemetry. For live server-side QuickML integration, deploy via AppSail (Mode A).

---

## 12. Dependency Resolution & Lockfile Integrity

- **Vite & esbuild Compatibility**: `vite@^8.3.0` specifies `peerOptional: { "esbuild": "^0.27.0 || ^0.28.0" }`. Specifying `"esbuild": "^0.28.0"` in `devDependencies` ensures deterministic resolution on npm without `--force` or `--legacy-peer-deps`.
- **Pre-Generated `package-lock.json`**: An exact `lockfileVersion: 3` lockfile is bundled with the project to guarantee identical, repeatable dependency installation on Zoho Catalyst build environments.

---

## 13. Known Limitations & Credential Status

1. **Zoho Catalyst QuickML Live Endpoint**: In the demonstration environment without live Zoho OAuth credentials, GridMind operates seamlessly on the verified local Ridge Regression model with full transparency (`forecast_source: "local_baseline_demo"`). When live credentials (`ZOHO_QUICKML_ENDPOINT`, `ZOHO_CATALYST_API_KEY`, `ZOHO_PROJECT_ID`) are configured in `.env`, GridMind automatically promotes to live QuickML inference.
2. **Physical Actuation**: GridMind is a decision-support, forecasting, and simulation platform; it does not issue direct electrical switching signals to physical grid hardware.

