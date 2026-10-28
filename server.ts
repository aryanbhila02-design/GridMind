/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * GridMind — Secure Full-Stack Express Server with Zoho Catalyst & QuickML Endpoints
 * HackSphere 3.0: Smart Energy Microgrid & Demand Forecasting Engine
 * 
 * Security & Architecture Principles:
 * - Zero client secret exposure (tokens strictly read from server environment)
 * - SSRF protection on QuickML testing
 * - File-backed persistence surviving server restarts
 * - Transparent forecast labeling (live_quickml vs fallback)
 * - Strict preservation of official HackSphere 3.0 controller rules
 */

import type { Request, Response } from 'express';
import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

import {
  executeForecast,
  evaluateTimeSeriesChronological,
  generateSynthetic24HourTelemetry,
} from './src/engine/forecastingEngine.ts';
import {
  validateTelemetryCsv,
  exportPredictionsToCsv,
} from './src/engine/telemetryValidator.ts';
import type {
  BatteryConfig,
  ForecastHourResult,
  ForecastingMode,
  GridEvent,
  QuickMLConfig,
  QuickMLConnectionStatus,
  TelemetryRow,
} from './src/types/microgrid.ts';
import { BatterySimulator, DEFAULT_BATTERY_CONFIG } from './src/engine/batterySimulator.ts';
import { validateBatteryConfig } from './src/engine/batteryValidator.ts';
import { evaluateGridTransition } from './src/engine/transitionController.ts';
import {
  savePersistedTelemetry,
  loadPersistedTelemetry,
  savePersistedForecast,
  loadPersistedForecast,
  savePersistedEvents,
  loadPersistedEvents,
  savePersistedSettings,
  loadPersistedSettings,
  isPersistenceLayerOperational,
} from './src/engine/persistenceService.ts';
import {
  getServerQuickMLConfig,
  isQuickMLConfigured,
  invokeQuickMLBatch,
} from './src/engine/quickmlClient.ts';
import {
  trainGradientDescent,
  trainPerceptronClassifier,
  trainNeuralNetworkMLP,
  getExperimentLog,
  recordExperiment,
} from './src/engine/mlLabEngine.ts';
import { computePermutationImportance } from './src/engine/explainabilityEngine.ts';
import { computeConformalUncertainty } from './src/engine/uncertaintyEngine.ts';
import { computePeakIntelligence } from './src/engine/peakIntelligenceEngine.ts';
import { runWhatIfScenarios } from './src/engine/whatIfEngine.ts';
import { fetchLiveWeatherTelemetry } from './src/engine/weatherEngine.ts';
import {
  getNationalGridReport,
  getAllNationalGridReports,
  getForecastingSuitabilityAudit,
  exportNationalGridToCsv,
  ORDERED_REPORT_KEYS,
} from './src/data/nationalGridData.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

// Standard CORS & Security Headers for Zoho Catalyst Slate & AppSail integration
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
    return;
  }
  next();
});

// Resilient file locator checking __dirname, process.cwd(), and parent directories
function resolveDataFilePath(filename: string): string {
  const candidates = [
    path.join(__dirname, filename),
    path.join(process.cwd(), filename),
    path.join(__dirname, '..', filename),
    path.resolve(filename),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return path.join(__dirname, filename);
}

// Application Lifecycle State & Health
export type AppLifecycleState = 'initializing' | 'ready' | 'degraded' | 'failed';
let appLifecycleState: AppLifecycleState = 'initializing';
let initializationError: string | null = null;

// Application In-Memory State (initialized from persistent storage)
let latestForecastResults: ForecastHourResult[] = [];
let latestTelemetryInput: TelemetryRow[] = [];
let latestForecastMetadata = {
  timestamp: new Date().toISOString(),
  modelUsed: 'Trained Ridge Regression (Held-Out Evaluated on Official Telemetry)',
  isQuickMlLive: false,
  forecastingMode: 'local_baseline_demo' as ForecastingMode,
  notes: 'System initialized. Ready for telemetry ingestion or live QuickML invocation.',
};

let activeBatteryConfig: BatteryConfig = { ...DEFAULT_BATTERY_CONFIG };
let lastQuickmlTestedAt: string | null = null;
let lastQuickmlStatusMessage = 'Zoho Catalyst QuickML endpoint not verified.';
let quickmlConnectionStatus: QuickMLConnectionStatus = isQuickMLConfigured()
  ? 'config_incomplete'
  : 'not_configured';

let eventHistory: GridEvent[] = [
  {
    id: 'evt-init',
    timestamp: new Date().toISOString(),
    type: 'FORECAST_RUN',
    severity: 'info',
    title: 'GridMind Engine Initialized',
    detail: 'GridMind Smart Energy Microgrid & Demand Forecasting Engine initialized in HackSphere 3.0 runtime.',
  },
];

/**
 * Loads persisted data from disk on startup, or falls back to bundled sample telemetry
 */
async function initializeApplicationState(): Promise<void> {
  try {
    appLifecycleState = 'initializing';

    // 1. Load persisted settings
    const persistedSettings = loadPersistedSettings();
    if (persistedSettings?.batteryConfig) {
      activeBatteryConfig = { ...DEFAULT_BATTERY_CONFIG, ...persistedSettings.batteryConfig };
    }

    // 2. Load persisted events
    const persistedEvents = loadPersistedEvents();
    if (persistedEvents.length > 0) {
      eventHistory = persistedEvents;
    }

    // 3. Load persisted telemetry & forecast
    const persistedRows = loadPersistedTelemetry();
    const persistedForecast = loadPersistedForecast();

    if (persistedRows.length > 0 && persistedForecast?.results?.length) {
      latestTelemetryInput = persistedRows;
      latestForecastResults = persistedForecast.results;
      latestForecastMetadata = {
        ...persistedForecast.metadata,
        forecastingMode: persistedForecast.metadata.forecastingMode as ForecastingMode,
      };
      appLifecycleState = latestForecastMetadata.isQuickMlLive ? 'ready' : 'degraded';
      return;
    }

    // 4. Fallback seed from sample_test_input.csv or train_telemetry.csv
    const samplePath = resolveDataFilePath('sample_test_input.csv');
    let seedRows: TelemetryRow[] = [];

    if (fs.existsSync(samplePath)) {
      try {
        const content = fs.readFileSync(samplePath, 'utf-8');
        const { validRecords } = validateTelemetryCsv(content);
        if (validRecords.length > 0) {
          seedRows = validRecords;
        }
      } catch {
        // Fallback below
      }
    }

    if (seedRows.length === 0) {
      seedRows = generateSynthetic24HourTelemetry();
    }

    latestTelemetryInput = seedRows;
    const sim = new BatterySimulator(activeBatteryConfig);
    const qmlConfig = {
      endpointUrl: getServerQuickMLConfig().endpointUrl,
      apiKey: getServerQuickMLConfig().apiKey,
      projectId: getServerQuickMLConfig().projectId,
      catalystEnv: getServerQuickMLConfig().catalystEnv as 'development' | 'production',
      timeoutMs: getServerQuickMLConfig().timeoutMs,
      isConnected: false,
      connectionStatus: quickmlConnectionStatus,
    };

    const output = await executeForecast(seedRows, qmlConfig, sim);
    latestForecastResults = output.results;
    latestForecastMetadata = {
      timestamp: output.executionTimestamp,
      modelUsed: output.modelUsed,
      isQuickMlLive: output.isQuickMlLive,
      forecastingMode: output.forecastingMode,
      notes: output.notes,
    };

    // Persist seeded state
    savePersistedTelemetry(seedRows);
    savePersistedForecast({
      metadata: latestForecastMetadata,
      results: latestForecastResults,
    });

    appLifecycleState = output.isQuickMlLive ? 'ready' : 'degraded';
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    initializationError = msg;
    appLifecycleState = 'failed';
    console.error('[GridMind] Application initialization error:', err);
  }
}

// ==========================================
// CORE API ROUTES
// ==========================================

// Truthful Health & Readiness Check Endpoint (Phase 12 & 14)
app.get('/api/health', (_req: Request, res: Response) => {
  const qmlConfig = getServerQuickMLConfig();
  const configured = Boolean(qmlConfig.endpointUrl && qmlConfig.endpointUrl.startsWith('http'));

  res.json({
    status: 'ok',
    service: 'GridMind',
    version: '3.0.0-hacksphere',
    timestamp: new Date().toISOString(),
    readiness: appLifecycleState,
    initializationError,
    quickmlConfigured: configured,
    quickmlStatus: quickmlConnectionStatus,
    lastQuickmlTestTimestamp: lastQuickmlTestedAt,
    loadedDatasetRows: latestTelemetryInput.length,
    predictionSource: latestForecastMetadata.isQuickMlLive ? 'live_quickml' : 'fallback',
    currentForecastSource: latestForecastMetadata.forecastingMode,
    modelUsed: latestForecastMetadata.modelUsed,
    persistenceAvailable: isPersistenceLayerOperational(),
    batteryCapacityKwh: activeBatteryConfig.capacity_kwh,
  });
});

// Grid Status Endpoint
app.get('/api/grid/status', (_req: Request, res: Response) => {
  const current = latestForecastResults[0] || {
    timestamp: new Date().toISOString(),
    hour: new Date().getHours(),
    predicted_demand_kw: 109.2,
    grid_mode: 'Main Grid Mode',
    critical_overload: false,
    decision_reason: 'Nominal operation',
    battery_soc_pct: activeBatteryConfig.initial_soc_pct,
    battery_soc_kwh: (activeBatteryConfig.initial_soc_pct / 100) * activeBatteryConfig.capacity_kwh,
    battery_power_kw: 0,
    grid_import_kw: 109.2,
    solar_used_kw: 0,
    solar_curtailed_kw: 0,
  };

  const peakDemand = latestForecastResults.reduce(
    (max, r) => (r.predicted_demand_kw > max.demand ? { demand: r.predicted_demand_kw, time: r.timestamp } : max),
    { demand: 0, time: '' }
  );

  const modeCounts = latestForecastResults.reduce(
    (acc, r) => {
      if (r.grid_mode === 'Renewable Storage Mode') acc.renewableStorage++;
      else acc.mainGrid++;
      return acc;
    },
    { renewableStorage: 0, mainGrid: 0 }
  );

  const overloadCount = latestForecastResults.filter(r => r.critical_overload).length;

  res.json({
    currentMode: current.grid_mode,
    latestDemand: current.predicted_demand_kw,
    peakDemand: peakDemand.demand,
    peakTimestamp: peakDemand.time,
    forecastHoursCount: latestForecastResults.length,
    hoursInRenewableStorage: modeCounts.renewableStorage,
    hoursInMainGrid: modeCounts.mainGrid,
    criticalOverloadAlerts: overloadCount,
    batterySocPct: current.battery_soc_pct,
    batterySocKwh: current.battery_soc_kwh,
    quickMlConnected: quickmlConnectionStatus === 'connected_verified',
    quickMlStatus: quickmlConnectionStatus,
    predictionSource: latestForecastMetadata.isQuickMlLive ? 'live_quickml' : 'fallback',
    lastForecastTimestamp: latestForecastMetadata.timestamp,
    modelUsed: latestForecastMetadata.modelUsed,
    forecastingMode: latestForecastMetadata.forecastingMode,
    notes: latestForecastMetadata.notes,
  });
});

// Latest Forecast Endpoint
app.get('/api/forecast/latest', (_req: Request, res: Response) => {
  res.json({
    success: true,
    source: latestForecastMetadata.isQuickMlLive ? 'live_quickml' : 'fallback',
    isQuickMlLive: latestForecastMetadata.isQuickMlLive,
    forecastingMode: latestForecastMetadata.forecastingMode,
    metadata: latestForecastMetadata,
    forecast: latestForecastResults,
    batteryConfig: activeBatteryConfig,
  });
});

// Execute Forecast on Provided Telemetry (Phase 4 & 5: Rigorous validation & secret isolation)
app.post('/api/forecast', async (req: Request, res: Response) => {
  try {
    if (!req.body || typeof req.body !== 'object') {
      res.status(400).json({ success: false, error: 'Request body must be a valid JSON object.' });
      return;
    }

    const { telemetryRows, batteryConfig } = req.body;

    // 1. Validate batteryConfig if supplied
    if (batteryConfig !== undefined) {
      const batVal = validateBatteryConfig(batteryConfig, activeBatteryConfig);
      if (!batVal.isValid || !batVal.config) {
        res.status(400).json({
          success: false,
          error: 'Invalid battery configuration parameters.',
          details: batVal.errors,
        });
        return;
      }
      activeBatteryConfig = batVal.config;
      savePersistedSettings({ batteryConfig: activeBatteryConfig, lastUpdated: new Date().toISOString() });
    }

    // 2. Validate telemetryRows
    let rows: TelemetryRow[] = [];
    if (telemetryRows !== undefined) {
      if (!Array.isArray(telemetryRows)) {
        res.status(400).json({ success: false, error: 'Field "telemetryRows" must be an array of objects.' });
        return;
      }
      if (telemetryRows.length === 0) {
        res.status(400).json({ success: false, error: 'Field "telemetryRows" cannot be empty.' });
        return;
      }
      if (telemetryRows.length > 720) {
        res.status(400).json({ success: false, error: 'telemetryRows length exceeds maximum limit of 720 hours.' });
        return;
      }

      for (let idx = 0; idx < telemetryRows.length; idx++) {
        const r = telemetryRows[idx];
        if (!r || typeof r !== 'object') {
          res.status(400).json({ success: false, error: `Row ${idx} must be a valid object.` });
          return;
        }

        if (!r.timestamp || typeof r.timestamp !== 'string') {
          res.status(400).json({ success: false, error: `Row ${idx} missing or invalid "timestamp" string.` });
          return;
        }
        const d = new Date(r.timestamp);
        if (isNaN(d.getTime())) {
          res.status(400).json({ success: false, error: `Row ${idx} contains invalid timestamp date: "${r.timestamp}".` });
          return;
        }

        const temp = Number(r.ambient_temp_c);
        if (isNaN(temp) || !Number.isFinite(temp) || temp < -60 || temp > 65) {
          res.status(400).json({
            success: false,
            error: `Row ${idx} invalid ambient_temp_c (${r.ambient_temp_c}). Must be between -60°C and 65°C.`,
          });
          return;
        }

        const hum = Number(r.humidity_pct);
        if (isNaN(hum) || !Number.isFinite(hum) || hum < 0 || hum > 100) {
          res.status(400).json({
            success: false,
            error: `Row ${idx} invalid humidity_pct (${r.humidity_pct}). Must be between 0% and 100%.`,
          });
          return;
        }

        const sol = Number(r.solar_irradiance_wm2);
        if (isNaN(sol) || !Number.isFinite(sol) || sol < 0 || sol > 2000) {
          res.status(400).json({
            success: false,
            error: `Row ${idx} invalid solar_irradiance_wm2 (${r.solar_irradiance_wm2}). Must be between 0 and 2000 W/m².`,
          });
          return;
        }

        rows.push({
          timestamp: r.timestamp,
          ambient_temp_c: temp,
          humidity_pct: hum,
          solar_irradiance_wm2: sol,
          ...(r.target_demand_kw !== undefined && Number.isFinite(Number(r.target_demand_kw))
            ? { target_demand_kw: Number(r.target_demand_kw) }
            : {}),
        });
      }
    } else {
      if (latestTelemetryInput.length > 0) {
        rows = latestTelemetryInput;
      } else {
        res.status(400).json({ success: false, error: 'No telemetry provided and no prior telemetry loaded.' });
        return;
      }
    }

    // Server-side QuickML credentials (client overrides strictly ignored)
    const serverQml = getServerQuickMLConfig();
    const qmlConfig: QuickMLConfig = {
      endpointUrl: serverQml.endpointUrl,
      apiKey: serverQml.apiKey,
      projectId: serverQml.projectId,
      catalystEnv: serverQml.catalystEnv as 'development' | 'production',
      timeoutMs: serverQml.timeoutMs,
      isConnected: quickmlConnectionStatus === 'connected_verified',
      connectionStatus: quickmlConnectionStatus,
    };

    const sim = new BatterySimulator(activeBatteryConfig);
    const output = await executeForecast(rows, qmlConfig, sim);

    latestForecastResults = output.results;
    latestTelemetryInput = rows;
    latestForecastMetadata = {
      timestamp: output.executionTimestamp,
      modelUsed: output.modelUsed,
      isQuickMlLive: output.isQuickMlLive,
      forecastingMode: output.forecastingMode,
      notes: output.notes,
    };

    // Save to persistence
    savePersistedTelemetry(rows);
    savePersistedForecast({
      metadata: latestForecastMetadata,
      results: latestForecastResults,
    });

    // Log event
    const newEvent: GridEvent = {
      id: `evt-fc-${Date.now()}`,
      timestamp: new Date().toISOString(),
      type: 'FORECAST_RUN',
      severity: output.forecastingMode === 'quickml_failed_fallback' ? 'warning' : 'info',
      title: 'Microgrid Forecast Completed',
      detail: `Generated ${output.results.length} hourly dispatch predictions via ${output.modelUsed}. Mode: ${output.forecastingMode}.`,
    };
    eventHistory.unshift(newEvent);
    savePersistedEvents(eventHistory);

    res.json({
      success: true,
      source: output.isQuickMlLive ? 'live_quickml' : 'fallback',
      isQuickMlLive: output.isQuickMlLive,
      forecastingMode: output.forecastingMode,
      metadata: latestForecastMetadata,
      results: output.results,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message || 'Forecast computation failed' });
  }
});

// CSV Upload & Telemetry Validation
app.post('/api/telemetry/upload', (req: Request, res: Response) => {
  try {
    const { csvText } = req.body;
    if (!csvText || typeof csvText !== 'string') {
      res.status(400).json({ success: false, error: 'Missing csvText string in request body' });
      return;
    }

    const { validRecords, summary } = validateTelemetryCsv(csvText);

    if (validRecords.length > 0) {
      latestTelemetryInput = validRecords;
      savePersistedTelemetry(validRecords);

      const uploadEvent: GridEvent = {
        id: `evt-up-${Date.now()}`,
        timestamp: new Date().toISOString(),
        type: 'DATA_UPLOAD',
        severity: summary.rejectedRows > 0 ? 'warning' : 'info',
        title: 'Telemetry Dataset Uploaded',
        detail: `Processed ${summary.totalRows} rows: ${summary.validRows} accepted, ${summary.rejectedRows} rejected.`,
      };
      eventHistory.unshift(uploadEvent);
      savePersistedEvents(eventHistory);
    }

    res.json({
      success: true,
      summary,
      validRecordsCount: validRecords.length,
      sampleRecords: validRecords.slice(0, 50),
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message || 'CSV Parsing error' });
  }
});

// TimeSeriesSplit Model Evaluation
app.post('/api/evaluate', (req: Request, res: Response) => {
  try {
    const { customCsvText } = req.body;
    let evalRows: TelemetryRow[] = [];

    if (customCsvText && typeof customCsvText === 'string') {
      const { validRecords } = validateTelemetryCsv(customCsvText, { requireTarget: true });
      evalRows = validRecords;
    } else {
      const trainPath = resolveDataFilePath('train_telemetry.csv');
      if (fs.existsSync(trainPath)) {
        const content = fs.readFileSync(trainPath, 'utf-8');
        const { validRecords } = validateTelemetryCsv(content, { requireTarget: true });
        evalRows = validRecords;
      }
    }

    if (evalRows.length === 0) {
      res.status(400).json({
        success: false,
        error: 'Evaluation requires telemetry data with "target_demand_kw" target column.',
      });
      return;
    }

    const evalResult = evaluateTimeSeriesChronological(evalRows, 5);
    res.json({
      success: true,
      totalObservations: evalRows.length,
      ...evalResult,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message || 'Model evaluation failed' });
  }
});

// Secure Server-Side QuickML Test Connection (Phase 2 & 3: SSRF protected, accepts GET & POST)
app.all('/api/quickml/test-connection', async (_req: Request, res: Response) => {
  const config = getServerQuickMLConfig();
  lastQuickmlTestedAt = new Date().toISOString();

  if (!config.endpointUrl) {
    quickmlConnectionStatus = 'not_configured';
    lastQuickmlStatusMessage = 'QuickML endpoint URL is not configured in server environment.';
    res.json({
      connected: false,
      status: 'not_configured' as QuickMLConnectionStatus,
      latencyMs: 0,
      message: 'Zoho Catalyst QuickML endpoint is not set. System is operating in Local Demonstration Mode using trained Ridge fallback.',
    });
    return;
  }

  // Generate test sample instance
  const sampleInstance: TelemetryRow[] = [
    {
      timestamp: new Date().toISOString(),
      ambient_temp_c: 24.5,
      humidity_pct: 55,
      solar_irradiance_wm2: 600,
    },
  ];

  const result = await invokeQuickMLBatch(sampleInstance, 8000);

  if (result.success && result.predictions && result.predictions.length > 0) {
    quickmlConnectionStatus = 'connected_verified';
    lastQuickmlStatusMessage = `Verified: QuickML returned valid prediction (${result.predictions[0].toFixed(2)} kW) in ${result.latencyMs}ms.`;

    eventHistory.unshift({
      id: `evt-qml-${Date.now()}`,
      timestamp: new Date().toISOString(),
      type: 'QUICKML_STATUS',
      severity: 'info',
      title: 'Zoho Catalyst QuickML Connected',
      detail: `Endpoint response successfully verified with ${result.latencyMs}ms latency.`,
    });
    savePersistedEvents(eventHistory);

    res.json({
      connected: true,
      status: 'connected_verified' as QuickMLConnectionStatus,
      latencyMs: result.latencyMs,
      httpStatus: result.httpStatus,
      message: lastQuickmlStatusMessage,
    });
  } else {
    quickmlConnectionStatus = 'connection_failed';
    lastQuickmlStatusMessage = result.error || 'Connection failed.';

    res.json({
      connected: false,
      status: 'connection_failed' as QuickMLConnectionStatus,
      latencyMs: result.latencyMs,
      httpStatus: result.httpStatus,
      message: `QuickML Verification Failed: ${result.error}`,
    });
  }
});

// Safe QuickML Configuration Status (Phase 2: Secrets never returned)
app.get('/api/quickml/config', (_req: Request, res: Response) => {
  const config = getServerQuickMLConfig();
  let hostDomain = '';
  try {
    if (config.endpointUrl) {
      hostDomain = new URL(config.endpointUrl).hostname;
    }
  } catch {
    // Ignore
  }

  res.json({
    isConfigured: isQuickMLConfigured(),
    hostDomain,
    catalystEnv: config.catalystEnv,
    connectionStatus: quickmlConnectionStatus,
    lastTestedAt: lastQuickmlTestedAt,
    statusMessage: lastQuickmlStatusMessage,
  });
});

// Battery Configuration Updates (Persisted with authoritative validation - Phase 6)
app.post('/api/battery/config', (req: Request, res: Response) => {
  try {
    const { config } = req.body;
    const validation = validateBatteryConfig(config, activeBatteryConfig);

    if (!validation.isValid || !validation.config) {
      res.status(400).json({
        success: false,
        error: 'Invalid battery configuration parameters.',
        details: validation.errors,
      });
      return;
    }

    activeBatteryConfig = validation.config;

    savePersistedSettings({
      batteryConfig: activeBatteryConfig,
      lastUpdated: new Date().toISOString(),
    });

    eventHistory.unshift({
      id: `evt-bat-${Date.now()}`,
      timestamp: new Date().toISOString(),
      type: 'MODE_CHANGE',
      severity: 'info',
      title: 'Battery Configuration Updated',
      detail: `Capacity: ${activeBatteryConfig.capacity_kwh} kWh, Limits: +${activeBatteryConfig.max_charge_kw} kW / -${activeBatteryConfig.max_discharge_kw} kW.`,
    });
    savePersistedEvents(eventHistory);

    res.json({
      success: true,
      batteryConfig: activeBatteryConfig,
      message: 'Battery configuration validated and saved successfully.',
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
});

// Events History
app.get('/api/grid/events', (_req: Request, res: Response) => {
  res.json({
    events: eventHistory.slice(0, 100),
  });
});

// National Grid Reference Data (from electric Data.pdf - CEA Peak Power Supply Position)
app.get('/api/grid/national-power-position', (req: Request, res: Response) => {
  const month = typeof req.query.month === 'string' ? req.query.month : undefined;
  const audit = getForecastingSuitabilityAudit();

  if (month) {
    const report = getNationalGridReport(month);
    res.json({
      success: true,
      report,
      audit,
      availableMonths: ORDERED_REPORT_KEYS,
    });
  } else {
    const allReports = getAllNationalGridReports();
    res.json({
      success: true,
      reports: allReports,
      audit,
      availableMonths: ORDERED_REPORT_KEYS,
    });
  }
});

// CSV Export for National Grid Reference Data
app.get('/api/grid/national-power-position/export', (req: Request, res: Response) => {
  const month = typeof req.query.month === 'string' ? req.query.month : '2026-05';
  const csvData = exportNationalGridToCsv(month);
  res.setHeader('Content-Type', 'text/csv; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename="CEA_National_Power_Supply_${month}.csv"`);
  res.send(csvData);
});

// Sample Data Provider
app.get('/api/sample-data/:type', (req: Request, res: Response) => {
  const type = req.params.type;
  let filename = 'sample_test_input.csv';
  if (type === 'train') filename = 'train_telemetry.csv';
  else if (type === 'predictions') filename = 'sample_predictions.csv';

  const filePath = resolveDataFilePath(filename);
  if (fs.existsSync(filePath)) {
    const content = fs.readFileSync(filePath, 'utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.type('text/csv').send(content);
  } else {
    res.status(404).send('Sample file not found');
  }
});

// ==========================================
// ML RESEARCH LAB & ADVANCED INTELLIGENCE ROUTES
// ==========================================

// P1: Gradient Descent Linear Regressor Training
app.post('/api/ml/train-gradient-descent', (req: Request, res: Response) => {
  try {
    const { learningRate, epochs, batchSize } = req.body;
    const trainPath = resolveDataFilePath('train_telemetry.csv');
    let trainRows = latestTelemetryInput.filter(r => r.target_demand_kw !== undefined);

    if (trainRows.length < 5 && fs.existsSync(trainPath)) {
      const content = fs.readFileSync(trainPath, 'utf-8');
      const { validRecords } = validateTelemetryCsv(content, { requireTarget: true });
      trainRows = validRecords;
    }

    const result = trainGradientDescent(trainRows, {
      learningRate: Number(learningRate) || 0.05,
      epochs: Number(epochs) || 200,
      batchSize: Number(batchSize) || 16,
    });

    recordExperiment({
      modelType: 'GradientDescent',
      modelName: 'Gradient Descent Linear Regressor',
      hyperparameters: {
        learningRate: result.learningRate,
        epochs: result.epochsRun,
        batchSize: result.batchSize,
      },
      metrics: {
        rmse: result.valRmse,
        r2: result.r2,
      },
      durationMs: result.durationMs,
      status: 'completed',
      notes: `Batch GD with normalized features. Train RMSE: ${result.trainRmse} kW, Val RMSE: ${result.valRmse} kW.`,
    });

    res.json({ success: true, result });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
});

// P1: Single Perceptron Classifier Training
app.post('/api/ml/train-perceptron', (req: Request, res: Response) => {
  try {
    const { learningRate, epochs, thresholdKw } = req.body;
    const trainPath = resolveDataFilePath('train_telemetry.csv');
    let trainRows = latestTelemetryInput.filter(r => r.target_demand_kw !== undefined);

    if (trainRows.length < 5 && fs.existsSync(trainPath)) {
      const content = fs.readFileSync(trainPath, 'utf-8');
      const { validRecords } = validateTelemetryCsv(content, { requireTarget: true });
      trainRows = validRecords;
    }

    const result = trainPerceptronClassifier(trainRows, {
      learningRate: Number(learningRate) || 0.05,
      epochs: Number(epochs) || 100,
      thresholdKw: Number(thresholdKw) || 160.0,
    });

    recordExperiment({
      modelType: 'Perceptron',
      modelName: 'Single Perceptron Binary Classifier',
      hyperparameters: {
        learningRate: Number(learningRate) || 0.05,
        thresholdKw: result.thresholdKw,
        epochs: result.epochsRun,
      },
      metrics: {
        accuracy: result.accuracy,
      },
      durationMs: 15,
      status: 'completed',
      notes: `Separates demand >= ${result.thresholdKw} kW. Accuracy: ${(result.accuracy * 100).toFixed(1)}%, F1: ${result.f1Score}.`,
    });

    res.json({ success: true, result });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
});

// P1: Neural Network MLP Regressor Training
app.post('/api/ml/train-neural-net', (req: Request, res: Response) => {
  try {
    const { learningRate, epochs, batchSize } = req.body;
    const trainPath = resolveDataFilePath('train_telemetry.csv');
    let trainRows = latestTelemetryInput.filter(r => r.target_demand_kw !== undefined);

    if (trainRows.length < 5 && fs.existsSync(trainPath)) {
      const content = fs.readFileSync(trainPath, 'utf-8');
      const { validRecords } = validateTelemetryCsv(content, { requireTarget: true });
      trainRows = validRecords;
    }

    const result = trainNeuralNetworkMLP(trainRows, {
      learningRate: Number(learningRate) || 0.005,
      epochs: Number(epochs) || 180,
      batchSize: Number(batchSize) || 16,
    });

    recordExperiment({
      modelType: 'MLP',
      modelName: 'Deep MLP Regressor (5-16-8-1)',
      hyperparameters: {
        learningRate: result.learningRate,
        epochs: result.epochsRun,
        batchSize: result.batchSize,
        architecture: result.architecture,
      },
      metrics: {
        rmse: result.valRmse,
        mae: result.valMae,
        r2: result.r2,
      },
      durationMs: result.durationMs,
      status: 'completed',
      notes: `Backpropagation with Xavier init. Finite diff check error: ${result.finiteDiffGradError}. Val RMSE: ${result.valRmse} kW.`,
    });

    res.json({ success: true, result });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
});

// P1: Model Experiment Log
app.get('/api/ml/experiments', (_req: Request, res: Response) => {
  res.json({ success: true, experiments: getExperimentLog() });
});

// P2: Prediction Explainability (Permutation Feature Importance)
const handleExplainability = (req: Request, res: Response) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    let trainRows: TelemetryRow[] = [];
    if (req.body?.telemetryRows && Array.isArray(req.body.telemetryRows) && req.body.telemetryRows.length >= 10) {
      trainRows = req.body.telemetryRows;
    } else {
      trainRows = latestTelemetryInput.filter(r => r.target_demand_kw !== undefined);
      const trainPath = resolveDataFilePath('train_telemetry.csv');
      if (trainRows.length < 10 && fs.existsSync(trainPath)) {
        const content = fs.readFileSync(trainPath, 'utf-8');
        const { validRecords } = validateTelemetryCsv(content, { requireTarget: true });
        trainRows = validRecords;
      }
    }

    if (trainRows.length < 10) {
      res.status(400).json({
        success: false,
        error: 'Permutation feature importance requires at least 10 labelled telemetry rows with target_demand_kw.',
      });
      return;
    }

    const result = computePermutationImportance(trainRows, 5);
    res.json({ success: true, ...result });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
};
app.get('/api/ml/explainability', handleExplainability);
app.post('/api/ml/explainability', handleExplainability);

// P2: Peak Demand Intelligence
const handlePeakIntelligence = (req: Request, res: Response) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    const forecastRows = (req.body?.forecastResults && Array.isArray(req.body.forecastResults))
      ? req.body.forecastResults
      : latestForecastResults;
    const summary = computePeakIntelligence(forecastRows);
    res.json({ success: true, summary });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
};
app.get('/api/ml/peak-intelligence', handlePeakIntelligence);
app.post('/api/ml/peak-intelligence', handlePeakIntelligence);

// P3: Conformal Prediction Uncertainty Analysis
const handleUncertainty = (req: Request, res: Response) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    let calRows: TelemetryRow[] = [];
    if (req.body?.calibrationRows && Array.isArray(req.body.calibrationRows) && req.body.calibrationRows.length >= 5) {
      calRows = req.body.calibrationRows;
    } else {
      calRows = latestTelemetryInput.filter(r => r.target_demand_kw !== undefined);
      const trainPath = resolveDataFilePath('train_telemetry.csv');
      if (calRows.length < 5 && fs.existsSync(trainPath)) {
        const content = fs.readFileSync(trainPath, 'utf-8');
        const { validRecords } = validateTelemetryCsv(content, { requireTarget: true });
        calRows = validRecords;
      }
    }

    const forecastRows = (req.body?.forecastResults && Array.isArray(req.body.forecastResults))
      ? req.body.forecastResults
      : latestForecastResults;

    const result = computeConformalUncertainty(calRows, forecastRows);
    res.json({ success: true, ...result });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
};
app.get('/api/ml/uncertainty', handleUncertainty);
app.post('/api/ml/uncertainty', handleUncertainty);

// P3: Weather-Aware Telemetry Ingestion (Open-Meteo with fallback)
app.get('/api/weather/forecast', async (req: Request, res: Response) => {
  try {
    const lat = Number(req.query.lat) || 28.6139;
    const lon = Number(req.query.lon) || 77.2090;

    const weatherData = await fetchLiveWeatherTelemetry(lat, lon);
    res.json({ success: true, weather: weatherData });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
});

// P4: What-If Energy Scenario Simulation
app.post('/api/ml/what-if', (req: Request, res: Response) => {
  try {
    const { scenarios } = req.body;

    const customConfigs = Array.isArray(scenarios) && scenarios.length > 0
      ? scenarios
      : [
          {
            name: 'Active Baseline Configuration',
            config: activeBatteryConfig,
          },
          {
            name: 'High Capacity Buffer (1000 kWh, 250 kW)',
            config: {
              ...activeBatteryConfig,
              capacity_kwh: 1000,
              max_charge_kw: 200,
              max_discharge_kw: 250,
            },
          },
          {
            name: 'Degraded Reserve (250 kWh, 60 kW)',
            config: {
              ...activeBatteryConfig,
              capacity_kwh: 250,
              max_charge_kw: 50,
              max_discharge_kw: 60,
              min_soc_pct: 25,
            },
          },
        ];

    const results = runWhatIfScenarios(latestForecastResults, customConfigs);
    res.json({
      success: true,
      scenarios: results,
      notice: 'Hypothetical Scenario Simulation — Not a Live Grid State',
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ success: false, error: message });
  }
});

// Explicit API 404 Handler - Guarantees API requests never return HTML fallback
app.all('/api/*', (req: Request, res: Response) => {
  res.status(404).setHeader('Content-Type', 'application/json; charset=utf-8').json({
    success: false,
    error: `API route not found: ${req.method} ${req.originalUrl}`,
    status: 404,
  });
});

// ==========================================
// VITE DEV MIDDLEWARE & PRODUCTION STATIC SERVING
// ==========================================
async function startServer() {
  await initializeApplicationState();

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true' ? undefined : false,
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = fs.existsSync(path.join(__dirname, 'dist'))
      ? path.join(__dirname, 'dist')
      : fs.existsSync(path.join(__dirname, 'index.html'))
        ? __dirname
        : path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[GridMind] Secure Energy Microgrid Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
