// ===========================================================================
// Nexus - k6 Load Test Script
// ===========================================================================
// Scenarios: smoke, load, stress
// Run:  k6 run load-test.js
//       k6 run --env BASE_URL=http://nexus.example.com load-test.js
//       k6 run --env SCENARIO=smoke load-test.js
// ===========================================================================

import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend, Counter } from "k6/metrics";
import { randomIntBetween } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

// ---------------------------------------------------------------------------
// Custom Metrics
// ---------------------------------------------------------------------------
const inferenceLatency = new Trend("inference_latency", true);
const inferenceErrors = new Rate("inference_errors");
const inferenceRequests = new Counter("inference_requests_total");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";

const MODELS = [
  "gpt2-small",
  "gpt2-medium",
  "bert-base",
];

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------
export const options = {
  thresholds: {
    http_req_duration: ["p(95)<100"],       // 95th percentile < 100ms
    http_req_failed: ["rate<0.01"],          // Error rate < 1%
    inference_latency: ["p(95)<100", "p(99)<250"],
    inference_errors: ["rate<0.01"],
  },

  // -------------------------------------------------------------------------
  // Scenarios
  // -------------------------------------------------------------------------
  scenarios: {
    // -----------------------------------------------------------------------
    // Smoke Test: 1 VU, 30 seconds - basic sanity check
    // -----------------------------------------------------------------------
    smoke: {
      executor: "constant-vus",
      vus: 1,
      duration: "30s",
      tags: { scenario: "smoke" },
      exec: "inferenceTest",
    },

    // -----------------------------------------------------------------------
    // Load Test: ramp to 50 VUs, sustain 2 min, ramp down
    // -----------------------------------------------------------------------
    load: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "1m", target: 50 },    // Ramp up to 50 VUs over 1 min
        { duration: "2m", target: 50 },     // Sustain 50 VUs for 2 min
        { duration: "30s", target: 0 },     // Ramp down over 30s
      ],
      tags: { scenario: "load" },
      exec: "inferenceTest",
      startTime: "35s",                     // Start after smoke finishes
    },

    // -----------------------------------------------------------------------
    // Stress Test: 100 VUs - push system limits
    // -----------------------------------------------------------------------
    stress: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 100 },   // Ramp to 100 VUs
        { duration: "2m", target: 100 },     // Hold at 100
        { duration: "30s", target: 0 },      // Ramp down
      ],
      tags: { scenario: "stress" },
      exec: "inferenceTest",
      startTime: "5m",                       // Start after load finishes
    },
  },
};

// ---------------------------------------------------------------------------
// Helper: Generate random float array (simulated model input)
// ---------------------------------------------------------------------------
function generateRandomInput(size) {
  const input = [];
  for (let i = 0; i < size; i++) {
    input.push(Math.random() * 2.0 - 1.0);  // Range [-1.0, 1.0]
  }
  return input;
}

// ---------------------------------------------------------------------------
// Main Test Function: POST /v1/infer
// ---------------------------------------------------------------------------
export function inferenceTest() {
  const model = MODELS[randomIntBetween(0, MODELS.length - 1)];
  const inputSize = randomIntBetween(64, 512);

  const payload = JSON.stringify({
    model_id: model,
    input: generateRandomInput(inputSize),
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    tags: {
      model: model,
      endpoint: "infer",
    },
  };

  const startTime = Date.now();
  const res = http.post(`${BASE_URL}/v1/infer`, payload, params);
  const latency = Date.now() - startTime;

  // Record custom metrics
  inferenceLatency.add(latency, { model: model });
  inferenceRequests.add(1, { model: model });

  // Validate response
  const success = check(res, {
    "status is 200": (r) => r.status === 200,
    "response has output field": (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.output !== undefined;
      } catch (e) {
        return false;
      }
    },
    "response time < 200ms": (r) => r.timings.duration < 200,
    "no server errors": (r) => r.status < 500,
  });

  inferenceErrors.add(!success);

  // Brief pause between requests to simulate realistic traffic
  sleep(randomIntBetween(1, 3) / 10);  // 0.1 - 0.3s
}

// ---------------------------------------------------------------------------
// Setup: Verify cluster is healthy before testing
// ---------------------------------------------------------------------------
export function setup() {
  const healthRes = http.get(`${BASE_URL}/v1/health/ready`);

  const healthy = check(healthRes, {
    "cluster is healthy": (r) => r.status === 200,
  });

  if (!healthy) {
    console.error(
      `Cluster health check failed: status=${healthRes.status} body=${healthRes.body}`
    );
  }

  return {
    startTime: new Date().toISOString(),
    baseUrl: BASE_URL,
  };
}

// ---------------------------------------------------------------------------
// Teardown: Summary logging
// ---------------------------------------------------------------------------
export function teardown(data) {
  console.log(`Load test completed. Started at: ${data.startTime}`);
  console.log(`Target: ${data.baseUrl}`);
}
