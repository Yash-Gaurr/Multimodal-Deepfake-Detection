"""
Standalone Forensic Web Server for Multimodal Deepfake Detection.
Runs with zero external web dependencies using standard library http.server
and provides an interactive, modern dashboard for media analysis.
"""

import base64
import io
import json
import mimetypes
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.inference.pipeline import DeepfakeDetectionPipeline
from src.utils.config import get_config

# Global pipeline instance
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        ckpt = PROJECT_ROOT / "checkpoints" / "best_multimodal_model.pth"
        ckpt_path = str(ckpt) if ckpt.exists() else None
        print(f"[Server] Initializing pipeline on GPU/CPU (Checkpoint: {ckpt_path})...")
        PIPELINE = DeepfakeDetectionPipeline(checkpoint_path=ckpt_path)
    return PIPELINE


def plot_spectrogram_base64(mel_spectrogram: np.ndarray) -> str:
    """Renders spectrogram to base64 PNG image."""
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
    im = ax.imshow(
        mel_spectrogram,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_title("Extracted Log-Mel Spectrogram", fontsize=10, fontweight="bold")
    ax.set_xlabel("Temporal Frame Windows", fontsize=8)
    ax.set_ylabel("Mel Bins (80)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def image_to_base64(img_arr: np.ndarray) -> str:
    """Converts numpy image array to base64 PNG."""
    from PIL import Image
    im = Image.fromarray(img_arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Deepfake Forensic Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #0B0F19; color: #E2E8F0; }
        .glass-panel { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
        .accent-gradient { background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <header class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
            <div>
                <div class="flex items-center space-x-3">
                    <span class="p-2.5 rounded-xl bg-blue-600/20 text-blue-400 border border-blue-500/30 text-2xl">🛡️</span>
                    <div>
                        <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight text-white">Multimodal Deepfake Forensic Analyzer</h1>
                        <p class="text-sm text-gray-400">Audio-Visual Cross-Modal Inconsistency & Phoneme-Viseme Desynchronization Detection</p>
                    </div>
                </div>
            </div>
            <div class="flex items-center gap-3">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    ● GPU Accelerated (CUDA RTX 3050)
                </span>
            </div>
        </header>

        <!-- Controls Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <!-- Source Selector -->
            <div class="glass-panel p-5 rounded-2xl md:col-span-2 space-y-4">
                <h2 class="text-base font-semibold text-white flex items-center gap-2">
                    <span>🎬</span> Select Media File
                </h2>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-gray-400 mb-1">Benchmark Media Samples</label>
                        <select id="videoSelect" onchange="runAnalysis()" class="w-full bg-gray-900/80 border border-gray-700 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <!-- Populated dynamically -->
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-gray-400 mb-1">Architecture / Ablation Mode</label>
                        <select id="modeSelect" onchange="runAnalysis()" class="w-full bg-gray-900/80 border border-gray-700 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="multimodal">Proposed: Multimodal Cross-Attention</option>
                            <option value="visual_only">Ablation 1: Visual-Only (Video Baseline)</option>
                            <option value="audio_only">Ablation 2: Audio-Only (Audio Baseline)</option>
                            <option value="late_fusion">Ablation 3: Late Fusion (Score Concat)</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Action Card -->
            <div class="glass-panel p-5 rounded-2xl flex flex-col justify-between">
                <div>
                    <h2 class="text-base font-semibold text-white mb-1">⚡ Instant Diagnosis</h2>
                    <p class="text-xs text-gray-400">Deep neural inspection of spatial face textures and acoustic frequency alignment.</p>
                </div>
                <div class="flex gap-3 mt-4">
                    <button onclick="runAnalysis()" class="flex-1 accent-gradient hover:opacity-90 transition font-medium py-2.5 px-4 rounded-xl text-sm text-white shadow-lg shadow-blue-500/20 flex items-center justify-center gap-2">
                        <span>▶</span> Analyze Media
                    </button>
                    <button onclick="downloadReport()" class="bg-gray-800 hover:bg-gray-700 border border-gray-700 py-2.5 px-4 rounded-xl text-sm text-gray-200 transition">
                        📥 Report
                    </button>
                </div>
            </div>
        </div>

        <!-- Diagnosis Verdict Banner -->
        <div id="verdictBanner" class="hidden p-6 rounded-2xl border transition-all duration-300">
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div class="flex items-center space-x-4">
                    <div id="verdictIcon" class="w-14 h-14 rounded-2xl flex items-center justify-center text-3xl font-bold"></div>
                    <div>
                        <div class="text-xs font-bold tracking-wider uppercase text-gray-400">Forensic Decision</div>
                        <div id="verdictText" class="text-2xl font-black"></div>
                        <div id="verdictDetail" class="text-sm text-gray-400 mt-0.5"></div>
                    </div>
                </div>
                <div class="flex items-center gap-6">
                    <div class="text-right">
                        <div class="text-xs text-gray-400 font-medium">Confidence Score</div>
                        <div id="confidenceVal" class="text-3xl font-extrabold"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scores Breakdown -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="glass-panel p-4 rounded-xl">
                <div class="text-xs text-gray-400 font-medium">Fake Probability</div>
                <div id="scoreOverall" class="text-2xl font-bold text-white mt-1">--</div>
                <div class="text-[11px] text-gray-500 mt-1">Classification Posterior</div>
            </div>
            <div class="glass-panel p-4 rounded-xl">
                <div class="text-xs text-gray-400 font-medium">Visual Artifact Index</div>
                <div id="scoreVisual" class="text-2xl font-bold text-blue-400 mt-1">--</div>
                <div class="text-[11px] text-gray-500 mt-1">Spatial warping & blending</div>
            </div>
            <div class="glass-panel p-4 rounded-xl">
                <div class="text-xs text-gray-400 font-medium">Voice Acoustic Index</div>
                <div id="scoreAudio" class="text-2xl font-bold text-emerald-400 mt-1">--</div>
                <div class="text-[11px] text-gray-500 mt-1">Spectral phase anomalies</div>
            </div>
            <div class="glass-panel p-4 rounded-xl">
                <div class="text-xs text-gray-400 font-medium">Phoneme-Viseme Desync</div>
                <div id="scoreSync" class="text-2xl font-bold text-rose-400 mt-1">--</div>
                <div class="text-[11px] text-gray-500 mt-1">Speech vs Lip Mismatch</div>
            </div>
        </div>

        <!-- Visual & Acoustic Inspection -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Video & Face Inspection -->
            <div class="glass-panel p-5 rounded-2xl space-y-4">
                <h3 class="text-sm font-semibold text-white flex items-center justify-between">
                    <span>📹 Video Stream & Face Tracker</span>
                    <span class="text-xs font-normal text-gray-400" id="videoMeta">Target: 32 Frames</span>
                </h3>
                <div class="aspect-video bg-black rounded-xl overflow-hidden flex items-center justify-center border border-gray-800">
                    <video id="player" controls class="w-full h-full object-contain"></video>
                </div>
                <div class="grid grid-cols-2 gap-4 pt-2">
                    <div class="bg-gray-900/60 p-3 rounded-xl border border-gray-800 text-center">
                        <span class="text-xs text-gray-400 block mb-2">Detected Face Crop</span>
                        <img id="faceImg" class="w-24 h-24 mx-auto object-cover rounded-lg border border-gray-700 bg-gray-800" />
                    </div>
                    <div class="bg-gray-900/60 p-3 rounded-xl border border-gray-800 text-center">
                        <span class="text-xs text-gray-400 block mb-2">Extracted Lip ROI</span>
                        <img id="lipImg" class="w-24 h-24 mx-auto object-cover rounded-lg border border-gray-700 bg-gray-800" />
                    </div>
                </div>
            </div>

            <!-- Spectrogram & Timeline -->
            <div class="glass-panel p-5 rounded-2xl space-y-4 flex flex-col justify-between">
                <div>
                    <h3 class="text-sm font-semibold text-white mb-3">📊 Spectrogram & Acoustic Energy</h3>
                    <div class="bg-gray-950 p-2 rounded-xl border border-gray-800 flex items-center justify-center min-h-[160px]">
                        <img id="specImg" class="w-full h-auto object-contain rounded-lg" />
                    </div>
                </div>

                <div class="pt-2">
                    <h3 class="text-sm font-semibold text-white mb-2">📈 Temporal Phoneme-Viseme Inconsistency Timeline</h3>
                    <div class="bg-gray-950 p-3 rounded-xl border border-gray-800">
                        <canvas id="timelineChart" height="130"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Comparative Ablation Study Table -->
        <div class="glass-panel p-5 rounded-2xl space-y-4">
            <div class="flex items-center justify-between">
                <div>
                    <h3 class="text-base font-semibold text-white">⚖️ Head-to-Head Ablation Comparison</h3>
                    <p class="text-xs text-gray-400">Inspect how each individual modality compares against cross-modal fusion on this file.</p>
                </div>
                <button onclick="runFullAblation()" class="text-xs font-medium text-blue-400 hover:text-blue-300 py-1.5 px-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    🔄 Run 4-Way Comparison
                </button>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead class="text-xs uppercase bg-gray-900/60 text-gray-400 border-b border-gray-800">
                        <tr>
                            <th class="py-3 px-4">Architecture Mode</th>
                            <th class="py-3 px-4">Verdict</th>
                            <th class="py-3 px-4">Fake Probability</th>
                            <th class="py-3 px-4">Confidence</th>
                            <th class="py-3 px-4">Detection Mechanism</th>
                        </tr>
                    </thead>
                    <tbody id="ablationBody" class="divide-y divide-gray-800 text-gray-300">
                        <!-- Populated dynamically -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let currentDiagnosis = null;
        let chartInstance = null;

        async function initPage() {
            try {
                const res = await fetch('/api/samples');
                const samples = await res.json();
                const sel = document.getElementById('videoSelect');
                sel.innerHTML = '';
                samples.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s.path;
                    opt.innerText = `${s.name} (${s.category})`;
                    sel.appendChild(opt);
                });
                if (samples.length > 0) {
                    runAnalysis();
                }
            } catch (e) {
                console.error(e);
            }
        }

        async function runAnalysis() {
            const videoPath = document.getElementById('videoSelect').value;
            const mode = document.getElementById('modeSelect').value;
            if (!videoPath) return;

            document.getElementById('player').src = `/media?file=${encodeURIComponent(videoPath)}`;

            try {
                const res = await fetch(`/api/predict?video=${encodeURIComponent(videoPath)}&mode=${mode}`);
                const data = await res.json();
                currentDiagnosis = data;
                renderDiagnosis(data);
            } catch (err) {
                alert('Analysis failed: ' + err);
            }
        }

        function renderDiagnosis(data) {
            const banner = document.getElementById('verdictBanner');
            const icon = document.getElementById('verdictIcon');
            const vText = document.getElementById('verdictText');
            const vDetail = document.getElementById('verdictDetail');
            const conf = document.getElementById('confidenceVal');

            banner.classList.remove('hidden', 'bg-emerald-950/40', 'border-emerald-500/40', 'bg-rose-950/40', 'border-rose-500/40');

            if (data.is_fake) {
                banner.classList.add('bg-rose-950/40', 'border-rose-500/40');
                icon.className = 'w-14 h-14 rounded-2xl flex items-center justify-center text-3xl font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30';
                icon.innerText = '🚨';
                vText.className = 'text-2xl font-black text-rose-400';
                vText.innerText = data.verdict;
                vDetail.innerText = `Synthetic anomaly detected in cross-modal stream (${data.mode_used.toUpperCase()})`;
                conf.className = 'text-3xl font-extrabold text-rose-400';
            } else {
                banner.classList.add('bg-emerald-950/40', 'border-emerald-500/40');
                icon.className = 'w-14 h-14 rounded-2xl flex items-center justify-center text-3xl font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
                icon.innerText = '✅';
                vText.className = 'text-2xl font-black text-emerald-400';
                vText.innerText = data.verdict;
                vDetail.innerText = `Facial dynamics align with acoustic speech frequencies (${data.mode_used.toUpperCase()})`;
                conf.className = 'text-3xl font-extrabold text-emerald-400';
            }

            conf.innerText = `${data.confidence}%`;
            document.getElementById('scoreOverall').innerText = `${(data.fake_probability * 100).toFixed(1)}%`;
            document.getElementById('scoreVisual').innerText = data.breakdown.visual_manipulation_prob !== null ? `${(data.breakdown.visual_manipulation_prob * 100).toFixed(1)}%` : 'N/A';
            document.getElementById('scoreAudio').innerText = data.breakdown.audio_manipulation_prob !== null ? `${(data.breakdown.audio_manipulation_prob * 100).toFixed(1)}%` : 'N/A';
            document.getElementById('scoreSync').innerText = data.breakdown.phoneme_viseme_inconsistency !== null ? `${data.breakdown.phoneme_viseme_inconsistency.toFixed(3)}` : 'N/A';

            if (data.media_artifacts) {
                document.getElementById('faceImg').src = 'data:image/png;base64,' + data.media_artifacts.sample_face;
                document.getElementById('lipImg').src = 'data:image/png;base64,' + data.media_artifacts.sample_lip;
                document.getElementById('specImg').src = 'data:image/png;base64,' + data.media_artifacts.spectrogram_png;
            }

            // Render Timeline Chart
            const diag = data.temporal_diagnostics;
            if (diag && diag.timestamps && diag.inconsistency_curve) {
                renderTimeline(diag.timestamps, diag.inconsistency_curve);
            }
        }

        function renderTimeline(timestamps, curve) {
            const ctx = document.getElementById('timelineChart').getContext('2d');
            if (chartInstance) {
                chartInstance.destroy();
            }
            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: timestamps.map(t => t.toFixed(2) + 's'),
                    datasets: [{
                        label: 'Phoneme-Viseme Inconsistency Δ(t)',
                        data: curve,
                        borderColor: '#E11D48',
                        backgroundColor: 'rgba(225, 29, 72, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 2,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#94A3B8', font: { size: 10 } } } },
                    scales: {
                        x: { ticks: { color: '#64748B', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { min: 0, max: 1, ticks: { color: '#64748B', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            });
        }

        async function runFullAblation() {
            const videoPath = document.getElementById('videoSelect').value;
            const tbody = document.getElementById('ablationBody');
            tbody.innerHTML = '<tr><td colspan="5" class="py-4 text-center text-gray-500">Evaluating 4 architectures...</td></tr>';

            const modes = [
                { id: 'visual_only', name: '1. Visual-Only (Video Baseline)', desc: 'ResNet-18 spatial features & BiGRU temporal flow' },
                { id: 'audio_only', name: '2. Audio-Only (Audio Baseline)', desc: '2D-CNN Log-Mel Spectrogram speech envelope' },
                { id: 'late_fusion', name: '3. Late Fusion (Score Concat)', desc: 'Dense concatenation of unimodal embeddings' },
                { id: 'multimodal', name: '4. Multimodal Cross-Attention (Proposed)', desc: 'Phoneme-viseme multi-head cross-attention' },
            ];

            let html = '';
            for (const m of modes) {
                const res = await fetch(`/api/predict?video=${encodeURIComponent(videoPath)}&mode=${m.id}`);
                const data = await res.json();
                const isFake = data.is_fake;
                const badge = isFake 
                    ? '<span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">MANIPULATED</span>'
                    : '<span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">AUTHENTIC</span>';

                html += `
                    <tr class="hover:bg-gray-900/40">
                        <td class="py-3 px-4 font-medium text-white">${m.name}</td>
                        <td class="py-3 px-4">${badge}</td>
                        <td class="py-3 px-4">${(data.fake_probability * 100).toFixed(2)}%</td>
                        <td class="py-3 px-4 font-semibold">${data.confidence}%</td>
                        <td class="py-3 px-4 text-xs text-gray-400">${m.desc}</td>
                    </tr>
                `;
            }
            tbody.innerHTML = html;
        }

        function downloadReport() {
            if (!currentDiagnosis) return;
            const text = `
============================================================
       MULTIMODAL DEEPFAKE FORENSIC ANALYSIS REPORT
============================================================
VERDICT           : ${currentDiagnosis.verdict}
CONFIDENCE LEVEL  : ${currentDiagnosis.confidence}%
FAKE PROBABILITY  : ${currentDiagnosis.fake_probability}
EVALUATION MODE   : ${currentDiagnosis.mode_used.toUpperCase()}
------------------------------------------------------------
MODALITY BREAKDOWN:
  * Visual Artifact Score      : ${currentDiagnosis.breakdown.visual_manipulation_prob}
  * Voice Acoustic Score       : ${currentDiagnosis.breakdown.audio_manipulation_prob}
  * Audio-Visual Desync Index  : ${currentDiagnosis.breakdown.phoneme_viseme_inconsistency}
============================================================
`;
            const blob = new Blob([text], { type: 'text/plain' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'deepfake_forensic_report.txt';
            a.click();
        }

        window.onload = initPage;
    </script>
</body>
</html>
"""


class ForensicRequestHandler(SimpleHTTPRequestHandler):
    """Handles REST API and static media streaming."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. Main Dashboard UI
        if path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
            return

        # 2. List Benchmark Media Samples
        if path == "/api/samples":
            sample_dir = PROJECT_ROOT / "data" / "sample_data"
            csv_path = sample_dir / "dataset_index.csv"
            samples = []
            if csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    samples.append({
                        "name": Path(row["video_path"]).name,
                        "path": str(row["video_path"]),
                        "category": str(row.get("category", "")),
                        "label": int(row.get("label", 0)),
                    })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(samples).encode("utf-8"))
            return

        # 3. Predict Deepfake Anomaly API
        if path == "/api/predict":
            video_path = query.get("video", [None])[0]
            mode = query.get("mode", ["multimodal"])[0]

            if not video_path or not os.path.exists(video_path):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid video path"}')
                return

            pipe = get_pipeline()
            diag = pipe.predict(video_path=video_path, mode=mode)

            # Format image artifacts to base64
            artifacts = diag.get("media_artifacts", {})
            spec_base64 = ""
            face_base64 = ""
            lip_base64 = ""

            if "mel_spectrogram" in artifacts:
                spec_base64 = plot_spectrogram_base64(artifacts["mel_spectrogram"])
            if "sample_face" in artifacts:
                face_base64 = image_to_base64(artifacts["sample_face"])
            if "sample_lip" in artifacts:
                lip_base64 = image_to_base64(artifacts["sample_lip"])

            payload = {
                "verdict": diag["verdict"],
                "is_fake": diag["is_fake"],
                "confidence": diag["confidence"],
                "fake_probability": diag["fake_probability"],
                "real_probability": diag["real_probability"],
                "mode_used": diag["mode_used"],
                "breakdown": diag["breakdown"],
                "temporal_diagnostics": diag["temporal_diagnostics"],
                "media_artifacts": {
                    "spectrogram_png": spec_base64,
                    "sample_face": face_base64,
                    "sample_lip": lip_base64,
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        # 4. Stream Raw Video Media
        if path == "/media":
            file_param = query.get("file", [None])[0]
            if file_param and os.path.exists(file_param):
                mime, _ = mimetypes.guess_type(file_param)
                mime = mime or "video/mp4"
                file_size = os.path.getsize(file_param)

                with open(file_param, "rb") as f:
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(file_size))
                    self.end_headers()
                    self.wfile.write(f.read())
                return

        super().do_GET()


def run_server(port: int = 8501):
    """Starts the forensic web application server."""
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, ForensicRequestHandler)
    print("=" * 70)
    print(f"[+] MULTIMODAL DEEPFAKE FORENSIC SERVER RUNNING")
    print(f"--> Local URL: http://localhost:{port}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(port)
