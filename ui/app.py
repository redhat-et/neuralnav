"""
NeuralNav - E2E LLM Deployment Recommendation System

A beautiful, presentation-ready Streamlit application demonstrating:
1. Business Context Extraction (Qwen 2.5 7B @ 95.1% accuracy)
2. MCDM Scoring (204 Open-Source Models)
3. Full Explainability with visual score breakdowns
4. SLO & Workload Impact Analysis
5. Hardware-aware recommendations

Usage:
    streamlit run ui/poc_app.py
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).parent.parent / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL NAME NORMALIZATION - Consistent Provider/Model-Name Format
# =============================================================================

# Provider mapping for model name normalization
PROVIDER_MAPPING = {
    "gpt-oss": "OpenAI",
    "gptoss": "OpenAI",
    "kimi": "Moonshot",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "llama": "Meta",
    "mistral": "Mistral",
    "gemma": "Google",
    "minimax": "MiniMax",
    "phi": "Microsoft",
    "falcon": "TII",
    "yi": "01.AI",
    "internlm": "Shanghai AI Lab",
    "baichuan": "Baichuan",
    "chatglm": "Zhipu",
    "glm": "Zhipu",
    "starcoder": "BigCode",
    "codellama": "Meta",
    "aya": "Cohere",
}


def normalize_model_name(raw_name: str) -> str:
    """
    Normalize model name to consistent Provider/Model-Name format.
    
    Examples:
        "GPT-OSS 120B" → "OpenAI/GPT-OSS-120B"
        "gpt-oss-120b" → "OpenAI/GPT-OSS-120B"
        "Moonshot/Kimi-K2-Thinking" → "Moonshot/Kimi-K2-Thinking"
        "kimi-k2-thinking" → "Moonshot/Kimi-K2-Thinking"
        "DeepSeek/DeepSeek-V3.1-Reasoning" → "DeepSeek/DeepSeek-V3.1-Reasoning"
    """
    if not raw_name:
        return "Unknown"
    
    name = raw_name.strip()
    
    # Already has provider prefix (e.g., "Moonshot/Kimi-K2")
    if "/" in name:
        parts = name.split("/", 1)
        provider = parts[0].strip()
        model = parts[1].strip()
        # Normalize model part: title case with hyphens
        model_normalized = "-".join(word.title() if not word.isupper() else word 
                                     for word in model.replace("_", "-").split("-"))
        return f"{provider}/{model_normalized}"
    
    # No provider - need to detect and add it
    name_lower = name.lower().replace(" ", "-").replace("_", "-")
    
    # Find matching provider
    detected_provider = None
    for keyword, provider in PROVIDER_MAPPING.items():
        if keyword in name_lower:
            detected_provider = provider
            break
    
    if not detected_provider:
        detected_provider = "Unknown"
    
    # Normalize model name: title case with hyphens
    # Keep version numbers and special terms intact
    model_parts = name.replace(" ", "-").replace("_", "-").split("-")
    normalized_parts = []
    for part in model_parts:
        if not part:
            continue
        # Keep uppercase terms (like "OSS", "W4A16", "B200")
        if part.isupper() or any(c.isdigit() for c in part):
            normalized_parts.append(part.upper() if part.isalpha() else part)
        else:
            normalized_parts.append(part.title())
    
    model_normalized = "-".join(normalized_parts)
    
    return f"{detected_provider}/{model_normalized}"


def format_display_name(raw_name: str) -> str:
    """
    Format model name for display (uppercase, spaces instead of hyphens).
    
    Examples:
        "OpenAI/GPT-OSS-120B" → "OPENAI / GPT OSS 120B"
        "Moonshot/Kimi-K2-Thinking" → "MOONSHOT / KIMI K2 THINKING"
    """
    normalized = normalize_model_name(raw_name)
    if "/" in normalized:
        provider, model = normalized.split("/", 1)
        model_display = model.replace("-", " ")
        return f"{provider.upper()} / {model_display.upper()}"
    return normalized.upper().replace("-", " ")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NeuralNav",
    page_icon="docs/neuralnav-logo-32.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS - Beautiful Enterprise Styling
# =============================================================================

st.markdown("""
<style>
    /* Import Google Fonts - Qualifire & HuggingFace Inspired Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        /* Dark background */
        background: 
            radial-gradient(ellipse at 20% 0%, rgba(238, 0, 0, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(238, 0, 0, 0.03) 0%, transparent 50%),
            linear-gradient(180deg, #0c0c0c 0%, #151515 25%, #1a1a1a 50%, #151515 75%, #0c0c0c 100%);
        background-attachment: fixed;
        color: #ffffff;
        font-size: 16px;
        line-height: 1.7;
        letter-spacing: -0.01em;
    }
    
    /* Keyframe Animations - Subtle, Professional */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 25px rgba(99, 102, 241, 0.15); }
        50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.25), 0 0 60px rgba(16, 185, 129, 0.1); }
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes slide-up {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes scale-in {
        from { opacity: 0; transform: scale(0.97); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes border-glow {
        0%, 100% { border-color: rgba(99, 102, 241, 0.3); }
        50% { border-color: rgba(99, 102, 241, 0.6); }
    }
    
    /* Corporate Color Palette */
    :root {
        --bg-primary: #0c0c0c;
        --bg-secondary: #151515;
        --bg-tertiary: #1a1a1a;
        --bg-card: rgba(21, 21, 21, 0.9);
        --bg-card-hover: rgba(26, 26, 26, 0.95);
        --border-default: rgba(255, 255, 255, 0.1);
        --border-accent: rgba(238, 0, 0, 0.5);
        --border-success: rgba(255, 255, 255, 0.3);
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --text-muted: #6a6a6a;
        --accent-indigo: #EE0000;
        --accent-purple: #EE0000;
        --accent-blue: #EE0000;
        --accent-cyan: #ffffff;
        --accent-emerald: #ffffff;
        --accent-green: #ffffff;
        --accent-yellow: #EE0000;
        --accent-orange: #EE0000;
        --accent-rose: #EE0000;
        --accent-pink: #EE0000;
        --gradient-primary: linear-gradient(135deg, #EE0000 0%, #cc0000 50%, #a00000 100%);
        --gradient-success: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        --gradient-hero: linear-gradient(135deg, #151515 0%, #1a1a1a 50%, #151515 100%);
        --gradient-card: linear-gradient(145deg, rgba(238, 0, 0, 0.03) 0%, rgba(255, 255, 255, 0.02) 100%);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        --shadow-glow: 0 0 40px rgba(238, 0, 0, 0.1);
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #151515 0%, #1a1a1a 50%, #151515 100%);
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(238, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 80%, rgba(238, 0, 0, 0.08) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(238, 0, 0, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
        pointer-events: none;
    }
    .hero-emoji {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        animation: float 5s ease-in-out infinite;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.4));
        position: relative;
        z-index: 1;
    }
    .hero-logo {
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 5px 15px rgba(238,0,0,0.4));
        position: relative;
        z-index: 1;
        display: inline-block;
        margin-right: 1rem;
        vertical-align: middle;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 15px rgba(0,0,0,0.4);
        letter-spacing: -1px;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        position: relative;
        z-index: 1;
        display: inline-block;
        vertical-align: middle;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        font-weight: 400;
        max-width: 700px;
        line-height: 1.4;
        position: relative;
        z-index: 1;
        margin-top: 0.5rem;
    }
    .hero-badges {
        display: none;
    }
    .hero-badge {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(16px);
        padding: 0.75rem 1.5rem;
        border-radius: 100px;
        font-size: 0.95rem;
        font-weight: 600;
        color: white;
        border: 1px solid rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.25s ease;
        font-family: 'Inter', sans-serif;
    }
    .hero-badge:hover {
        background: rgba(255,255,255,0.18);
        transform: translateY(-3px);
        border-color: rgba(255,255,255,0.3);
    }
    
    /* Stats Cards - HuggingFace Leaderboard Inspired */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1.25rem;
        margin: 2.5rem 0;
    }
    .stat-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        padding: 1.75rem 1.25rem;
        border-radius: 1rem;
        text-align: center;
        border: 1px solid var(--border-default);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-primary);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .stat-card::after {
        content: '';
        position: absolute;
        inset: 0;
        background: var(--gradient-card);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: var(--border-accent);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.1), 0 8px 20px rgba(0,0,0,0.2);
    }
    .stat-card:hover::before {
        opacity: 1;
    }
    .stat-card:hover::after {
        opacity: 1;
    }
    .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: block;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple), var(--accent-emerald));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Editable Fields with Pencil Icons */
    .editable-field {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .editable-field:hover {
        background: rgba(102, 126, 234, 0.15);
    }
    .pencil-icon {
        font-size: 0.9rem;
        color: #38ef7d;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .editable-field:hover .pencil-icon {
        opacity: 1;
    }
    .edit-input {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 0.5rem;
        padding: 0.5rem 0.75rem;
        color: white;
        font-weight: 600;
        width: 100px;
        text-align: center;
    }
    .edit-input:focus {
        border-color: #38ef7d;
        outline: none;
        box-shadow: 0 0 0 3px rgba(56, 239, 125, 0.25);
    }
    
    /* TOP LEADERBOARD TABLE - HuggingFace/Qualifire Inspired */
    .leaderboard-container {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border-radius: 1rem;
        border: 1px solid var(--border-default);
        overflow-x: auto;
        overflow-y: hidden;
        margin: 2rem 0;
        animation: fade-in 0.5s ease-out;
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
        width: 100%;
    }
    .leaderboard-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        padding: 1.25rem 1.75rem;
        border-bottom: 1px solid var(--border-default);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .leaderboard-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .leaderboard-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
    }
    .leaderboard-table th {
        background: rgba(88, 166, 255, 0.08);
        color: var(--text-secondary);
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 1rem 0.5rem;
        text-align: center;
        border-bottom: 1px solid var(--border-default);
        white-space: nowrap;
    }
    /* Column widths - FIXED layout with explicit widths (must add to 100%) */
    .leaderboard-table th:nth-child(1),
    .leaderboard-table td:nth-child(1) { width: 5%; }     /* Rank */
    .leaderboard-table th:nth-child(2),
    .leaderboard-table td:nth-child(2) { width: 18%; text-align: left; }  /* Model */
    .leaderboard-table th:nth-child(3),
    .leaderboard-table td:nth-child(3) { width: 10%; }    /* Accuracy */
    .leaderboard-table th:nth-child(4),
    .leaderboard-table td:nth-child(4) { width: 10%; }    /* Latency */
    .leaderboard-table th:nth-child(5),
    .leaderboard-table td:nth-child(5) { width: 10%; }    /* Cost */
    .leaderboard-table th:nth-child(6),
    .leaderboard-table td:nth-child(6) { width: 10%; }    /* Capacity */
    .leaderboard-table th:nth-child(7),
    .leaderboard-table td:nth-child(7) { width: 10%; }    /* Final Score */
    .leaderboard-table th:nth-child(8),
    .leaderboard-table td:nth-child(8) { width: 17%; }    /* Pros & Cons */
    .leaderboard-table th:nth-child(9),
    .leaderboard-table td:nth-child(9) { width: 10%; }    /* Action */
    .leaderboard-table td {
        padding: 1rem 0.5rem;
        border-bottom: 1px solid var(--border-default);
        color: var(--text-primary);
        vertical-align: middle;
        text-align: center;
    }
    .leaderboard-table tr:hover td {
        background: rgba(88, 166, 255, 0.05);
    }
    .leaderboard-table tr:last-child td {
        border-bottom: none;
    }
    /* Consistent row height */
    .leaderboard-table tr {
        height: 85px;
    }
    
    /* Rank Badges - HuggingFace Style Medals */
    .rank-badge {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        color: white;
        margin: 0 auto;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .rank-badge:hover {
        transform: scale(1.08);
    }
    .rank-1 {
        background: linear-gradient(145deg, #fcd34d, #f59e0b);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
        color: #78350f;
    }
    .rank-2 {
        background: linear-gradient(145deg, #e5e7eb, #9ca3af);
        box-shadow: 0 4px 12px rgba(156, 163, 175, 0.4);
        color: #374151;
    }
    .rank-3 {
        background: linear-gradient(145deg, #fdba74, #ea580c);
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.35);
        color: #7c2d12;
    }
    /* Ranks 4-10 - Indigo gradient */
    .rank-4, .rank-5, .rank-6, .rank-7, .rank-8, .rank-9, .rank-10 {
        background: linear-gradient(145deg, #6366f1, #8b5cf6);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Score Bars - Corporate Enhanced Style */
    .score-mini-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        width: 100%;
        max-width: 120px;
        margin: 0 auto;
        padding: 0.5rem 0;
    }
    .score-mini-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        width: 100%;
        position: relative;
    }
    .score-mini-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .score-mini-label {
        font-size: 1.4rem;
        font-weight: 700;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.02em;
    }
    .score-num {
        display: none;
    }
    .fill-accuracy { background: linear-gradient(90deg, #db2777, #ec4899); }
    .fill-quality { background: linear-gradient(90deg, #059669, #10b981); }
    .fill-latency { background: linear-gradient(90deg, #2563eb, #3b82f6); }
    .fill-cost { background: linear-gradient(90deg, #ea580c, #f97316); }
    .fill-capacity { background: linear-gradient(90deg, #7c3aed, #8b5cf6); }
    
    /* Score label colors - Enhanced visibility */
    .label-accuracy { color: #f472b6; text-shadow: 0 0 12px rgba(244, 114, 182, 0.3); }
    .label-quality { color: #34d399; text-shadow: 0 0 12px rgba(16, 185, 129, 0.3); }
    .label-latency { color: #60a5fa; text-shadow: 0 0 12px rgba(59, 130, 246, 0.3); }
    .label-cost { color: #fb923c; text-shadow: 0 0 12px rgba(249, 115, 22, 0.3); }
    .label-capacity { color: #a78bfa; text-shadow: 0 0 12px rgba(139, 92, 246, 0.3); }
    
    /* Model Card in Table - Corporate Typography */
    .model-cell {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .model-info {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .model-name {
        font-weight: 600;
        font-size: 1.05rem;
        color: #f9fafb;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }
    .model-provider {
        font-size: 0.85rem;
        color: #9ca3af;
        font-weight: 500;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Enhanced Select Button - Corporate Style */
    .select-btn {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        padding: 0.6rem 1.25rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
    .select-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35);
    }
    
    /* Final Score Display - BIG and prominent */
    .final-score {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.02em;
        display: block;
        text-align: center;
    }
    
    /* Enhanced table row spacing */
    .leaderboard-table tbody tr {
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .leaderboard-table tbody tr:hover {
        background: rgba(99, 102, 241, 0.08);
    }
    .leaderboard-table td {
        padding: 1rem 0.75rem !important;
        vertical-align: middle;
    }
    .leaderboard-table th {
        padding: 1rem 0.75rem !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.5);
        font-weight: 600;
    }
    
    /* Enhanced Slider Styling */
    .stSlider {
        padding: 0.5rem 0;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.3), rgba(56, 239, 125, 0.3)) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #38ef7d) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        background: linear-gradient(135deg, #667eea, #38ef7d) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 0.5rem !important;
        padding: 0.25rem 0.5rem !important;
    }
    .stSlider label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Pros/Cons Tags - Clean pill design like HuggingFace */
    .tag-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .tag {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        white-space: nowrap;
        text-align: center;
        transition: all 0.2s ease;
    }
    .tag:hover {
        transform: scale(1.05);
    }
    .tag-pro {
        background: rgba(63, 185, 80, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .tag-con {
        background: rgba(248, 81, 73, 0.15);
        color: var(--accent-orange);
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    .tag-neutral {
        background: rgba(255,255,255,0.08);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
    }
    
    /* Action Button - HuggingFace style rounded */
    .action-cell {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Extraction Card - Clean, spacious design */
    .extraction-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 1.25rem;
        border: 1px solid var(--border-default);
        margin: 2rem 0;
        animation: slide-up 0.4s ease-out;
    }
    .extraction-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
    }
    .extraction-item {
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 1rem;
        display: flex;
        align-items: center;
        gap: 1.25rem;
        border: 1px solid var(--border-default);
        transition: all 0.25s ease;
    }
    .extraction-item:hover {
        border-color: var(--accent-blue);
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.1);
    }
    .extraction-icon {
        display: none;  /* Hide extraction icons */
    }
    .extraction-icon-usecase { display: none; }
    .extraction-icon-users { display: none; }
    .extraction-icon-priority { display: none; }
    .extraction-icon-hardware { display: none; }
    .extraction-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }
    .extraction-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 4px;
        font-family: 'Inter', sans-serif;
    }
    
    /* SLO & Workload Cards - Clean, readable design */
    .slo-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .slo-card {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        padding: 1.75rem;
        border-radius: 1rem;
        border: 1px solid var(--border-default);
        transition: all 0.25s ease;
        position: relative;
    }
    .slo-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(88, 166, 255, 0.12);
    }
    .slo-card-editable {
        cursor: pointer;
    }
    .slo-card .edit-indicator {
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 0.9rem;
        color: var(--accent-green);
        opacity: 0;
        transition: opacity 0.2s ease;
    }
    .slo-card:hover .edit-indicator {
        opacity: 1;
    }
    .slo-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 1.25rem;
    }
    .slo-icon {
        font-size: 1.75rem;
    }
    .slo-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--accent-purple);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .slo-metrics {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .slo-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        white-space: nowrap;
        padding: 0.5rem;
        border-radius: 8px;
        transition: background 0.2s ease;
    }
    .slo-metric:hover {
        background: rgba(88, 166, 255, 0.08);
    }
    .slo-metric-name {
        font-size: 0.95rem;
        color: var(--text-secondary);
        flex-shrink: 0;
    }
    .slo-metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        text-align: right;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    .slo-metric-value .edit-icon {
        font-size: 0.85rem;
        color: var(--accent-green);
        opacity: 0;
        cursor: pointer;
        transition: opacity 0.2s ease;
    }
    .slo-metric:hover .edit-icon {
        opacity: 1;
    }
    .slo-metric-good {
        color: var(--accent-green);
    }
    .slo-metric-warn {
        color: var(--accent-orange);
    }
    
    /* Impact Visualization */
    .impact-container {
        background: rgba(255,255,255,0.02);
        border-radius: 1.25rem;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin: 1rem 0;
    }
    .impact-header {
        font-size: 1rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .impact-item {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 1rem;
        padding: 0.85rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .impact-item:last-child {
        border-bottom: none;
    }
    .impact-factor {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .impact-icon {
        font-size: 1.25rem;
    }
    .impact-name {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }
    .impact-effect {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.4rem 0.8rem;
        background: rgba(56, 239, 125, 0.1);
        border-radius: 0.5rem;
        min-width: 180px;
        text-align: right;
    }
    .impact-positive { 
        color: #38ef7d; 
        background: rgba(56, 239, 125, 0.1);
    }
    .impact-negative { 
        color: #f5576c; 
        background: rgba(245, 87, 108, 0.1);
    }
    .impact-neutral { 
        color: rgba(255,255,255,0.7); 
        background: rgba(255,255,255,0.05);
    }
    
    /* Priority Badges - Clean pill design with black background */
    .priority-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        color: white;
        background: #000000;
        transition: transform 0.2s ease;
    }
    .priority-badge:hover {
        transform: scale(1.03);
    }
    .priority-low_latency { background: #000000; color: white; }
    .priority-cost_saving { background: #000000; color: white; }
    .priority-high_accuracy { background: #000000; color: white; }
    .priority-high_throughput { background: #000000; color: white; }
    .priority-balanced { background: #000000; color: white; }
    
    /* Score Bars - Clean, readable with AA-style */
    .score-container {
        margin: 1.25rem 0;
        animation: slide-up 0.4s ease-out;
    }
    .score-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        color: var(--text-primary);
        font-weight: 500;
        font-size: 1rem;
    }
    .score-bar-bg {
        background: var(--bg-tertiary);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .score-bar-quality { background: linear-gradient(90deg, #059669, var(--accent-green)); }
    .score-bar-latency { background: linear-gradient(90deg, #3b82f6, var(--accent-blue)); }
    .score-bar-cost { background: linear-gradient(90deg, #f97316, var(--accent-orange)); }
    .score-bar-capacity { background: linear-gradient(90deg, #8b5cf6, var(--accent-purple)); }
    
    /* Section Headers - Clean, modern */
    .section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2.5rem 0 2rem 0;
        color: var(--text-primary);
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.08), rgba(163, 113, 247, 0.05));
        padding: 1.25rem 1.75rem;
        border-radius: 1rem;
        border: 1px solid var(--border-default);
        font-family: 'Inter', sans-serif;
    }
    .section-header span:first-child {
        font-size: 2rem;
    }
    
    /* Pipeline Steps - Clean card design */
    .pipeline-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
        margin: 2.5rem 0;
    }
    .pipeline-step {
        background: var(--bg-card);
        padding: 2.5rem 2rem;
        border-radius: 1rem;
        border: 1px solid var(--border-default);
        text-align: center;
        transition: all 0.25s ease;
        position: relative;
    }
    .pipeline-step:hover {
        transform: translateY(-6px);
        border-color: var(--accent-blue);
        box-shadow: 0 15px 45px rgba(88, 166, 255, 0.12);
    }
    .pipeline-number {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: white;
        font-size: 1.75rem;
        margin: 0 auto 1.25rem;
        font-family: 'Inter', sans-serif;
    }
    .pipeline-number-1 { background: linear-gradient(135deg, var(--accent-purple), #7c3aed); }
    .pipeline-number-2 { background: linear-gradient(135deg, #059669, var(--accent-green)); }
    .pipeline-number-3 { background: linear-gradient(135deg, var(--accent-orange), var(--accent-pink)); }
    .pipeline-title {
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        font-family: 'Inter', sans-serif;
    }
    .pipeline-desc {
        font-size: 1rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Input Container - Qualifire-inspired clean design */
    .input-container {
        background: var(--bg-card);
        padding: 2.5rem;
        border-radius: 1.25rem;
        border: 1px solid var(--border-default);
        margin: 2rem 0;
    }
    
    /* Sidebar Styling - Clean dark theme matching main */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-default) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-secondary) !important;
    }
    .sidebar-section {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        border: 1px solid var(--border-default);
    }
    .sidebar-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted) !important;
        font-weight: 700;
        margin-bottom: 1.25rem;
    }
    
    /* Weight Bars - Clean design */
    .weight-item {
        margin: 1rem 0;
    }
    .weight-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }
    .weight-name {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    .weight-value {
        font-weight: 700;
        color: var(--accent-blue) !important;
        font-family: 'Inter', sans-serif;
    }
    .weight-bar {
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        background: var(--bg-tertiary);
    }
    .weight-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    .weight-fill-quality { background: linear-gradient(90deg, #059669, var(--accent-green)); }
    .weight-fill-latency { background: linear-gradient(90deg, #3b82f6, var(--accent-blue)); }
    .weight-fill-cost { background: linear-gradient(90deg, #f97316, var(--accent-orange)); }
    .weight-fill-capacity { background: linear-gradient(90deg, #8b5cf6, var(--accent-purple)); }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 5px; }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(135deg, #667eea, #764ba2); 
        border-radius: 5px; 
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #151515;
        border-radius: 4px;
        padding: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 4px;
        padding: 12px 24px;
        border: none;
        color: #a0a0a0;
        transition: all 0.2s ease;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: rgba(238, 0, 0, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: #EE0000 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(238, 0, 0, 0.3);
    }
    /* Override the blue tab indicator line */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #EE0000 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: rgba(238, 0, 0, 0.3) !important;
    }
    /* Hide the default blue indicator */
    .stTabs [role="tablist"] > div:last-child {
        background-color: #EE0000 !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 4px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background: #1a1a1a !important;
        color: #ffffff !important;
        padding: 12px 24px;
        font-size: 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(238, 0, 0, 0.2);
        border-color: #EE0000;
        color: white !important;
    }
    .stButton > button[kind="primary"] {
        background: #EE0000 !important;
        color: white !important;
        border: none;
        box-shadow: 0 4px 15px rgba(238, 0, 0, 0.35);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 30px rgba(238, 0, 0, 0.45);
        transform: translateY(-3px);
        background: #cc0000 !important;
        color: white !important;
    }
    
    /* Number Input Styling - Clean design */
    .stNumberInput input {
        background: var(--bg-tertiary) !important;
        border: 2px solid var(--border-default) !important;
        border-radius: 8px !important;
        color: var(--accent-green) !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stNumberInput input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2) !important;
    }
    .stNumberInput [data-testid="stNumberInputContainer"] {
        background: transparent !important;
    }
    .stNumberInput button {
        background: var(--bg-tertiary) !important;
        color: var(--accent-blue) !important;
        border-radius: 6px !important;
    }
    .stNumberInput button:hover {
        background: rgba(88, 166, 255, 0.2) !important;
    }
    
    /* Enhanced Selectbox Styling */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 10px !important;
    }
    .stSelectbox > div > div:hover {
        border-color: var(--accent-blue) !important;
    }
    
    /* Enhanced Slider Styling */
    .stSlider > div > div > div {
        background: var(--bg-tertiary) !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-green)) !important;
    }
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid var(--accent-blue) !important;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3) !important;
    }
    
    /* Filter Panel Styling */
    .filter-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Badges - Clean pill design */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    .metric-badge:hover {
        transform: scale(1.03);
    }
    .metric-badge-accuracy {
        background: rgba(63, 185, 80, 0.12);
        color: var(--accent-green);
        border: 1px solid rgba(63, 185, 80, 0.25);
    }
    .metric-badge-latency {
        background: rgba(88, 166, 255, 0.12);
        color: var(--accent-blue);
        border: 1px solid rgba(88, 166, 255, 0.25);
    }
    .metric-badge-cost {
        background: rgba(249, 115, 22, 0.12);
        color: var(--accent-orange);
        border: 1px solid rgba(249, 115, 22, 0.25);
    }
    .metric-badge-capacity {
        background: rgba(163, 113, 247, 0.12);
        color: var(--accent-purple);
        border: 1px solid rgba(163, 113, 247, 0.25);
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #EE0000, #CC0000, #EE0000) !important;
        border-radius: 8px;
    }
    .stProgress > div {
        background: #1a1a1a !important;
        border-radius: 8px;
    }
    
    /* Spinner Styling */
    .stSpinner > div {
        border-top-color: #EE0000 !important;
    }
    .stSpinner > div > div {
        border-top-color: #EE0000 !important;
    }
    [data-testid="stSpinner"] svg {
        color: #EE0000 !important;
        fill: #EE0000 !important;
    }
    
    /* Info/Warning/Error Message Styling */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--border-default) !important;
    }
    .stAlert > div {
        padding: 1.25rem 1.5rem !important;
    }
    
    /* Caption Styling */
    .stCaption {
        color: var(--text-muted) !important;
    }
    
    /* Text Area Focus Styling */
    .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15) !important;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Disabled Button Styling */
    .stButton > button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }
    
    /* Smooth Page Transitions */
    .main .block-container {
        animation: fade-in 0.25s ease-out;
        max-width: 1400px;
        padding: 2rem 3rem;
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Better Tooltip Styling */
    [data-testid="stTooltipIcon"] {
        color: var(--text-muted) !important;
    }
    [data-testid="stTooltipIcon"]:hover {
        color: var(--accent-blue) !important;
    }
    
    /* Custom Scrollbar - Subtle */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); border-radius: 4px; }
    ::-webkit-scrollbar-thumb { 
        background: var(--bg-tertiary); 
        border-radius: 4px; 
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }
    
    /* Legend for charts - AA inspired */
    .chart-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin: 1.5rem 0;
        padding: 1rem 1.5rem;
        background: var(--bg-card);
        border-radius: 10px;
        border: 1px solid var(--border-default);
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    .legend-color {
        width: 14px;
        height: 14px;
        border-radius: 4px;
    }
    .legend-color-accuracy { background: var(--accent-green); }
    .legend-color-latency { background: var(--accent-blue); }
    .legend-color-cost { background: var(--accent-orange); }
    .legend-color-capacity { background: var(--accent-purple); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "models_df" not in st.session_state:
    st.session_state.models_df = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
# Workflow approval state
if "extraction_approved" not in st.session_state:
    st.session_state.extraction_approved = None  # None = pending, True = approved, False = editing
if "slo_approved" not in st.session_state:
    st.session_state.slo_approved = None
if "edited_extraction" not in st.session_state:
    st.session_state.edited_extraction = None
# Editable SLO values
if "edit_slo" not in st.session_state:
    st.session_state.edit_slo = False
if "custom_ttft" not in st.session_state:
    st.session_state.custom_ttft = None
if "custom_itl" not in st.session_state:
    st.session_state.custom_itl = None
if "custom_e2e" not in st.session_state:
    st.session_state.custom_e2e = None
if "custom_qps" not in st.session_state:
    st.session_state.custom_qps = None
# SLO percentile selection (Mean, P90, P95, P99)
if "slo_percentile" not in st.session_state:
    st.session_state.slo_percentile = "p95"  # Default to P95

# Ranking weights are initialized in the PRIORITIES section of the Tech Spec tab
# using values from priority_weights.json. Do NOT hardcode defaults here.
# See fetch_priority_weights() and the PRIORITIES section around line 5317.

if "include_near_miss" not in st.session_state:
    st.session_state.include_near_miss = False

# Category expansion state (for inline expand/collapse of additional options)
if "expanded_categories" not in st.session_state:
    st.session_state.expanded_categories = set()

# Winner dialog state - must be explicitly initialized to False
if "show_winner_dialog" not in st.session_state:
    st.session_state.show_winner_dialog = False
if "balanced_winner" not in st.session_state:
    st.session_state.balanced_winner = None
if "winner_priority" not in st.session_state:
    st.session_state.winner_priority = "balanced"
if "winner_extraction" not in st.session_state:
    st.session_state.winner_extraction = {}

# Use case tracking
if "detected_use_case" not in st.session_state:
    st.session_state.detected_use_case = "chatbot_conversational"

# Category exploration dialog state
if "show_category_dialog" not in st.session_state:
    st.session_state.show_category_dialog = False
if "explore_category" not in st.session_state:
    st.session_state.explore_category = "balanced"
if "show_full_table_dialog" not in st.session_state:
    st.session_state.show_full_table_dialog = False
if "show_options_list_expanded" not in st.session_state:
    st.session_state.show_options_list_expanded = False
if "top5_balanced" not in st.session_state:
    st.session_state.top5_balanced = []
if "top5_accuracy" not in st.session_state:
    st.session_state.top5_accuracy = []
if "top5_latency" not in st.session_state:
    st.session_state.top5_latency = []
if "top5_cost" not in st.session_state:
    st.session_state.top5_cost = []
if "top5_simplest" not in st.session_state:
    st.session_state.top5_simplest = []

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_use_case_name(use_case: str) -> str:
    """Format use case name with proper capitalization for acronyms."""
    if not use_case:
        return "Unknown"
    # Replace underscores and title case
    formatted = use_case.replace('_', ' ').title()
    # Fix common acronyms
    acronyms = {
        'Rag': 'RAG',
        'Llm': 'LLM', 
        'Ai': 'AI',
        'Api': 'API',
        'Gpu': 'GPU',
        'Cpu': 'CPU',
        'Slo': 'SLO',
        'Qps': 'QPS',
        'Rps': 'RPS',
    }
    for wrong, right in acronyms.items():
        formatted = formatted.replace(wrong, right)
    return formatted

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_206_models() -> pd.DataFrame:
    """Load all 206 models from backend API."""
    try:
        # Fetch benchmark data from backend API
        response = requests.get(
            f"{API_BASE_URL}/api/v1/benchmarks",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Convert JSON to DataFrame
        if data.get("success") and data.get("benchmarks"):
            df = pd.DataFrame(data["benchmarks"])
            # Ensure Model Name column exists and is clean
            if 'Model Name' in df.columns:
                df = df.dropna(subset=['Model Name'])
                df = df[df['Model Name'].str.strip() != '']
            return df
        else:
            logger.warning("No benchmark data returned from API")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load benchmarks from API: {e}")
        return pd.DataFrame()

@st.cache_data
def load_research_slo_ranges():
    """Load research-backed SLO ranges from JSON file (includes benchmark data)."""
    try:
        json_path = DATA_DIR / "research" / "slo_ranges.json"
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def get_benchmark_ranges_for_token_config(prompt_tokens: int, output_tokens: int) -> dict:
    """Get actual min-max ranges for each percentile from benchmark data for a specific token config.
    
    OUTLIER FILTERING: Caps max values at realistic thresholds to exclude bad benchmark entries
    (e.g., 24B models on L4 GPUs that result in 200+ second TTFT).
    """
    try:
        json_path = DATA_DIR / "benchmarks_redhat_performance.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        benchmarks = data.get("benchmarks", [])
        
        # Filter by token config
        matching = [b for b in benchmarks 
                    if b.get('prompt_tokens') == prompt_tokens and b.get('output_tokens') == output_tokens]
        
        if not matching:
            return {"config_count": 0}
        
        # OUTLIER CAPS - realistic max values to filter bad benchmark entries
        # Based on analysis: mean TTFT ~100-900ms, outliers are 200,000+ ms
        OUTLIER_CAPS = {
            'ttft': 10000,    # Cap TTFT at 10 seconds (outliers are 200+ seconds)
            'itl': 500,       # Cap ITL at 500ms
            'e2e': 120000,    # Cap E2E at 2 minutes (outliers are 5+ minutes)
        }
        
        # Calculate min/max for each percentile with outlier filtering
        result = {"config_count": len(matching)}
        
        for percentile in ['mean', 'p90', 'p95', 'p99']:
            for metric in ['ttft', 'itl', 'e2e']:
                key = f"{metric}_{percentile}"
                cap = OUTLIER_CAPS.get(metric, float('inf'))
                
                # Filter values: must be > 0 and below the outlier cap
                vals = [b.get(key, 0) for b in matching 
                        if b.get(key) and 0 < b.get(key) <= cap]
                
                if vals:
                    result[f"{key}_min"] = min(vals)
                    result[f"{key}_max"] = max(vals)
                else:
                    # Fallback if no values pass filter
                    result[f"{key}_min"] = 0
                    result[f"{key}_max"] = cap
        
        return result
    except Exception:
        return {"config_count": 0}

@st.cache_data  
def load_research_workload_patterns():
    """Load research-backed workload patterns from JSON file (includes benchmark data)."""
    try:
        json_path = DATA_DIR / "research" / "workload_patterns.json"
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_performance_benchmarks():
    """Load performance benchmark data for real hardware/model performance validation."""
    try:
        # Benchmark file is in data/ root, not data/benchmarks/
        json_path = DATA_DIR / "benchmarks_BLIS.json"
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load benchmark data: {e}")
        return None

def get_slo_targets_for_use_case(use_case: str, priority: str = "balanced") -> dict:
    """Get research-backed SLO targets (min/max) for a use case with priority adjustment.
    
    Returns dict with:
    - ttft_target: {"min": X, "max": Y} - the acceptable range
    - itl_target: {"min": X, "max": Y}
    - e2e_target: {"min": X, "max": Y}
    - token_config: {"prompt": X, "output": Y}
    """
    research_data = load_research_slo_ranges()
    if not research_data:
        return None
    
    slo_ranges = research_data.get('slo_ranges', {})
    priority_adjustments = research_data.get('priority_adjustments', {})
    
    use_case_data = slo_ranges.get(use_case)
    if not use_case_data:
        return None
    
    # Get priority adjustment factors
    priority_factor = priority_adjustments.get(priority, {})
    ttft_factor = priority_factor.get('ttft_factor', 1.0)
    itl_factor = priority_factor.get('itl_factor', 1.0)
    e2e_factor = priority_factor.get('e2e_factor', 1.0)
    
    return {
        "use_case": use_case,
        "priority": priority,
        "description": use_case_data.get('description', ''),
        "ttft_target": {
            "min": int(use_case_data['ttft_ms']['min'] * ttft_factor),
            "max": int(use_case_data['ttft_ms']['max'] * ttft_factor),
        },
        "itl_target": {
            "min": int(use_case_data['itl_ms']['min'] * itl_factor),
            "max": int(use_case_data['itl_ms']['max'] * itl_factor),
        },
        "e2e_target": {
            "min": int(use_case_data['e2e_ms']['min'] * e2e_factor),
            "max": int(use_case_data['e2e_ms']['max'] * e2e_factor),
        },
        "token_config": use_case_data.get('token_config', {"prompt": 512, "output": 256}),
        "tokens_per_sec_target": use_case_data.get('tokens_per_sec', {}).get('target', 100),
        "research_note": use_case_data.get('research_note', ''),
        "recommended_hardware": priority_factor.get('recommended_hardware', 'H100_x2'),
    }


# =============================================================================
# RANKED RECOMMENDATIONS (Backend API Integration)
# =============================================================================

@st.cache_data(ttl=300)
def fetch_slo_defaults(use_case: str) -> dict | None:
    """Fetch default SLO values for a use case from the backend API.

    Returns dict with ttft_ms, itl_ms, e2e_ms each containing min, max, default.
    Cached for 5 minutes.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/slo-defaults/{use_case}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("slo_defaults")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch SLO defaults for {use_case}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_expected_rps(use_case: str, user_count: int) -> dict | None:
    """Fetch expected RPS for a use case from the backend API.

    Uses research-backed workload patterns to calculate:
    - expected_rps: average requests per second
    - peak_rps: peak capacity needed
    - workload_params: active_fraction, requests_per_min, etc.

    Cached for 5 minutes.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/expected-rps/{use_case}",
            params={"user_count": user_count},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch expected RPS for {use_case}: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_priority_weights() -> dict | None:
    """Fetch priority weights configuration from the backend API.

    Returns dict with priority_weights mapping and defaults.
    Cached for 1 hour (config rarely changes).
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/priority-weights",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("priority_weights")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch priority weights: {e}")
        # Return hardcoded fallback defaults (complexity removed from UI, weight=0)
        return {
            "priority_weights": {
                "accuracy": {"low": 3, "medium": 5, "high": 7},
                "cost": {"low": 2, "medium": 4, "high": 6},
                "latency": {"low": 1, "medium": 2, "high": 3},
            },
            "defaults": {"priority": "medium", "weights": {"accuracy": 5, "cost": 4, "latency": 2}}
        }


def fetch_ranked_recommendations(
    use_case: str,
    user_count: int,
    priority: str,
    prompt_tokens: int,
    output_tokens: int,
    expected_qps: float,
    ttft_target_ms: int,
    itl_target_ms: int,
    e2e_target_ms: int,
    weights: dict = None,
    include_near_miss: bool = False,
    percentile: str = "p95",
) -> dict | None:
    """Fetch ranked recommendations from the backend API.

    Args:
        use_case: Use case identifier (e.g., "chatbot_conversational")
        user_count: Number of concurrent users
        priority: UI priority (maps to latency_requirement/budget_constraint)
        prompt_tokens: Input prompt token count
        output_tokens: Output generation token count
        expected_qps: Queries per second
        ttft_target_ms: TTFT SLO target
        itl_target_ms: ITL SLO target
        e2e_target_ms: E2E SLO target
        weights: Optional dict with accuracy, price, latency, complexity weights (0-10)
        include_near_miss: Whether to include near-SLO configurations
        percentile: Which percentile to use for SLO comparison (mean, p90, p95, p99)

    Returns:
        RankedRecommendationsResponse as dict, or None on error
    """
    import requests

    # Map UI priority to backend latency_requirement and budget_constraint
    priority_mapping = {
        "low_latency": {"latency_requirement": "very_high", "budget_constraint": "flexible"},
        "balanced": {"latency_requirement": "high", "budget_constraint": "moderate"},
        "cost_saving": {"latency_requirement": "medium", "budget_constraint": "strict"},
        "high_throughput": {"latency_requirement": "high", "budget_constraint": "moderate"},
        "high_accuracy": {"latency_requirement": "medium", "budget_constraint": "flexible"},
    }

    mapping = priority_mapping.get(priority, priority_mapping["balanced"])

    # Build request payload
    # min_accuracy=35 filters out models with 30% fallback (no AA data)
    payload = {
        "use_case": use_case,
        "user_count": user_count,
        "latency_requirement": mapping["latency_requirement"],
        "budget_constraint": mapping["budget_constraint"],
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "expected_qps": expected_qps,
        "ttft_target_ms": ttft_target_ms,
        "itl_target_ms": itl_target_ms,
        "e2e_target_ms": e2e_target_ms,
        "percentile": percentile,  # mean, p90, p95, p99
        "include_near_miss": include_near_miss,
        "min_accuracy": 35,  # Filter out models without AA accuracy data (30% fallback)
    }

    if weights:
        payload["weights"] = weights

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/ranked-recommend-from-spec",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch ranked recommendations: {e}")
        return None


def render_weight_controls() -> None:
    """Render weight controls and configuration options in a collapsible section.

    Updates session state directly:
    - weight_accuracy, weight_cost, weight_latency (0-10 each)
    - include_near_miss (bool)
    Note: weight_complexity removed from UI (hardcoded to 0)
    """
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("""
            <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">
                <strong>Ranking Weights</strong> (0-10) — adjust to customize balanced score
            </div>
            """, unsafe_allow_html=True)

            wcol1, wcol2, wcol3 = st.columns(3)

            with wcol1:
                accuracy = st.number_input(
                    "Accuracy",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.get("weight_accuracy", 5),
                    key="input_accuracy",
                    help="Higher = prefer more capable models"
                )
                if accuracy != st.session_state.get("weight_accuracy", 5):
                    st.session_state.weight_accuracy = accuracy

            with wcol2:
                cost = st.number_input(
                    "Cost",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.get("weight_cost", 4),
                    key="input_cost",
                    help="Higher = prefer cheaper configurations"
                )
                if cost != st.session_state.get("weight_cost", 4):
                    st.session_state.weight_cost = cost

            with wcol3:
                latency = st.number_input(
                    "Latency",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.get("weight_latency", 2),
                    key="input_latency",
                    help="Higher = prefer lower latency"
                )
                if latency != st.session_state.get("weight_latency", 2):
                    st.session_state.weight_latency = latency

        with col2:
            st.markdown("""
            <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">
                <strong>Options</strong>
            </div>
            """, unsafe_allow_html=True)

            include_near_miss = st.checkbox(
                "Include near-miss configs",
                value=st.session_state.include_near_miss,
                key="checkbox_near_miss",
                help="Include configurations that nearly meet SLO targets"
            )
            if include_near_miss != st.session_state.include_near_miss:
                st.session_state.include_near_miss = include_near_miss

            # Re-evaluate button to apply weight changes (styled as primary/blue)
            if st.button("Re-Evaluate", key="re_evaluate_btn", type="primary", help="Apply weight changes and re-fetch recommendations"):
                st.rerun()


def render_recommendation_category_card(
    category_name: str,
    category_emoji: str,
    category_color: str,
    recommendations: list,
):
    """Render a single recommendation category card with top pick and expander.

    Args:
        category_name: Display name (e.g., "Balanced", "Best Accuracy")
        category_emoji: Emoji for the category
        category_color: Hex color for styling
        recommendations: List of recommendation dicts from backend
    """
    if not recommendations:
        st.markdown(f"""
        <div style="background: var(--bg-card); padding: 1rem; border-radius: 0.75rem;
                    border: 1px solid {category_color}40; min-height: 150px;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.25rem;">{category_emoji}</span>
                <span style="color: {category_color}; font-weight: 700;">{category_name}</span>
            </div>
            <p style="color: rgba(255,255,255,0.5); font-style: italic;">No configurations found</p>
        </div>
        """, unsafe_allow_html=True)
        return

    top = recommendations[0]
    model_name = top.get("model_name", "Unknown")
    gpu_type = top.get("gpu_config", {}).get("gpu_type", "Unknown") if isinstance(top.get("gpu_config"), dict) else "Unknown"
    gpu_count = top.get("gpu_config", {}).get("gpu_count", 1) if isinstance(top.get("gpu_config"), dict) else 1
    ttft = top.get("predicted_ttft_p95_ms", 0)
    cost = top.get("cost_per_month_usd", 0)
    meets_slo = top.get("meets_slo", False)
    scores = top.get("scores", {})
    balanced_score = scores.get("balanced_score", 0) if isinstance(scores, dict) else 0
    accuracy_score = scores.get("accuracy_score", 0) if isinstance(scores, dict) else 0

    slo_badge = "Yes" if meets_slo else "No"

    st.markdown(f"""
    <div style="background: var(--bg-card); padding: 1rem; border-radius: 0.75rem;
                border: 1px solid {category_color}40;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
            <span style="font-size: 1.25rem;">{category_emoji}</span>
            <span style="color: {category_color}; font-weight: 700;">{category_name}</span>
            <span style="margin-left: auto; font-size: 0.9rem;">{slo_badge}</span>
        </div>
        <div style="margin-bottom: 0.5rem;">
            <span style="color: white; font-weight: 600; font-size: 1.1rem;">{model_name}</span>
        </div>
        <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">
            {gpu_count}x {gpu_type}
        </div>
        <div style="display: flex; gap: 1rem; font-size: 0.85rem; color: rgba(255,255,255,0.6);">
            <span>⏱️ {ttft:.0f}ms</span>
            <span>${cost:,.0f}/mo</span>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem;">
            <span style="color: {category_color};">Score: {balanced_score:.1f}</span>
            <span style="color: rgba(255,255,255,0.5);"> | Accuracy: {accuracy_score:.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Expander for additional options
    if len(recommendations) > 1:
        with st.expander(f"Show {len(recommendations) - 1} more options"):
            for rec in recommendations[1:]:
                rec_model = rec.get("model_name", "Unknown")
                rec_gpu_type = rec.get("gpu_config", {}).get("gpu_type", "Unknown") if isinstance(rec.get("gpu_config"), dict) else "Unknown"
                rec_gpu_count = rec.get("gpu_config", {}).get("gpu_count", 1) if isinstance(rec.get("gpu_config"), dict) else 1
                rec_ttft = rec.get("predicted_ttft_p95_ms", 0)
                rec_cost = rec.get("cost_per_month_usd", 0)
                rec_meets_slo = "Yes" if rec.get("meets_slo", False) else "No"
                rec_scores = rec.get("scores", {})
                rec_balanced = rec_scores.get("balanced_score", 0) if isinstance(rec_scores, dict) else 0

                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; background: rgba(255,255,255,0.03);
                            border-radius: 0.5rem; font-size: 0.9rem;">
                    <span style="color: white; font-weight: 500;">{rec_meets_slo} {rec_model}</span>
                    <span style="color: rgba(255,255,255,0.5);"> — {rec_gpu_count}x {rec_gpu_type},
                          {rec_ttft:.0f}ms, ${rec_cost:,.0f}/mo (Score: {rec_balanced:.1f})</span>
                </div>
                """, unsafe_allow_html=True)


def render_ranked_recommendations(response: dict, show_config: bool = True):
    """Render all 5 ranked recommendation categories in a table format.

    Args:
        response: RankedRecommendationsResponse from backend
        show_config: Whether to show the configuration controls (default True)
    """
    total_configs = response.get("total_configs_evaluated", 0)
    configs_after_filters = response.get("configs_after_filters", 0)

    st.markdown(
        '<div class="section-header" style="background: #1a1a1a; border: 1px solid rgba(255,255,255,0.2);">Recommended Solutions</div>',
        unsafe_allow_html=True
    )

    # Render configuration controls right under the header
    if show_config:
        render_weight_controls()

    st.markdown(
        f'<div style="color: #ffffff; margin-bottom: 1rem; font-size: 0.9rem;">Evaluated <span style="color: #06b6d4; font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="color: #10b981; font-weight: 600;">{configs_after_filters}</span> unique options</div>',
        unsafe_allow_html=True
    )

    # Define categories in the requested order
    categories = [
        ("balanced", "Balanced", "", "#EE0000"),
        ("best_accuracy", "Best Accuracy", "", "#ffffff"),
        ("lowest_cost", "Lowest Cost", "", "#f59e0b"),
        ("lowest_latency", "Lowest Latency", "", "#ffffff"),
        ("simplest", "Simplest", "", "#ec4899"),
    ]

    # Helper function to format GPU config with TP and replicas
    def format_gpu_config(gpu_config: dict) -> str:
        if not isinstance(gpu_config, dict):
            return "Unknown"
        gpu_type = gpu_config.get("gpu_type", "Unknown")
        gpu_count = gpu_config.get("gpu_count", 1)
        tp = gpu_config.get("tensor_parallel", 1)
        replicas = gpu_config.get("replicas", 1)
        return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"

    # Get selected percentile for table display
    selected_percentile = st.session_state.get('slo_percentile', 'p95')
    percentile_label = selected_percentile.upper() if selected_percentile != 'mean' else 'Mean'
    
    # Helper function to build a table row from a recommendation
    def build_row(rec: dict, cat_color: str, cat_name: str = "", cat_emoji: str = "", is_top: bool = False, more_count: int = 0) -> str:
        model_name = format_display_name(rec.get("model_name", "Unknown"))
        gpu_config = rec.get("gpu_config", {})
        gpu_str = format_gpu_config(gpu_config)
        # Get TTFT from benchmark_metrics using selected percentile, fallback to p95
        benchmark_metrics = rec.get("benchmark_metrics", {}) or {}
        ttft = benchmark_metrics.get(f'ttft_{selected_percentile}', rec.get("predicted_ttft_p95_ms", 0))
        cost = rec.get("cost_per_month_usd", 0)
        meets_slo = rec.get("meets_slo", False)
        scores = rec.get("scores", {})
        # Use raw AA accuracy from CSV (same as cards) instead of backend score
        use_case = st.session_state.get("detected_use_case", "chatbot_conversational")
        accuracy_score = get_raw_aa_accuracy(model_name, use_case)
        price_score = scores.get("price_score", 0) if isinstance(scores, dict) else 0
        latency_score = scores.get("latency_score", 0) if isinstance(scores, dict) else 0
        complexity_score = scores.get("complexity_score", 0) if isinstance(scores, dict) else 0
        balanced_score = scores.get("balanced_score", 0) if isinstance(scores, dict) else 0
        slo_badge = "Yes" if meets_slo else "No"

        if is_top:
            more_badge = f'<span style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-left: 0.5rem;">(+{more_count})</span>' if more_count > 0 else ''
            cat_cell = f'<td style="padding: 0.75rem 0.5rem;"><span style="color: {cat_color}; font-weight: 600;">{cat_emoji} {cat_name}</span>{more_badge}</td>'
        else:
            cat_cell = f'<td style="padding: 0.75rem 0.5rem; padding-left: 2rem;"><span style="color: {cat_color}40;">↳</span></td>'

        return (
            f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);{"background: rgba(255,255,255,0.02);" if not is_top else ""}">'
            f'{cat_cell}'
            f'<td style="padding: 0.75rem 0.5rem; color: white; font-weight: {"500" if is_top else "400"};">{model_name}</td>'
            f'<td style="padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.8); font-size: 0.85rem;">{gpu_str}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: right; color: #06b6d4;">{ttft:.0f}ms</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: right; color: #f59e0b;">${cost:,.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #10b981;">{accuracy_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #f59e0b;">{price_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #06b6d4;">{latency_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #ec4899;">{complexity_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #8b5cf6; font-weight: 600;">{balanced_score:.1f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center;">{slo_badge}</td>'
            f'</tr>'
        )

    # Table header styles
    th_style = 'style="text-align: left; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;"'
    th_style_right = 'style="text-align: right; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;"'
    th_style_center = 'style="text-align: center; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;"'

    table_header = (
        '<thead>'
        f'<tr style="border-bottom: 2px solid rgba(255,255,255,0.2);">'
        f'<th {th_style}>Category</th>'
        f'<th {th_style}>Model</th>'
        f'<th {th_style}>GPU Config</th>'
        f'<th {th_style_right}>TTFT ({percentile_label})</th>'
        f'<th {th_style_right}>Cost/mo</th>'
        f'<th {th_style_center}>Acc</th>'
        f'<th {th_style_center}>Cost</th>'
        f'<th {th_style_center}>Lat</th>'
        f'<th {th_style_center}>Cap</th>'
        f'<th {th_style_center}>Bal</th>'
        f'<th {th_style_center}>SLO</th>'
        '</tr>'
        '</thead>'
    )

    # Build ONE unified table with all categories
    # Add a first column for expand/collapse buttons rendered as part of the Category cell

    # Collect expansion state and category data
    category_data = []
    for cat_key, cat_name, cat_emoji, cat_color in categories:
        recs = response.get(cat_key, [])
        is_expanded = cat_key in st.session_state.expanded_categories
        more_count = len(recs) - 1 if len(recs) > 1 else 0
        category_data.append({
            "key": cat_key,
            "name": cat_name,
            "emoji": cat_emoji,
            "color": cat_color,
            "recs": recs,
            "is_expanded": is_expanded,
            "more_count": more_count,
        })

    # Render expand/collapse toggle buttons in a compact row
    # These appear above the table, one per category with expandable options
    expandable_cats = [c for c in category_data if c["more_count"] > 0]
    if expandable_cats:
        st.markdown(
            '<div style="margin-bottom: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">',
            unsafe_allow_html=True
        )
        toggle_cols = st.columns(len(expandable_cats) + 1)
        toggle_cols[0].markdown(
            '<span style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Expand:</span>',
            unsafe_allow_html=True
        )
        for idx, cat in enumerate(expandable_cats):
            with toggle_cols[idx + 1]:
                btn_label = f"−{cat['emoji']}" if cat["is_expanded"] else f"+{cat['emoji']}"
                if st.button(
                    btn_label,
                    key=f"toggle_{cat['key']}",
                    help=f"{'Collapse' if cat['is_expanded'] else 'Expand'} {cat['name']} (+{cat['more_count']})"
                ):
                    if cat["is_expanded"]:
                        st.session_state.expanded_categories.discard(cat["key"])
                    else:
                        st.session_state.expanded_categories.add(cat["key"])
                    st.rerun()

    # Build all rows for the unified table
    all_rows = []

    for cat in category_data:
        cat_key = cat["key"]
        cat_name = cat["name"]
        cat_emoji = cat["emoji"]
        cat_color = cat["color"]
        recs = cat["recs"]
        is_expanded = cat["is_expanded"]
        more_count = cat["more_count"]

        if not recs:
            # Empty category row
            all_rows.append(
                f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">'
                f'<td style="padding: 0.75rem 0.5rem;"><span style="color: {cat_color}; font-weight: 600;">{cat_emoji} {cat_name}</span></td>'
                f'<td colspan="10" style="padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.5); font-style: italic;">No configurations found</td>'
                f'</tr>'
            )
        else:
            # Top recommendation for this category
            all_rows.append(build_row(recs[0], cat_color, cat_name, cat_emoji, is_top=True, more_count=more_count))

            # Add additional rows if expanded
            if is_expanded and len(recs) > 1:
                for rec in recs[1:]:
                    all_rows.append(build_row(rec, cat_color, "", "", is_top=False, more_count=0))

    # Render the single unified table
    unified_table_html = (
        f'<table style="width: 100%; border-collapse: collapse;">'
        + table_header +
        '<tbody>' + ''.join(all_rows) + '</tbody>'
        '</table>'
    )

    st.markdown(unified_table_html, unsafe_allow_html=True)


def get_slo_for_model(model_name: str, use_case: str, hardware: str = "H100") -> dict:
    """Get REAL benchmark SLO data for a specific model and use case.
    
    IMPORTANT: Only returns data if we have ACTUAL benchmarks for this model.
    Returns None if no matching benchmark data exists (we don't fake data).
    
    Benchmark dataset contains only these models:
    - qwen2.5-7b, llama-3.1-8b, llama-3.3-70b, phi-4
    - mistral-small-24b, mixtral-8x7b, granite-3.1-8b
    - gpt-oss-120b, gpt-oss-20b
    """
    benchmark_data = load_performance_benchmarks()
    if not benchmark_data or 'benchmarks' not in benchmark_data:
        return None
    
    benchmarks = benchmark_data['benchmarks']
    
    # Map use case to token config
    token_configs = {
        "code_completion": (512, 256),
        "chatbot_conversational": (512, 256),
        "code_generation_detailed": (1024, 1024),
        "translation": (512, 256),
        "content_generation": (512, 256),
        "summarization_short": (4096, 512),
        "document_analysis_rag": (4096, 512),
        "long_document_summarization": (10240, 1536),
        "research_legal_analysis": (10240, 1536),
    }
    
    prompt_tokens, output_tokens = token_configs.get(use_case, (512, 256))
    
    # STRICT mapping: Only map if we're confident it's the same model family
    # Key = pattern to find in recommended model name, Value = benchmark repo
    model_mapping = {
        "qwen2.5": "qwen/qwen2.5-7b-instruct",
        "qwen 2.5": "qwen/qwen2.5-7b-instruct",
        "llama 3.1 8b": "meta-llama/llama-3.1-8b-instruct",
        "llama 3.1 instruct 8b": "meta-llama/llama-3.1-8b-instruct",
        "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
        "llama 3.3": "meta-llama/llama-3.3-70b-instruct",
        "llama-3.3": "meta-llama/llama-3.3-70b-instruct",
        "granite": "ibm-granite/granite-3.1-8b-instruct",
        "phi-4": "microsoft/phi-4",
        "phi 4": "microsoft/phi-4",
        "mistral small": "mistralai/mistral-small-24b-instruct-2501",
        "mistral-small": "mistralai/mistral-small-24b-instruct-2501",
        "mixtral": "mistralai/mixtral-8x7b-instruct-v0.1",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
    }
    
    # Find matching model in benchmark data - STRICT matching
    model_lower = model_name.lower()
    benchmark_model_repo = None
    is_exact_match = False
    
    for key, repo in model_mapping.items():
        if key in model_lower:
            benchmark_model_repo = repo
            is_exact_match = True
            break
    
    # If no exact match, return None - we don't want to show misleading data
    if not benchmark_model_repo:
        return None
    
    # Filter benchmarks by token config
    matching = [b for b in benchmarks 
                if b['prompt_tokens'] == prompt_tokens 
                and b['output_tokens'] == output_tokens]
    
    # MUST match the specific model - no fallbacks
    model_matches = [b for b in matching if b['model_hf_repo'] == benchmark_model_repo]
    if not model_matches:
        return None  # No benchmark data for this model
    
    matching = model_matches
    
    # Filter by hardware if specified
    if hardware:
        hw_matches = [b for b in matching if hardware.upper() in b['hardware'].upper()]
        if hw_matches:
            matching = hw_matches
    
    if not matching:
        return None
    
    # Get best benchmark (lowest E2E at reasonable RPS)
    matching.sort(key=lambda x: x['e2e_mean'])
    best = matching[0]
    
    return {
        "model_repo": best['model_hf_repo'],
        "is_exact_match": is_exact_match,
        "recommended_model": model_name,
        "hardware": best['hardware'],
        "hardware_count": best['hardware_count'],
        "token_config": {"prompt": prompt_tokens, "output": output_tokens},
        "slo_actual": {
            "ttft_mean_ms": round(best['ttft_mean'], 1),
            "ttft_p95_ms": round(best['ttft_p95'], 1),
            "ttft_p99_ms": round(best['ttft_p99'], 1),
            "itl_mean_ms": round(best['itl_mean'], 1),
            "itl_p95_ms": round(best['itl_p95'], 1),
            "itl_p99_ms": round(best['itl_p99'], 1),
            "e2e_mean_ms": round(best['e2e_mean'], 0),
            "e2e_p95_ms": round(best['e2e_p95'], 0),
            "e2e_p99_ms": round(best['e2e_p99'], 0),
        },
        "throughput": {
            "tokens_per_sec": round(best['tokens_per_second'], 0),
            "requests_per_sec": round(best['requests_per_second'], 2),
            "responses_per_sec": round(best['responses_per_second'], 2),
        },
        "benchmark_samples": len(matching),
        "ranges": {
            "ttft_range_ms": [round(min(b['ttft_mean'] for b in matching), 1), 
                             round(max(b['ttft_mean'] for b in matching), 1)],
            "e2e_range_ms": [round(min(b['e2e_mean'] for b in matching), 0),
                           round(max(b['e2e_mean'] for b in matching), 0)],
        }
    }

def get_workload_insights(use_case: str, qps: int, user_count: int) -> list:
    """Get workload pattern insights based on research data and benchmarks.
    
    Returns list of tuples: (icon, color, message, severity)
    """
    messages = []
    workload_data = load_research_workload_patterns()
    
    if not workload_data or 'workload_distributions' not in workload_data:
        return messages
    
    workload_patterns = workload_data.get('workload_distributions', {})
    traffic_profiles = workload_data.get('traffic_profiles', {})
    hardware_throughput = workload_data.get('hardware_throughput', {})
    capacity_guidance = workload_data.get('capacity_planning', {}).get('blis_guidance', {})
    
    # Get use case specific pattern
    pattern = workload_patterns.get(use_case)
    traffic = traffic_profiles.get(use_case)
    
    if pattern:
        distribution = pattern.get('distribution', 'poisson')
        active_fraction = pattern.get('active_fraction', {}).get('mean', 0.2)
        peak_multiplier = pattern.get('peak_multiplier', 2.0)
        req_per_min = pattern.get('requests_per_active_user_per_min', {}).get('mean', 0.5)
        
        # Get benchmark data for this use case
        benchmark_perf = pattern.get('benchmark_perfmark', {})
        benchmark_optimal_rps = benchmark_perf.get('optimal_rps', 1.0)
        benchmark_max_rps = benchmark_perf.get('max_rps_tested', 10)
        benchmark_e2e_p95 = benchmark_perf.get('e2e_p95_at_optimal', 5000)
        
        # Calculate expected metrics
        expected_concurrent = int(user_count * active_fraction)
        expected_rps = (expected_concurrent * req_per_min) / 60
        expected_peak_rps = expected_rps * peak_multiplier
        
        messages.append((
            "", "#000000",
            f"Pattern: {distribution.replace('_', ' ').title()} | {int(active_fraction*100)}% concurrent users",
            "info"
        ))
        
        # Note: Peak multiplier info now shown inline in workload profile box
    
    if traffic:
        prompt_tokens = traffic.get('prompt_tokens', 512)
        output_tokens = traffic.get('output_tokens', 256)
        # Note: Token profile info now shown inline in workload profile box
    
    # Hardware recommendations moved to Recommendation tab (uses benchmark data)
    
    return messages

@st.cache_data
def load_weighted_scores(use_case: str) -> pd.DataFrame:
    """Load use-case-specific weighted scores from backend API."""
    try:
        # Fetch weighted scores from backend API
        response = requests.get(
            f"{API_BASE_URL}/api/v1/weighted-scores/{use_case}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Convert JSON to DataFrame
        if data.get("success") and data.get("scores"):
            df = pd.DataFrame(data["scores"])
            return df
        else:
            logger.warning(f"No weighted scores returned from API for use case: {use_case}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load weighted scores from API for {use_case}: {e}")
        return pd.DataFrame()

# Model name mapping from benchmark/backend names to AA CSV names (exact mapping)
BENCHMARK_TO_AA_NAME_MAP = {
    # GPT-OSS - specific size mapping
    "gpt-oss-120b": "gpt-oss-120b (high)",
    "gpt-oss 120b": "gpt-oss-120b (high)",
    "gpt-oss-20b": "gpt-oss-20b (high)",
    "gpt-oss 20b": "gpt-oss-20b (high)",
    # Llama models
    "llama-4-maverick-17b-128e-instruct-fp8": "llama 4 maverick",
    "llama-4-scout-17b-16e-instruct": "llama 4 scout",
    "llama-4-scout-17b-16e-instruct-fp8-dynamic": "llama 4 scout",
    "llama-3.3-70b-instruct": "llama 3.3 instruct 70b",
    # Phi
    "phi-4": "phi-4",
    "phi-4-fp8-dynamic": "phi-4",
    # Mistral
    "mistral-small-24b-instruct-2501": "mistral small 3",
    "mistral-small-3.1-24b-instruct-2503": "mistral small 3.1",
    "mistral-small-3.1-24b-instruct-2503-fp8-dynamic": "mistral small 3.1",
    "mixtral-8x7b-instruct-v0.1": "mixtral 8x7b instruct",
    # Qwen
    "qwen2.5-7b-instruct": "qwen2.5 7b instruct",
    "qwen2.5-7b-instruct-fp8-dynamic": "qwen2.5 7b instruct",
}

def get_raw_aa_accuracy(model_name: str, use_case: str) -> float:
    """Get raw AA benchmark accuracy for a model from the weighted scores CSV.
    
    This returns the actual benchmark score, NOT the composite quality score.
    """
    df = load_weighted_scores(use_case)
    if df.empty:
        return 0.0
    
    # Normalize model name - remove extra spaces, convert to lowercase
    model_lower = model_name.lower().strip().replace('  ', ' ')
    
    # Extract size identifier (e.g., "120b", "20b", "70b") for differentiation
    import re
    size_match = re.search(r'(\d+)b', model_lower)
    model_size = size_match.group(1) if size_match else None
    
    # Try direct mapping first
    aa_name = BENCHMARK_TO_AA_NAME_MAP.get(model_lower)
    if not aa_name:
        # Try with dashes converted to spaces
        aa_name = BENCHMARK_TO_AA_NAME_MAP.get(model_lower.replace('-', ' '))
    if not aa_name:
        aa_name = model_lower
    
    # Look for EXACT model in CSV (case-insensitive)
    for _, row in df.iterrows():
        csv_model = str(row.get('Model Name', row.get('model_name', ''))).lower().strip()
        
        # Exact match with mapped name
        if csv_model == aa_name.lower():
            score_str = str(row.get('Use Case Score', row.get('Weighted Score', '0')))
            try:
                return float(score_str.replace('%', ''))
            except:
                return 0.0
    
    # Partial match - but must match SIZE to avoid 120B/20B confusion
    for _, row in df.iterrows():
        csv_model = str(row.get('Model Name', row.get('model_name', ''))).lower().strip()
        
        # Check if base model name matches AND size matches
        base_name = model_lower.replace('-', ' ').replace('_', ' ').split()[0] if model_lower else ""
        
        if base_name and base_name in csv_model:
            # Verify size matches to avoid 120B vs 20B confusion
            csv_size_match = re.search(r'(\d+)b', csv_model)
            csv_size = csv_size_match.group(1) if csv_size_match else None
            
            if model_size and csv_size and model_size == csv_size:
                # Size matches - this is the right model
                score_str = str(row.get('Use Case Score', row.get('Weighted Score', '0')))
                try:
                    return float(score_str.replace('%', ''))
                except:
                    return 0.0
            elif not model_size and not csv_size:
                # No size in either - match on name
                score_str = str(row.get('Use Case Score', row.get('Weighted Score', '0')))
                try:
                    return float(score_str.replace('%', ''))
                except:
                    return 0.0
    
    return 0.0

@st.cache_data
def load_model_pricing() -> pd.DataFrame:
    """Load model pricing and latency data from model_pricing.csv.
    
    This provides REAL data for:
    - Cost scoring: price_blended ($/1M tokens)
    - Latency scoring: median_output_tokens_per_sec, median_ttft_seconds
    """
    csv_path = DATA_DIR / "benchmarks" / "models" / "model_pricing.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['model_name'])
        df = df[df['model_name'].str.strip() != '']
        return df
    except Exception:
        return pd.DataFrame()

# =============================================================================
# API FUNCTIONS (with Mock fallback for demo)
# =============================================================================

from typing import Optional, Dict, List

def extract_business_context(user_input: str) -> Optional[dict]:
    """Extract business context using Qwen 2.5 7B."""
    try:
        logger.info(f"Calling LLM extraction API at {API_BASE_URL}/api/v1/extract")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract",
            json={"text": user_input},
            timeout=60,
        )
        if response.status_code == 200:
            result = response.json()
            # Map preferred_gpu_type to hardware for UI compatibility
            if 'preferred_gpu_type' in result:
                gpu = result['preferred_gpu_type']
                result['hardware'] = None if gpu == "Any GPU" else gpu
            logger.info(f"LLM extraction successful: {result.get('use_case')}, hardware={result.get('hardware')}, priorities: acc={result.get('accuracy_priority')}, cost={result.get('cost_priority')}, lat={result.get('latency_priority')}, comp={result.get('complexity_priority')}")
            return result
        else:
            logger.warning(f"LLM extraction API returned status {response.status_code}: {response.text[:200]}")
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Backend not reachable: {e}")
    except requests.exceptions.Timeout as e:
        logger.warning(f"LLM extraction timed out: {e}")
    except Exception as e:
        logger.warning(f"LLM extraction failed: {type(e).__name__}: {e}")

    # Mock response for demo
    logger.info("Falling back to mock extraction")
    return mock_extraction(user_input)


def mock_extraction(user_input: str) -> dict:
    """Mock extraction for demo purposes - supports all 9 use cases with robust error handling."""
    import re
    
    # Input validation - handle edge cases
    if not user_input or not isinstance(user_input, str):
        return {
            "use_case": "chatbot_conversational",
            "user_count": 100,
            "hardware": None,
            "priority": "balanced",
            "accuracy_priority": "medium",
            "cost_priority": "medium",
            "latency_priority": "medium",
            "complexity_priority": "medium",
        }
    
    # Clean and normalize input
    text_lower = user_input.lower().strip()
    
    # USE CASE DETECTION - Order matters! More specific patterns first
    use_case = "chatbot_conversational"  # default
    
    # 1. Translation - check first (very specific)
    if any(kw in text_lower for kw in ["translat", "language pair", "multilingual", "locali"]):
        use_case = "translation"
    
    # 2. Long Document Summarization - check before short summarization
    elif any(kw in text_lower for kw in ["long summar", "long document", "book summar", "report summar", 
                                          "chapter", "extensive summar", "lengthy", "50+ page", "research paper"]):
        use_case = "long_document_summarization"
    
    # 3. Short Summarization - check BEFORE content generation (article summarization != content generation)
    elif any(kw in text_lower for kw in ["summar", "tldr", "brief", "condense", "digest", "news summar"]):
        use_case = "summarization_short"
    
    # 4. Content Generation - check before code
    elif any(kw in text_lower for kw in ["content generat", "content creation", "creative writ", "marketing content", 
                                          "blog post", "copywriting", "content tool"]):
        use_case = "content_generation"
    
    # 5. Code Generation (detailed) - check before code completion
    elif any(kw in text_lower for kw in ["code generat", "full code", "implement", "build software", 
                                          "create application", "write program"]):
        use_case = "code_generation_detailed"
    
    # 6. Code Completion - IDE autocomplete
    elif any(kw in text_lower for kw in ["code complet", "autocomplete", "code assist", "ide", "copilot",
                                          "code suggestion", "developer tool"]):
        use_case = "code_completion"
    
    # 7. Research/Legal Analysis
    elif any(kw in text_lower for kw in ["legal", "research", "analys", "contract", "compliance", 
                                          "academic", "scientific", "review paper"]):
        use_case = "research_legal_analysis"
    
    # 8. Document RAG
    elif any(kw in text_lower for kw in ["rag", "retriev", "knowledge base", "document q&a", 
                                          "document search", "enterprise search"]):
        use_case = "document_analysis_rag"
    
    # 9. Chatbot Conversational (default for chat-related)
    elif any(kw in text_lower for kw in ["chatbot", "chat", "customer support", "virtual assist", 
                                          "conversation", "support bot", "help desk"]):
        use_case = "chatbot_conversational"
    
    # Detect user count - look for numbers followed by user-related words
    # Avoid matching "50+ pages", "7B params", "80GB", etc.
    user_patterns = [
        r'(\d+[,.]?\d*)\s*k?\s*(users|developers|researchers|engineers|team members|people|employees|customers|daily users)',
        r'(\d+[,.]?\d*)\s*(k|thousand)\s*(users|developers|researchers|people)?',
        r'for\s+(\d+[,.]?\d*)\s*(k)?\s*(users|developers|researchers|people)?',
        r'(\d+[,.]?\d*)\s+team',
    ]
    
    user_count = 1000  # default
    for pattern in user_patterns:
        user_match = re.search(pattern, text_lower)
        if user_match:
            num_str = user_match.group(1).replace(',', '')
            num = float(num_str)
            # Check if 'k' or 'thousand' is in the match
            full_match = user_match.group(0).lower()
            if 'k ' in full_match or 'k\t' in full_match or full_match.endswith('k') or 'thousand' in full_match:
                num *= 1000
            user_count = int(num)
            break
    
    # Detect hardware preference
    hardware = None
    if "h200" in text_lower:
        hardware = "H200"
    elif "h100" in text_lower:
        hardware = "H100"
    elif "a100" in text_lower:
        hardware = "A100"
    elif "l4" in text_lower:
        hardware = "L4"
    elif "l40" in text_lower:
        hardware = "L40S"
    elif "b200" in text_lower:
        hardware = "B200"
    
    # Detect priority from user input
    priority = "balanced"  # default
    
    # Quality keywords - check these FIRST (accuracy is more specific than generic "critical")
    quality_keywords = ["accuracy", "accurate", "quality", "precision", "high quality", "top quality", 
                        "accuracy is critical", "quality is critical", "quality is most important",
                        "accuracy is most important", "best quality", "highest accuracy"]
    
    # Latency keywords - "critical" removed (too generic)
    latency_keywords = ["latency", "fast", "speed", "quick", "responsive", "real-time", "instant", 
                        "low latency", "latency is critical", "under 200ms", "under 100ms", "millisecond"]
    
    cost_keywords = ["cost", "cheap", "budget", "efficient", "affordable", "save money", "cost-effective",
                     "budget is tight", "minimize cost"]
    
    throughput_keywords = ["throughput", "scale", "high volume", "capacity", "concurrent", "many users",
                           "high traffic", "peak load"]
    
    # Check for QUALITY priority FIRST (most specific signals)
    if any(kw in text_lower for kw in quality_keywords):
        priority = "high_accuracy"
    # Check for latency priority
    elif any(kw in text_lower for kw in latency_keywords):
        priority = "low_latency"
    # Check for cost priority
    elif any(kw in text_lower for kw in cost_keywords):
        priority = "cost_saving"
    # Check for throughput priority
    elif any(kw in text_lower for kw in throughput_keywords):
        priority = "high_throughput"
    
    # Detect priority hints for each dimension (default to "medium")
    accuracy_priority = "medium"
    cost_priority = "medium"
    latency_priority = "medium"
    complexity_priority = "medium"

    # Accuracy priority detection
    if any(kw in text_lower for kw in ["accuracy is important", "quality matters", "best model",
                                        "highest accuracy", "accuracy critical", "top quality"]):
        accuracy_priority = "high"
    elif any(kw in text_lower for kw in ["accuracy less important", "good enough", "accuracy not critical"]):
        accuracy_priority = "low"

    # Cost priority detection
    if any(kw in text_lower for kw in ["cost is important", "budget constrained", "cost-efficient",
                                        "cost critical", "minimize cost", "budget is tight"]):
        cost_priority = "high"
    elif any(kw in text_lower for kw in ["not cost sensitive", "budget flexible", "money not an issue",
                                          "cost doesn't matter"]):
        cost_priority = "low"

    # Latency priority detection
    if any(kw in text_lower for kw in ["latency is a concern", "speed matters", "fast response",
                                        "low latency critical", "real-time", "instant"]):
        latency_priority = "high"
    elif any(kw in text_lower for kw in ["latency less important", "can wait", "latency not critical"]):
        latency_priority = "low"

    # Complexity priority detection
    if any(kw in text_lower for kw in ["simple deployment", "easy to manage", "minimal complexity",
                                        "simple setup", "straightforward"]):
        complexity_priority = "high"
    elif any(kw in text_lower for kw in ["complexity is fine", "advanced setup ok", "complex ok"]):
        complexity_priority = "low"

    return {
        "use_case": use_case,
        "user_count": user_count,
        "hardware": hardware,
        "priority": priority,
        "accuracy_priority": accuracy_priority,
        "cost_priority": cost_priority,
        "latency_priority": latency_priority,
        "complexity_priority": complexity_priority,
    }


def get_enhanced_recommendation(business_context: dict) -> Optional[dict]:
    """Get enhanced recommendation with explainability.

    ALL logic is in backend - UI only fetches data.
    """
    try:
        # Extract values from business_context for the API
        use_case = business_context.get("use_case", "chatbot_conversational")
        priority = business_context.get("priority", "balanced")
        user_count = business_context.get("user_count", 1000)
        prompt_tokens = business_context.get("prompt_tokens", 512)
        output_tokens = business_context.get("output_tokens", 256)
        expected_qps = business_context.get("expected_qps", 1)  # Default to 1 RPS, not user_count!
        ttft_target = business_context.get("ttft_p95_target_ms", 500)
        itl_target = business_context.get("itl_p95_target_ms", 50)
        e2e_target = business_context.get("e2e_p95_target_ms", 10000)
        percentile = business_context.get("percentile", "p95")

        # Get weights from session state (set in PRIORITIES section of Tech Spec tab)
        weights = {
            "accuracy": st.session_state.get("weight_accuracy", 5),
            "price": st.session_state.get("weight_cost", 4),
            "latency": st.session_state.get("weight_latency", 2),
            "complexity": 0,  # Complexity scoring disabled in UI (backend logic preserved)
        }

        # Use the ranked-recommend-from-spec endpoint with proper fields
        response = requests.post(
            f"{API_BASE_URL}/api/ranked-recommend-from-spec",
            json={
                "use_case": use_case,
                "user_count": user_count,
                "latency_requirement": "high",
                "budget_constraint": "moderate",
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "expected_qps": expected_qps,
                "ttft_p95_target_ms": ttft_target,  # Fixed: was ttft_target_ms
                "itl_p95_target_ms": itl_target,    # Fixed: was itl_target_ms
                "e2e_p95_target_ms": e2e_target,    # Fixed: was e2e_target_ms
                "percentile": percentile,
                "include_near_miss": False,  # Strict filtering!
                "min_accuracy": 35,
                "weights": weights,  # Pass user-configured weights for balanced scoring
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API returned status {response.status_code}: {response.text[:500]}")
    except Exception as e:
        st.error(f"Backend call failed: {e}")
    
    # Fallback only when backend unavailable - THIS DOESN'T RESPECT SLO FILTERS!
    st.warning("⚠️ Using fallback - SLO filters may not be applied!")
    return benchmark_recommendation(business_context)


# =============================================================================
# VALID MODELS - Only models with BOTH AA Quality AND Performance benchmark data
# These 25 variants are the only ones we should recommend (have both AA quality + benchmark performance)
# =============================================================================
VALID_BENCHMARK_MODELS = {
    # GPT-OSS (highest accuracy for chatbot!)
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
    # Phi-4 variants
    'microsoft/phi-4',
    'microsoft/phi-4-fp8-dynamic',
    'microsoft/phi-4-quantized.w4a16',
    'microsoft/phi-4-quantized.w8a8',
    # Mistral Small 3/3.1 variants
    'mistralai/mistral-small-24b-instruct-2501',
    'mistralai/mistral-small-3.1-24b-instruct-2503',
    'mistralai/mistral-small-3.1-24b-instruct-2503-fp8-dynamic',
    'mistralai/mistral-small-3.1-24b-instruct-2503-quantized.w4a16',
    'mistralai/mistral-small-3.1-24b-instruct-2503-quantized.w8a8',
    # Mixtral 8x7B
    'mistralai/mixtral-8x7b-instruct-v0.1',
    # Llama 4 Scout variants
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'meta-llama/llama-4-scout-17b-16e-instruct-fp8-dynamic',
    'meta-llama/llama-4-scout-17b-16e-instruct-quantized.w4a16',
    # Llama 4 Maverick
    'meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
    # Qwen 2.5 7B variants (note: quantized use redhatai/ prefix)
    'qwen/qwen2.5-7b-instruct',
    'redhatai/qwen2.5-7b-instruct-fp8-dynamic',
    'redhatai/qwen2.5-7b-instruct-quantized.w4a16',
    'redhatai/qwen2.5-7b-instruct-quantized.w8a8',
    # Llama 3.3 70B variants (note: quantized use redhatai/ prefix)
    'meta-llama/llama-3.3-70b-instruct',
    'redhatai/llama-3.3-70b-instruct-quantized.w4a16',
    'redhatai/llama-3.3-70b-instruct-quantized.w8a8',
}

# Maps benchmark repo names to AA quality CSV model names
BENCHMARK_TO_QUALITY_MODEL_MAP = {
    # GPT-OSS (highest accuracy)
    'openai/gpt-oss-120b': 'gpt-oss-120B (high)',
    'openai/gpt-oss-20b': 'gpt-oss-20B (high)',
    # Phi-4
    'microsoft/phi-4': 'Phi-4',
    'microsoft/phi-4-fp8-dynamic': 'Phi-4',
    'microsoft/phi-4-quantized.w4a16': 'Phi-4',
    'microsoft/phi-4-quantized.w8a8': 'Phi-4',
    # Mistral Small
    'mistralai/mistral-small-24b-instruct-2501': 'Mistral Small 3',
    'mistralai/mistral-small-3.1-24b-instruct-2503': 'Mistral Small 3.1',
    'mistralai/mistral-small-3.1-24b-instruct-2503-fp8-dynamic': 'Mistral Small 3.1',
    'mistralai/mistral-small-3.1-24b-instruct-2503-quantized.w4a16': 'Mistral Small 3.1',
    'mistralai/mistral-small-3.1-24b-instruct-2503-quantized.w8a8': 'Mistral Small 3.1',
    'mistralai/mixtral-8x7b-instruct-v0.1': 'Mixtral 8x7B Instruct',
    # Llama 4
    'meta-llama/llama-4-scout-17b-16e-instruct': 'Llama 4 Scout',
    'meta-llama/llama-4-scout-17b-16e-instruct-fp8-dynamic': 'Llama 4 Scout',
    'meta-llama/llama-4-scout-17b-16e-instruct-quantized.w4a16': 'Llama 4 Scout',
    'meta-llama/llama-4-maverick-17b-128e-instruct-fp8': 'Llama 4 Maverick',
    # Qwen 2.5 7B (note: quantized use redhatai/ prefix)
    'qwen/qwen2.5-7b-instruct': 'Qwen2.5 Max',
    'redhatai/qwen2.5-7b-instruct-fp8-dynamic': 'Qwen2.5 Max',
    'redhatai/qwen2.5-7b-instruct-quantized.w4a16': 'Qwen2.5 Max',
    'redhatai/qwen2.5-7b-instruct-quantized.w8a8': 'Qwen2.5 Max',
    # Llama 3.3 70B (note: quantized use redhatai/ prefix)
    'meta-llama/llama-3.3-70b-instruct': 'Llama 3.3 Instruct 70B',
    'redhatai/llama-3.3-70b-instruct-quantized.w4a16': 'Llama 3.3 Instruct 70B',
    'redhatai/llama-3.3-70b-instruct-quantized.w8a8': 'Llama 3.3 Instruct 70B',
}

# Hardware costs (monthly) - BOTH H100 and A100-80 are real benchmark data
HARDWARE_COSTS = {
    ('H100', 1): 2500,
    ('H100', 2): 5000,
    ('H100', 4): 10000,
    ('H100', 8): 20000,
    ('A100-80', 1): 1600,
    ('A100-80', 2): 3200,
    ('A100-80', 4): 6400,
}


def benchmark_recommendation(context: dict) -> dict:
    """Benchmark-based recommendation using ACTUAL benchmark data.
    
    NEW ARCHITECTURE:
    - Model quality: from weighted_scores CSVs (use-case specific)
    - Latency/throughput: from ACTUAL benchmarks (model+hardware specific)
    - Cost: from hardware tier (cheaper hardware = higher cost score)
    
    Creates MODEL+HARDWARE combinations ranked by priority:
    - cost_saving: cheapest hardware that meets SLO for best models
    - low_latency: fastest hardware (lowest TTFT) for best models
    - high_accuracy: best model accuracy with hardware that meets SLO
    - balanced: weighted combination of all factors
    """
    use_case = context.get("use_case", "chatbot_conversational")
    priority = context.get("priority", "balanced")
    user_count = context.get("user_count", 1000)
    
    # Load performance benchmark data
    benchmark_data = load_performance_benchmarks()
    if not benchmark_data or 'benchmarks' not in benchmark_data:
        return mock_recommendation_fallback(context)
    
    benchmarks = benchmark_data['benchmarks']
    
    # Load quality scores for this use case
    weighted_df = load_weighted_scores(use_case)
    quality_lookup = {}
    if not weighted_df.empty:
        for _, row in weighted_df.iterrows():
            model_name = row.get('Model Name', '')
            score_str = row.get('Use Case Score', '0')
            if isinstance(score_str, str):
                score = float(score_str.replace('%', '')) if '%' in score_str else float(score_str) * 100
            else:
                score = float(score_str) * 100 if score_str <= 1 else float(score_str)
            quality_lookup[model_name] = score
    
    # Get SLO targets for this use case
    slo_data = get_slo_targets_for_use_case(use_case, priority)
    if slo_data is None:
        # Fallback defaults if use case not found
        slo_data = {
            'ttft_target': {'min': 50, 'max': 5000},
            'e2e_target': {'min': 500, 'max': 60000},
        }
    ttft_max = slo_data.get('ttft_target', {}).get('max', 200)
    e2e_max = slo_data.get('e2e_target', {}).get('max', 5000)
    
    # Priority weights for MCDM
    weights = {
        "balanced": {"accuracy": 0.30, "latency": 0.30, "cost": 0.25, "throughput": 0.15},
        "low_latency": {"accuracy": 0.15, "latency": 0.50, "cost": 0.15, "throughput": 0.20},
        "cost_saving": {"accuracy": 0.20, "latency": 0.15, "cost": 0.50, "throughput": 0.15},
        "high_accuracy": {"accuracy": 0.50, "latency": 0.20, "cost": 0.15, "throughput": 0.15},
        "high_throughput": {"accuracy": 0.15, "latency": 0.15, "cost": 0.15, "throughput": 0.55},
    }[priority]
    
    # Aggregate benchmark data by model+hardware (use best config per combo)
    # FILTER: Only include models that have BOTH AA quality AND benchmark performance data
    model_hw_combos = {}
    for b in benchmarks:
        model_repo = b['model_hf_repo']
        
        # Skip models not in our valid list (must have both AA + benchmark data)
        if model_repo not in VALID_BENCHMARK_MODELS:
            continue
            
        hw = b['hardware']
        hw_count = b['hardware_count']
        key = (model_repo, hw, hw_count)
        
        # Keep the benchmark with lowest TTFT for each combo
        if key not in model_hw_combos or b['ttft_p95'] < model_hw_combos[key]['ttft_p95']:
            model_hw_combos[key] = {
                'model_repo': model_repo,
                'model_name': model_repo.split('/')[-1],
                'hardware': hw,
                'hardware_count': hw_count,
                'ttft_mean': b['ttft_mean'],
                'ttft_p95': b['ttft_p95'],
                'itl_mean': b['itl_mean'],
                'itl_p95': b['itl_p95'],
                'e2e_mean': b['e2e_mean'],
                'e2e_p95': b['e2e_p95'],
                'tokens_per_second': b['tokens_per_second'],
                'prompt_tokens': b['prompt_tokens'],
                'output_tokens': b['output_tokens'],
            }
    
    # Calculate scores for each model+hardware combo
    scored_combos = []
    
    # Find max values for normalization
    max_ttft = max(c['ttft_p95'] for c in model_hw_combos.values()) or 1
    max_tps = max(c['tokens_per_second'] for c in model_hw_combos.values()) or 1
    max_cost = max(HARDWARE_COSTS.values()) or 1
    
    for key, combo in model_hw_combos.items():
        # Get quality score from CSV (mapped model name)
        quality_model = BENCHMARK_TO_QUALITY_MODEL_MAP.get(combo['model_repo'], combo['model_name'])
        quality_score = quality_lookup.get(quality_model, 30.0)  # Default 30 if not found
        
        # Latency score: lower TTFT = higher score (inverted, normalized 0-100)
        latency_score = 100 - (combo['ttft_p95'] / max_ttft * 100)
        latency_score = max(10, min(100, latency_score))
        
        # Throughput score: higher TPS = higher score
        throughput_score = (combo['tokens_per_second'] / max_tps) * 100
        throughput_score = max(10, min(100, throughput_score))
        
        # Cost score: lower hardware cost = higher score (inverted)
        hw_cost = HARDWARE_COSTS.get((combo['hardware'], combo['hardware_count']), 10000)
        cost_score = 100 - (hw_cost / max_cost * 80)  # Leave headroom
        cost_score = max(10, min(100, cost_score))
        
        # Check if meets SLO
        meets_slo = combo['ttft_p95'] <= ttft_max and combo['e2e_p95'] <= e2e_max
        
        # Calculate weighted MCDM score
        final_score = (
            weights['accuracy'] * quality_score +
            weights['latency'] * latency_score +
            weights['cost'] * cost_score +
            weights['throughput'] * throughput_score
        )
        
        # Bonus for meeting SLO
        if meets_slo:
            final_score += 5
        
        scored_combos.append({
            **combo,
            'quality_score': round(quality_score, 1),
            'latency_score': round(latency_score, 1),
            'cost_score': round(cost_score, 1),
            'throughput_score': round(throughput_score, 1),
            'final_score': round(final_score, 2),
            'meets_slo': meets_slo,
            'hw_cost_monthly': hw_cost,
            'quality_model_name': quality_model,
        })
    
    # Sort by final score (highest first)
    scored_combos.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Get top recommendation
    if not scored_combos:
        return mock_recommendation_fallback(context)
    
    top = scored_combos[0]
    
    # Build recommendation response
    return {
        "model_name": format_display_name(top['model_name']),
        "model_hf_repo": top['model_repo'],
        "score": top['final_score'],
        "intent": {
            "use_case": use_case,
            "priority": priority,
            "user_count": user_count,
        },
        "gpu_config": {
            "gpu_type": top['hardware'],
            "gpu_count": top['hardware_count'],
        },
        "meets_slo": top['meets_slo'],
        "slo_targets": slo_data,
        "hardware_recommendation": {
            "recommended": {
                "hardware": top['hardware'],
                "hardware_count": top['hardware_count'],
                "ttft_p95": top['ttft_p95'],
                "itl_mean": top['itl_mean'],
                "e2e_p95": top['e2e_p95'],
                "tokens_per_second": top['tokens_per_second'],
                "cost_monthly": top['hw_cost_monthly'],
            },
            "selection_reason": get_selection_reason(top, priority),
            "alternatives": [
                {
                    "model": format_display_name(c['model_name']),
                    "hardware": c['hardware'],
                    "hardware_count": c['hardware_count'],
                    "ttft_p95": c['ttft_p95'],
                    "e2e_p95": c['e2e_p95'],
                    "cost_monthly": c['hw_cost_monthly'],
                    "score": c['final_score'],
                    "meets_slo": c['meets_slo'],
                }
                for c in scored_combos[1:6]  # Top 5 alternatives
            ],
        },
        "score_breakdown": {
            "accuracy": {"score": top['quality_score'], "weight": weights['accuracy']},
            "latency": {"score": top['latency_score'], "weight": weights['latency']},
            "cost": {"score": top['cost_score'], "weight": weights['cost']},
            "throughput": {"score": top['throughput_score'], "weight": weights['throughput']},
        },
        "benchmark_actual": {
            "ttft_mean": top['ttft_mean'],
            "ttft_p95": top['ttft_p95'],
            "itl_mean": top['itl_mean'],
            "itl_p95": top['itl_p95'],
            "e2e_mean": top['e2e_mean'],
            "e2e_p95": top['e2e_p95'],
            "tokens_per_second": top['tokens_per_second'],
            "prompt_tokens": top['prompt_tokens'],
            "output_tokens": top['output_tokens'],
        },
        "recommendations": [
            {
                "rank": i + 1,
                "model_name": f"{format_display_name(c['model_name'])} on {c['hardware']} x{c['hardware_count']}",
                "model_id": c['model_repo'],
                "hardware": c['hardware'],
                "hardware_count": c['hardware_count'],
                "final_score": c['final_score'],
                "score_breakdown": {
                    "quality_score": c['quality_score'],
                    "latency_score": c['latency_score'],
                    "cost_score": c['cost_score'],
                    "capacity_score": c['throughput_score'],
                    "accuracy_contribution": round(c['quality_score'] * weights['accuracy'] / 100 * c['final_score'], 1),
                    "latency_contribution": round(c['latency_score'] * weights['latency'] / 100 * c['final_score'], 1),
                    "cost_contribution": round(c['cost_score'] * weights['cost'] / 100 * c['final_score'], 1),
                    "capacity_contribution": round(c['throughput_score'] * weights['throughput'] / 100 * c['final_score'], 1),
                },
                "benchmark_slo": {
                    "slo_actual": {
                        "ttft_mean_ms": c['ttft_mean'],
                        "ttft_p95_ms": c['ttft_p95'],
                        "itl_mean_ms": c['itl_mean'],
                        "itl_p95_ms": c['itl_p95'],
                        "e2e_mean_ms": c['e2e_mean'],
                        "e2e_p95_ms": c['e2e_p95'],
                    },
                    "throughput": {
                        "tokens_per_sec": c['tokens_per_second'],
                    },
                    "token_config": {
                        "prompt": c['prompt_tokens'],
                        "output": c['output_tokens'],
                    },
                    "hardware": c['hardware'],
                    "hardware_count": c['hardware_count'],
                    "model_repo": c['model_repo'],
                    "benchmark_samples": 1,
                },
                "cost_monthly": c['hw_cost_monthly'],
                "meets_slo": c['meets_slo'],
                "pros": get_model_pros(c, priority),
                "cons": get_model_cons(c, priority),
            }
            for i, c in enumerate(scored_combos[:10])
        ],
    }


def get_selection_reason(top: dict, priority: str) -> str:
    """Generate human-readable selection reason."""
    model = format_display_name(top['model_name'])
    hw = f"{top['hardware']} x{top['hardware_count']}"
    ttft = top['ttft_p95']
    cost = top['hw_cost_monthly']
    tps = top['tokens_per_second']
    
    if priority == "cost_saving":
        return f"{model} on {hw} is the most cost-effective option (${cost:,}/mo) that meets your SLO requirements with {ttft:.0f}ms TTFT."
    elif priority == "low_latency":
        return f"{model} on {hw} delivers the lowest latency ({ttft:.0f}ms TTFT P95) from actual benchmarks."
    elif priority == "high_accuracy":
        return f"{model} has the highest accuracy score for your use case, running on {hw} with {ttft:.0f}ms TTFT."
    elif priority == "high_throughput":
        return f"{model} on {hw} achieves {tps:.0f} tokens/sec throughput from actual benchmarks."
    else:  # balanced
        return f"{model} on {hw} provides optimal balance: {ttft:.0f}ms TTFT, {tps:.0f} tokens/sec, ${cost:,}/mo."


def get_model_pros(combo: dict, priority: str) -> list:
    """Generate pros based on ACTUAL benchmark metrics."""
    pros = []
    ttft = combo['ttft_p95']
    tps = combo['tokens_per_second']
    cost = combo['hw_cost_monthly']
    quality = combo['quality_score']
    
    if ttft < 50:
        pros.append(f"Ultra-fast TTFT ({ttft:.0f}ms)")
    elif ttft < 100:
        pros.append(f"Fast TTFT ({ttft:.0f}ms)")
    
    if tps > 400:
        pros.append(f"🚀 High throughput ({tps:.0f} tok/s)")
    elif tps > 200:
        pros.append(f"Good throughput ({tps:.0f} tok/s)")
    
    if cost < 3000:
        pros.append(f"Cost-efficient (${cost:,}/mo)")
    
    if quality > 50:
        pros.append(f"High accuracy ({quality:.0f}%)")
    
    if combo['meets_slo']:
        pros.append("Meets SLO targets")
    
    return pros[:4] if pros else ["Benchmarked"]


def get_model_cons(combo: dict, priority: str) -> list:
    """Generate cons based on ACTUAL benchmark metrics."""
    cons = []
    ttft = combo['ttft_p95']
    tps = combo['tokens_per_second']
    cost = combo['hw_cost_monthly']
    quality = combo['quality_score']
    
    if ttft > 200:
        cons.append(f"⏱️ Higher latency ({ttft:.0f}ms)")
    
    if tps < 100:
        cons.append(f"📉 Lower throughput ({tps:.0f} tok/s)")
    
    if cost > 10000:
        cons.append(f"💸 Premium cost (${cost:,}/mo)")
    
    if quality < 40:
        cons.append(f"Lower accuracy score ({quality:.0f}%)")
    
    if not combo['meets_slo']:
        cons.append("May not meet SLO")
    
    return cons[:2]


def mock_recommendation_fallback(context: dict) -> dict:
    """Fallback recommendation when benchmark data unavailable."""
    return mock_recommendation(context)


def mock_recommendation(context: dict) -> dict:
    """FALLBACK: Recommendation using CSV data when benchmarks unavailable.
    
    Data sources:
    - Accuracy: weighted_scores/{use_case}.csv (task-specific benchmark scores)
    - Cost: model_pricing.csv (price_blended - $/1M tokens)
    - Latency: model_pricing.csv (median_output_tokens_per_sec, median_ttft_seconds)
    
    Enterprise-grade error handling with graceful fallbacks.
    """
    # Validate context input
    if not context or not isinstance(context, dict):
        context = {}
    
    use_case = context.get("use_case", "chatbot_conversational")
    priority = context.get("priority", "balanced")
    
    # Validate use_case is in allowed list
    valid_use_cases = [
        "chatbot_conversational", "code_completion", "code_generation_detailed",
        "document_analysis_rag", "summarization_short", "long_document_summarization",
        "translation", "content_generation", "research_legal_analysis"
    ]
    if use_case not in valid_use_cases:
        use_case = "chatbot_conversational"
    
    # Validate priority is in allowed list
    valid_priorities = ["balanced", "low_latency", "cost_saving", "high_accuracy", "high_throughput"]
    if priority not in valid_priorities:
        priority = "balanced"
    
    # Load use-case-specific weighted scores (the QUALITY component)
    weighted_df = load_weighted_scores(use_case)
    # Also load 206-model benchmark for validation
    all_models_df = load_206_models()
    # Load REAL pricing and latency data
    pricing_df = load_model_pricing()
    
    # Create pricing lookup dict (model_name -> pricing data)
    pricing_lookup = {}
    if not pricing_df.empty:
        for _, row in pricing_df.iterrows():
            model_name = row.get('model_name', '')
            if model_name:
                pricing_lookup[model_name] = {
                    'price_blended': row.get('price_blended', 0),
                    'price_input': row.get('price_per_1m_input_tokens', 0),
                    'price_output': row.get('price_per_1m_output_tokens', 0),
                    'tokens_per_sec': row.get('median_output_tokens_per_sec', 0),
                    'ttft_seconds': row.get('median_ttft_seconds', 0),
                }
    
    # Calculate normalization ranges for scoring
    # Cost: Lower is better (0 = most expensive, 100 = cheapest/free)
    max_price = max((p['price_blended'] for p in pricing_lookup.values() if p['price_blended'] > 0), default=10)
    # Latency: Higher tokens/sec is better (faster = higher score)
    max_tokens_sec = max((p['tokens_per_sec'] for p in pricing_lookup.values() if p['tokens_per_sec'] > 0), default=500)
    
    # Priority-based weights for MCDM scoring
    weights = {
        "balanced": {"accuracy": 0.30, "latency": 0.25, "cost": 0.25, "capacity": 0.20},
        "low_latency": {"accuracy": 0.20, "latency": 0.45, "cost": 0.15, "capacity": 0.20},
        "cost_saving": {"accuracy": 0.20, "latency": 0.15, "cost": 0.50, "capacity": 0.15},
        "high_accuracy": {"accuracy": 0.50, "latency": 0.20, "cost": 0.15, "capacity": 0.15},
        "high_throughput": {"accuracy": 0.20, "latency": 0.15, "cost": 0.15, "capacity": 0.50},
    }[priority]
    
    # Parse use case score from weighted_scores CSV
    def parse_score(x):
        if pd.isna(x) or x == 'N/A':
            return 0
        if isinstance(x, str):
            return float(x.replace('%', '')) if '%' in x else float(x) * 100
        return float(x) * 100 if x <= 1 else float(x)
    
    def calculate_cost_score(model_name: str) -> float:
        """Calculate cost score from REAL pricing data.
        Lower price = higher score (100 = free, 0 = most expensive)
        """
        pricing = pricing_lookup.get(model_name, {})
        price = pricing.get('price_blended', 0)
        
        if price <= 0:
            # Free model = perfect cost score
            return 95
        
        # Normalize: lower price = higher score
        # Score = 100 - (price / max_price * 100)
        cost_score = 100 - (price / max_price * 80)  # Cap at 80% reduction
        return max(min(cost_score, 100), 20)  # Range: 20-100
    
    def calculate_latency_score(model_name: str) -> float:
        """Calculate latency score from REAL speed data.
        Higher tokens/sec = higher score
        """
        import math
        pricing = pricing_lookup.get(model_name, {})
        tokens_sec = pricing.get('tokens_per_sec', 0) or 0
        ttft = pricing.get('ttft_seconds', 0) or 0
        
        # Handle NaN values
        if isinstance(tokens_sec, float) and math.isnan(tokens_sec):
            tokens_sec = 0
        if isinstance(ttft, float) and math.isnan(ttft):
            ttft = 0
        
        if tokens_sec <= 0:
            # No data - estimate based on model size
            model_lower = model_name.lower()
            is_large = any(s in model_lower for s in ["405b", "70b", "72b", "235b", "120b", "80b"])
            is_medium = any(s in model_lower for s in ["32b", "27b", "22b", "14b", "49b", "20b"])
            is_small = any(s in model_lower for s in ["7b", "8b", "3b", "4b", "1.7b", "1b"])
            if is_large:
                return 45.0
            elif is_medium:
                return 60.0
            elif is_small:
                return 80.0
            return 55.0  # Default for unknown size
        
        # Normalize: higher tokens/sec = higher score
        safe_max = max_tokens_sec if max_tokens_sec > 0 else 500
        latency_score = (tokens_sec / safe_max) * 100
        
        # Bonus for low TTFT (< 0.5s = +10, < 1s = +5)
        if 0 < ttft < 0.5:
            latency_score += 10
        elif 0 < ttft < 1.0:
            latency_score += 5
        
        return float(max(min(latency_score, 100), 20))
    
    models = []
    
    # Use weighted_scores CSV for quality (already ranked by use case)
    if not weighted_df.empty:
        # Get valid model names from the 204-model benchmark
        valid_models = set(all_models_df['Model Name'].dropna().tolist()) if not all_models_df.empty else set()
        
        # Filter weighted_scores to only include models in the 204 benchmark
        weighted_df = weighted_df[weighted_df['Model Name'].isin(valid_models)] if valid_models else weighted_df
        
        # Get top 10 models from weighted scores (already sorted by use case quality)
        top_models = weighted_df.head(10)
        
        for _, row in top_models.iterrows():
            model_name = row.get("Model Name", "")
            provider = row.get("Provider", "Unknown")
            
            # Get QUALITY from USE-CASE-SPECIFIC weighted score!
            quality_score = parse_score(row.get('Use Case Score', 0))
            
            # Skip models with no quality score
            if quality_score == 0:
                continue
            
            # Get REAL cost and latency scores from model_pricing.csv
            cost_score = calculate_cost_score(model_name)
            latency_score = calculate_latency_score(model_name)
            
            # Capacity score based on throughput and model architecture
            model_lower = model_name.lower()
            is_moe = ("a" in model_lower and "b" in model_lower) or "moe" in model_lower or "mixture" in model_lower
            is_small = any(s in model_lower for s in ["7b", "8b", "3b", "4b", "1.7b", "1b"])
            
            # Ensure latency_score is a valid number
            safe_latency = latency_score if latency_score and latency_score > 0 else 55.0
            
            if is_moe:
                capacity_score = 80 + (safe_latency / 10)  # MoE = high capacity
            elif is_small:
                capacity_score = 75 + (safe_latency / 8)
            else:
                capacity_score = 50 + (safe_latency / 5)
            
            # Ensure all scores are valid floats
            import math
            quality_score = float(quality_score) if quality_score and not (isinstance(quality_score, float) and math.isnan(quality_score)) else 50.0
            latency_score = float(safe_latency)
            cost_score = float(cost_score) if cost_score and not (isinstance(cost_score, float) and math.isnan(cost_score)) else 50.0
            capacity_score = float(capacity_score) if capacity_score and not (isinstance(capacity_score, float) and math.isnan(capacity_score)) else 60.0
            
            models.append({
                "name": model_name,
                "provider": provider,
                "quality": min(max(quality_score, 0), 100),
                "latency": min(max(latency_score, 0), 100),
                "cost": min(max(cost_score, 0), 100),
                "capacity": min(max(capacity_score, 0), 100),
            })
    
    # Fallback to hardcoded models if CSV fails (using exact names from CSV)
    if not models:
        models = [
            {"name": "Kimi K2 Thinking", "provider": "Moonshot AI", "quality": 94, "latency": 62, "cost": 52, "capacity": 65},
            {"name": "MiniMax-M2", "provider": "MiniMax", "quality": 92, "latency": 68, "cost": 58, "capacity": 72},
            {"name": "DeepSeek V3 (Dec '24)", "provider": "DeepSeek", "quality": 95, "latency": 55, "cost": 45, "capacity": 58},
            {"name": "Qwen2.5 Instruct 72B", "provider": "Alibaba", "quality": 90, "latency": 58, "cost": 48, "capacity": 62},
            {"name": "Llama 3.1 Instruct 70B", "provider": "Meta", "quality": 88, "latency": 60, "cost": 50, "capacity": 65},
        ]
    
    # Calculate final MCDM score (ensure no NaN values)
    import math
    for m in models:
        # Ensure all component scores are valid numbers
        quality = m["quality"] if m["quality"] and not math.isnan(m["quality"]) else 50.0
        latency = m["latency"] if m["latency"] and not math.isnan(m["latency"]) else 50.0
        cost = m["cost"] if m["cost"] and not math.isnan(m["cost"]) else 50.0
        capacity = m["capacity"] if m["capacity"] and not math.isnan(m["capacity"]) else 50.0
        
        m["final_score"] = (
            quality * weights["accuracy"] +
            latency * weights["latency"] +
            cost * weights["cost"] +
            capacity * weights["capacity"]
        )
        
        # Ensure final_score is valid
        if math.isnan(m["final_score"]):
            m["final_score"] = 50.0
    
    # Sort by score
    models.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Get hardware from context
    user_hardware = context.get("hardware", None)
    
    # Get optimal hardware recommendation based on priority and SLO requirements
    hw_recommendation = recommend_optimal_hardware(use_case, priority, user_hardware)
    
    # Use recommended hardware or user-specified
    if user_hardware:
        hardware = user_hardware
    elif hw_recommendation and hw_recommendation.get('recommended'):
        hardware = hw_recommendation['recommended']['hardware']
    else:
        hardware = "H100"
    
    # Build recommendations (top 10 to support filtering)
    recommendations = []
    for m in models[:10]:
        pros = []
        cons = []
        
        if m["quality"] >= 90:
            pros.append("Top Quality")
        elif m["quality"] >= 80:
            pros.append("Good Quality")
        if m["latency"] >= 85:
            pros.append("Ultra Fast")
        elif m["latency"] >= 75:
            pros.append("🚀 Fast")
        if m["cost"] >= 80:
            pros.append("Cost-Efficient")
        if m["capacity"] >= 85:
            pros.append("High Capacity")
        
        if m["quality"] < 75:
            cons.append("📉 Lower Quality")
        if m["latency"] < 55:
            cons.append("🐢 Slower")
        if m["cost"] < 45:
            cons.append("💸 Expensive")
        if m["capacity"] < 50:
            cons.append("Limited Capacity")
        
        # Get REAL benchmark SLO data for this model
        benchmark_slo = get_slo_for_model(m["name"], use_case, hardware)
        
        recommendation = {
            "model_name": m["name"],
            "provider": m["provider"],
            "final_score": m["final_score"],
            "score_breakdown": {
                "quality_score": m["quality"],
                "latency_score": m["latency"],
                "cost_score": m["cost"],
                "capacity_score": m["capacity"],
                "accuracy_contribution": m["quality"] * weights["accuracy"],
                "latency_contribution": m["latency"] * weights["latency"],
                "cost_contribution": m["cost"] * weights["cost"],
                "capacity_contribution": m["capacity"] * weights["capacity"],
            },
            "pros": pros if pros else ["Balanced Performance"],
            "cons": cons if cons else ["No significant weaknesses"],
        }
        
        # Add benchmark SLO data if available
        if benchmark_slo:
            recommendation["benchmark_slo"] = benchmark_slo
        
        recommendations.append(recommendation)
    
    # Build response with hardware recommendation
    response = {"recommendations": recommendations}
    
    # Add hardware recommendation details
    if hw_recommendation:
        response["hardware_recommendation"] = hw_recommendation
        response["slo_targets"] = hw_recommendation.get("slo_targets")
    
    return response

# =============================================================================
# VISUAL COMPONENTS
# =============================================================================

def render_hero():
    """Render compact hero section."""
    st.markdown("""
    <div class="hero-container">
        <img src="app/static/neuralnav-logo.png" width="48" height="48" style="margin-right: 12px; vertical-align: middle;" alt="NeuralNav Logo">
        <span class="hero-title">NeuralNav</span>
        <div class="hero-subtitle">AI-Powered LLM Deployment Recommendations — From Natural Language to Production in Seconds</div>
    </div>
    """, unsafe_allow_html=True)


def render_about_section(models_df: pd.DataFrame):
    """Render About section at the bottom with expandable info."""
    st.markdown("""
    <div style="margin-top: 2rem; padding: 1.5rem; background: #000000; 
                border-radius: 1rem; border: 1px solid rgba(255,255,255,0.2);">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <span style="color: #EE0000; font-weight: 700; font-size: 1.2rem;">About</span>
        </div>
        <div style="display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 0.5rem;">
            <span style="color: rgba(255,255,255,0.8); font-size: 0.9rem;"><strong style="color: white;">2,662</strong> Model-Hardware Configs</span>
            <span style="color: rgba(255,255,255,0.8); font-size: 0.9rem;"><strong style="color: white;">40</strong> Models with Performance</span>
            <span style="color: rgba(255,255,255,0.8); font-size: 0.9rem;"><strong style="color: white;">50</strong> Models with Accuracy</span>
            <span style="color: rgba(255,255,255,0.8); font-size: 0.9rem;"><strong style="color: white;">6</strong> GPU Types</span>
            <span style="color: rgba(255,255,255,0.8); font-size: 0.9rem;"><strong style="color: white;">9</strong> Use Cases</span>
        </div>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin: 0;">
            Powered by <strong style="color: white;">Qwen 2.5 7B</strong> for context extraction, <strong style="color: white;">Performance Benchmarks,</strong> and <strong style="color: white;">Artificial Analysis</strong> accuracy scores.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # MCDM Expander styling
    st.markdown("""
    <style>
        [data-testid="stExpander"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: 12px !important;
        }
        [data-testid="stExpander"] summary {
            background: rgba(88, 166, 255, 0.08) !important;
            border-radius: 11px 11px 0 0 !important;
            padding: 1rem 1.25rem !important;
        }
        [data-testid="stExpander"] summary span {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        [data-testid="stExpander"] svg {
            color: var(--accent-blue) !important;
        }
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] th,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] td,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
            color: var(--text-primary) !important;
        }
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] code {
            background: rgba(88, 166, 255, 0.1) !important;
            color: var(--accent-blue) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Three expanders for extra info
    with st.expander("**Scoring Methodology** - How each category is calculated", expanded=False):
        st.markdown('<h4 style="color: #EE0000 !important; margin-bottom: 1.25rem; font-family: Inter, sans-serif;">4 Recommendation Categories</h4>', unsafe_allow_html=True)
        st.markdown("""
<table style="width: 100%; border-collapse: collapse; margin-top: 1rem; background: transparent;">
<tr style="border-bottom: 2px solid rgba(255,255,255,0.25);">
    <th style="text-align: left; padding: 0.75rem; color: #EE0000 !important; font-weight: 700; width: 140px;">Category</th>
    <th style="text-align: left; padding: 0.75rem; color: #EE0000 !important; font-weight: 700;">Scoring Logic</th>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
    <td style="padding: 0.75rem; color: white !important; font-weight: 600;">Best Accuracy</td>
    <td style="padding: 0.75rem; color: white !important;"><strong>RAW</strong> accuracy score from Artificial Analysis benchmarks (MMLU-Pro, GPQA, etc.) - Top 5 unique models</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
    <td style="padding: 0.75rem; color: white !important; font-weight: 600;">Best Latency</td>
    <td style="padding: 0.75rem; color: white !important;">From the <strong>top 5 accuracy models</strong>, finds the hardware config with <strong>lowest TTFT</strong> (fastest response)</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
    <td style="padding: 0.75rem; color: white !important; font-weight: 600;">Best Cost</td>
    <td style="padding: 0.75rem; color: white !important;">From the <strong>top 5 accuracy models</strong>, finds the hardware config with <strong>cheapest GPU setup</strong></td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.2); background: rgba(238,0,0,0.1);">
    <td style="padding: 0.75rem; color: #EE0000 !important; font-weight: 600;">Balanced</td>
    <td style="padding: 0.75rem; color: white !important;"><strong>TASK-OPTIMIZED</strong>: 70% (Accuracy + Task Bonus) + 30% avg(Latency, Cost) - diversifies recommendations per use case</td>
</tr>
</table>
        """, unsafe_allow_html=True)
        
        st.markdown('<h4 style="color: #EE0000 !important; margin-top: 1.5rem; margin-bottom: 1rem; font-family: Inter, sans-serif;">Balanced Score (Task-Optimized Formula)</h4>', unsafe_allow_html=True)
        st.code("BALANCED = (Accuracy + Task_Bonus) × 70% + avg(Latency, Cost) × 30%", language=None)
        st.markdown("""
<p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 1rem;">
    <strong style="color: #EE0000;">Task Bonuses:</strong> Different model families get bonuses based on their strengths for specific use cases.
    For example, <strong>DeepSeek</strong> gets +20 for code tasks, <strong>Qwen</strong> gets +20 for translation, <strong>MiniMax</strong> gets +18 for long documents.
</p>
<p style="color: rgba(255,255,255,0.8); margin-top: 0.5rem; font-size: 0.9rem;">
    This ensures <strong style="color: #EE0000;">diverse, task-appropriate recommendations</strong> - the best code model for code, best multilingual for translation, etc.
    The other 3 categories (Best Accuracy, Best Latency, Best Cost) still use RAW scores without task bonuses.
</p>
<p style="color: rgba(255,255,255,0.6); margin-top: 0.5rem; font-size: 0.85rem;">
    <em>Example for Code task: GPT-OSS (Acc=67, Bonus=+12) → (67+12)×0.7 + 55×0.3 = <strong>71.8</strong><br>
    DeepSeek (Acc=62, Bonus=+20) → (62+20)×0.7 + 60×0.3 = <strong>75.4</strong> → DeepSeek wins for code!</em>
</p>
        """, unsafe_allow_html=True)
    
    with st.expander("Model Catalog - Browse 204 open-source models", expanded=False):
        render_catalog_content(models_df)
    
    with st.expander("How It Works - End-to-end pipeline documentation", expanded=False):
        render_how_it_works_content()


def render_catalog_content(models_df: pd.DataFrame):
    """Model catalog content for About section expander."""
    st.markdown("""
    <p style="color: rgba(255,255,255,0.9); margin: 0 0 1rem 0; font-size: 0.95rem;">
        Complete benchmark data from <strong style="color: #D4AF37;">Performance DB</strong> + 
        <strong style="color: #38ef7d;">Artificial Analysis</strong> covering 
        <span style="color: #38ef7d; font-weight: 700;">50 benchmarked models</span> with 
        <span style="color: #667eea; font-weight: 700;">50 models having accuracy scores</span> across 
        <span style="color: #a371f7; font-weight: 700;">15 benchmark datasets</span>.
    </p>
    """, unsafe_allow_html=True)
    
    if models_df is not None and not models_df.empty:
        # Search
        search = st.text_input("Search models", placeholder="e.g., Llama, Qwen, DeepSeek...", key="about_catalog_search")
        
        filtered_df = models_df.copy()
        if search:
            filtered_df = filtered_df[filtered_df.apply(lambda row: search.lower() in str(row).lower(), axis=1)]
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(models_df)} models**")
        st.dataframe(filtered_df.head(20), use_container_width=True, height=400)
    else:
        st.info("Model catalog data not available.")


def render_how_it_works_content():
    """How It Works content for About section expander."""
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h4 style="color: #EE0000; margin-bottom: 1rem;">End-to-End Pipeline</h4>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px; padding: 1rem; background: #000000; border-radius: 0.75rem; border-left: 3px solid #EE0000;">
                <div style="font-weight: 700; color: #EE0000; margin-bottom: 0.5rem;">1. Context Extraction</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">Qwen 2.5 7B extracts use case, users, priority & hardware from natural language</div>
            </div>
            <div style="flex: 1; min-width: 200px; padding: 1rem; background: #000000; border-radius: 0.75rem; border-left: 3px solid white;">
                <div style="font-weight: 700; color: white; margin-bottom: 0.5rem;">2. MCDM Scoring</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">Score 204 models on Accuracy, Latency, Cost & Capacity with weighted criteria</div>
            </div>
            <div style="flex: 1; min-width: 200px; padding: 1rem; background: #000000; border-radius: 0.75rem; border-left: 3px solid #EE0000;">
                <div style="font-weight: 700; color: #EE0000; margin-bottom: 0.5rem;">3. Recommendation</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">Best models with explainability, SLO compliance & deployment config</div>
            </div>
        </div>
    </div>
    
    <h4 style="color: #EE0000; margin: 1.5rem 0 1rem 0;">Supported Use Cases</h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem;">
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Chat Completion</div>
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Code Completion</div>
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Document Q&A (RAG)</div>
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Summarization</div>
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Legal Analysis</div>
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Translation</div>
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Content Generation</div>
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Long Doc Summary</div>
        <div style="padding: 0.75rem; background: #000000; border-radius: 0.5rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.2);">Code Generation</div>
    </div>
    
    <h4 style="color: #EE0000; margin: 1.5rem 0 1rem 0;">Data Sources</h4>
    <ul style="color: rgba(255,255,255,0.8); font-size: 0.9rem; line-height: 1.8; margin: 0; padding-left: 1.5rem;">
        <li><strong style="color: white;">Artificial Analysis</strong> - Model benchmarks, pricing, and performance data</li>
        <li><strong style="color: white;">Performance Benchmarks</strong> - Real hardware deployment SLOs (TTFT, ITL, E2E latency)</li>
        <li><strong style="color: white;">Use-Case CSVs</strong> - Pre-computed weighted scores for each use case</li>
    </ul>
        """, unsafe_allow_html=True)


def render_pipeline():
    """Render the pipeline visualization."""
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="pipeline-number pipeline-number-1">1</div>
            <div class="pipeline-title">Context Extraction</div>
            <div class="pipeline-desc">Qwen 2.5 7B extracts use case, users, priority & hardware from natural language</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-number pipeline-number-2">2</div>
            <div class="pipeline-title">MCDM Scoring</div>
            <div class="pipeline-desc">Score 204 models on Accuracy, Latency, Cost & Capacity with weighted criteria</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-number pipeline-number-3">3</div>
            <div class="pipeline-title">Recommendation</div>
            <div class="pipeline-desc">Top 5 models with explainability, tradeoffs, SLO compliance & deployment config</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_top5_table(recommendations: list, priority: str):
    """Render beautiful Top 5 recommendation leaderboard table with filtering.
    
    NOTE: The backend now implements the ACCURACY-FIRST strategy in ranking_service.py.
    The UI uses the backend's pre-ranked lists directly from st.session_state.ranked_response.
    """
    
    # Filter controls
    st.markdown("""
    <style>
        /* Filter section - ALL text white and visible */
        div[data-testid="stHorizontalBlock"] label {
            color: white !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }
        /* Selectbox text - white */
        .stSelectbox > div > div {
            background: rgba(102, 126, 234, 0.2) !important;
        }
        .stSelectbox [data-baseweb="select"] > div {
            color: white !important;
            background: rgba(102, 126, 234, 0.15) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
        }
        .stSelectbox [data-baseweb="select"] span {
            color: white !important;
        }
        /* Slider labels white */
        .stSlider label {
            color: white !important;
            font-weight: 600 !important;
        }
        .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {
            color: rgba(255,255,255,0.6) !important;
        }
        /* Red-themed SLO Sliders - Clean minimal design */
        .stSlider > div > div > div {
            background: rgba(238, 0, 0, 0.25) !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        .stSlider > div > div > div > div {
            background: #EE0000 !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        /* Small fully red circle thumb */
        .stSlider > div > div > div > div > div {
            background: #EE0000 !important;
            border: none !important;
            width: 16px !important;
            height: 16px !important;
            top: -5px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        /* HIDE the default tooltip with value - multiple selectors for compatibility */
        .stSlider [data-testid="stThumbValue"],
        .stSlider div[data-testid="stThumbValue"],
        [data-testid="stThumbValue"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }
        /* HIDE the default min/max tick labels */
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"],
        [data-testid="stTickBarMin"],
        [data-testid="stTickBarMax"],
        .stSlider > div > div:first-child,
        .stSlider > div > div:last-child {
            display: none !important;
            visibility: hidden !important;
        }
        /* Hide ALL slider text except our custom ones */
        .stSlider [data-baseweb="slider"] + div {
            display: none !important;
        }
        /* Carousel navigation buttons - small, inside card aesthetic */
        .stButton button[kind="secondary"] {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: rgba(255,255,255,0.7) !important;
            padding: 0.3rem !important;
            min-height: 2rem !important;
            font-size: 0.9rem !important;
        }
        .stButton button[kind="secondary"]:hover {
            background: rgba(238,0,0,0.2) !important;
            border-color: #EE0000 !important;
            color: white !important;
        }
    </style>
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; padding: 1rem; 
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(56, 239, 125, 0.05)); 
                border-radius: 1rem; border: 1px solid rgba(102, 126, 234, 0.2);">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;"></span>
            <span style="color: white; font-weight: 700; font-size: 1.1rem;">Best Model Recommendations</span>
        </div>
        <span style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Click ◀ ▶ to browse top 5</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Show filter summary stats
    ranked_response_stats = st.session_state.get("ranked_response", {})
    total_configs = ranked_response_stats.get("total_configs_evaluated", 0)
    passed_configs = ranked_response_stats.get("configs_after_filters", 0)
    
    # Count unique models from passed configs
    all_passed = []
    for cat in ["balanced", "best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
        all_passed.extend(ranked_response_stats.get(cat, []))
    unique_models = len(set(r.get("model_name", "") for r in all_passed if r.get("model_name")))
    
    if total_configs > 0:
        filter_pct = (passed_configs / total_configs * 100) if total_configs > 0 else 0
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1rem; padding: 0.6rem 1rem;
                    background: rgba(238,0,0,0.08); border-radius: 8px; border: 1px solid rgba(238,0,0,0.2);">
            <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                <strong style="color: #10B981;">{passed_configs:,}</strong> configs passed SLO filter 
                from <strong>{total_configs:,}</strong> total 
                <span style="color: rgba(255,255,255,0.4);">({filter_pct:.0f}% match)</span>
            </span>
            <span style="color: rgba(255,255,255,0.5);">|</span>
            <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                <strong style="color: #EE0000;">{unique_models}</strong> unique models
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Get use case for raw accuracy lookup
    use_case = st.session_state.get("detected_use_case", "chatbot_conversational")
    
    if not recommendations:
        st.info("No models available. Please check your requirements.")
        return
    
    # Helper function to get scores from recommendation
    # ALL scores come from BACKEND - no UI calculation
    def get_scores(rec):
        backend_scores = rec.get("scores", {}) or {}
        
        return {
            "accuracy": backend_scores.get("accuracy_score", 0),
            "latency": backend_scores.get("latency_score", 0),
            "cost": backend_scores.get("price_score", 0),
            "complexity": backend_scores.get("complexity_score", 0),
            "final": backend_scores.get("balanced_score", 0),
        }
    
    # ==========================================================================
    # USE BACKEND'S PRE-RANKED LISTS (ACCURACY-FIRST strategy applied in backend)
    # The backend's ranking_service.py implements:
    # 1. Get top 5 unique models by raw accuracy (quality baseline)
    # 2. Filter all configs to only those high-quality models
    # 3. Best Latency/Cost/etc. are ranked WITHIN that quality tier
    # ==========================================================================
    ranked_response = st.session_state.get("ranked_response", {})
    
    # Get pre-ranked lists from backend (already filtered to high-quality models)
    top5_balanced = ranked_response.get("balanced", [])[:5]
    top5_accuracy = ranked_response.get("best_accuracy", [])[:5]
    top5_latency = ranked_response.get("lowest_latency", [])[:5]
    top5_cost = ranked_response.get("lowest_cost", [])[:5]
    top5_simplest = ranked_response.get("simplest", [])[:5]
    
    # Fallback to local sorting if backend response is empty (shouldn't happen)
    if not top5_balanced and recommendations:
        top5_balanced = sorted(recommendations, key=lambda x: get_scores(x)["final"], reverse=True)[:5]
    if not top5_accuracy and recommendations:
        top5_accuracy = sorted(recommendations, key=lambda x: get_scores(x)["accuracy"], reverse=True)[:5]
    if not top5_latency and recommendations:
        top5_latency = sorted(recommendations, key=lambda x: get_scores(x)["latency"], reverse=True)[:5]
    if not top5_cost and recommendations:
        top5_cost = sorted(recommendations, key=lambda x: get_scores(x)["cost"], reverse=True)[:5]
    if not top5_simplest and recommendations:
        top5_simplest = sorted(recommendations, key=lambda x: get_scores(x)["complexity"], reverse=True)[:5]
    
    # Best = first in each list
    best_overall = top5_balanced[0] if top5_balanced else None
    best_accuracy = top5_accuracy[0] if top5_accuracy else None
    best_latency = top5_latency[0] if top5_latency else None
    best_cost = top5_cost[0] if top5_cost else None
    best_simplest = top5_simplest[0] if top5_simplest else None
    
    # Store top 5 for each category in session state (for explore dialogs)
    st.session_state.top5_balanced = top5_balanced
    st.session_state.top5_accuracy = top5_accuracy
    st.session_state.top5_latency = top5_latency
    st.session_state.top5_cost = top5_cost
    st.session_state.top5_simplest = top5_simplest
    
    # Helper to render a "Best" card
    def render_best_card(title, icon, color, rec, highlight_field):
        scores = get_scores(rec)
        model_name = format_display_name(rec.get('model_name', 'Unknown'))
        gpu_cfg = rec.get('gpu_config', {}) or {}
        hw_type = gpu_cfg.get('gpu_type', rec.get('hardware', 'H100'))
        hw_count = gpu_cfg.get('gpu_count', rec.get('hardware_count', 1))
        hw_display = f"{hw_count}x {hw_type}"
        
        highlight_value = scores.get(highlight_field, 0)
        final_score = scores.get("final", 0)
        
        return f'''
        <div style="background: linear-gradient(135deg, rgba(30,30,40,0.9), rgba(40,40,55,0.9)); 
                    border: 2px solid {color}40; border-radius: 16px; padding: 1.25rem; 
                    box-shadow: 0 8px 32px {color}20;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <span style="color: {color}; font-weight: 700; font-size: 1.1rem;">{title}</span>
        </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="flex: 1;">
                    <div style="color: white; font-weight: 700; font-size: 1.15rem;">{model_name}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">{hw_display}</div>
        </div>
                <div style="text-align: right;">
                    <div style="color: {color}; font-size: 2rem; font-weight: 800;">{highlight_value:.0f}</div>
                    <div style="color: rgba(255,255,255,0.4); font-size: 0.7rem;">SCORE</div>
        </div>
        </div>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <span style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Acc {scores["accuracy"]:.0f}</span>
                <span style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Lat {scores["latency"]:.0f}</span>
                <span style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Cost {scores["cost"]:.0f}</span>
                <span style="color: #ffffff; font-size: 0.8rem; font-weight: 700;">Final: {final_score:.1f}</span>
    </div>
                        </div>
        '''
    
    # Helper to render a CAROUSEL card with arrows INSIDE the card design
    def render_carousel_card(title, color, recs_list, highlight_field, category_key, col):
        """Render a card with ← → arrows embedded inside the card."""
        if not recs_list:
            return
        
        # Initialize carousel index
        idx_key = f"carousel_{category_key}"
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0
        
        total = min(5, len(recs_list))
        current_idx = st.session_state[idx_key] % total
        rec = recs_list[current_idx]
        
        scores = get_scores(rec)
        # Normalize model name to Provider/Model format for display
        raw_model_name = rec.get('model_name', 'Unknown')
        model_name = format_display_name(raw_model_name)
        gpu_cfg = rec.get('gpu_config', {}) or {}
        hw_type = gpu_cfg.get('gpu_type', rec.get('hardware', 'H100'))
        hw_count = gpu_cfg.get('gpu_count', rec.get('hardware_count', 1))
        replicas = gpu_cfg.get('replicas', 1)
        benchmark_metrics = rec.get('benchmark_metrics', {}) or {}
        rps_per_replica = benchmark_metrics.get('requests_per_second', 0)
        
        # Get the primary metric selected by user (TTFT, ITL, or E2E)
        primary_metric = st.session_state.get('slo_primary_metric', 'TTFT')
        # Get selected percentile (mean, p90, p95, p99)
        selected_percentile = st.session_state.get('slo_percentile', 'p95')
        
        # Map percentile to display label
        percentile_labels = {'mean': 'Mean', 'p90': 'P90', 'p95': 'P95', 'p99': 'P99'}
        percentile_display = percentile_labels.get(selected_percentile, 'P95')
        
        # Get metric values based on selected primary metric AND percentile
        if primary_metric == "ITL":
            metric_key = f'itl_{selected_percentile}'
            metric_value = benchmark_metrics.get(metric_key, benchmark_metrics.get('itl_mean', 0))
            metric_label = f"ITL ({percentile_display})"
            metric_unit = "ms"
        elif primary_metric == "E2E":
            metric_key = f'e2e_{selected_percentile}'
            metric_value = benchmark_metrics.get(metric_key, benchmark_metrics.get('e2e_mean', 0))
            metric_label = f"E2E ({percentile_display})"
            metric_unit = "ms"
        else:  # Default to TTFT
            metric_key = f'ttft_{selected_percentile}'
            metric_value = benchmark_metrics.get(metric_key, benchmark_metrics.get('ttft_mean', 0))
            metric_label = f"TTFT ({percentile_display})"
            metric_unit = "ms"
        
        # Get throughput with percentile label too
        tps_key = f'tps_{selected_percentile}'
        throughput_tps = benchmark_metrics.get(tps_key, benchmark_metrics.get('tps_mean', 0))
        # Check if data is estimated or real (validated)
        is_estimated = benchmark_metrics.get('estimated', False)
        hw_display = f"{hw_count}x{hw_type}"
        highlight_value = scores.get(highlight_field, 0)
        final_score = scores.get("final", 0)
        
        # Dots indicator
        dots_html = ""
        for i in range(total):
            dot_color = "#EE0000" if i == current_idx else "rgba(255,255,255,0.3)"
            dots_html += f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{dot_color};margin:0 2px;"></span>'
        
        with col:
            # Card with arrows INSIDE using CSS absolute positioning
            card_html = f'''<div style="background: linear-gradient(135deg, rgba(30,30,40,0.95), rgba(40,40,55,0.95)); border: 2px solid {color}40; border-radius: 16px; padding: 1.75rem; box-shadow: 0 8px 32px {color}20; min-height: 380px; position: relative;">
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
<span style="color: {color}; font-weight: 700; font-size: 1.3rem;">{title}</span>
<span style="color: rgba(255,255,255,0.5); font-size: 1rem;">{current_idx + 1}/{total}</span>
                        </div>
<div style="color: white; font-weight: 700; font-size: 1.6rem; margin-bottom: 1.25rem; text-transform: uppercase;">{model_name}</div>
<div style="display: flex; gap: 0.6rem; margin-bottom: 1.25rem;">
<div style="flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 0.6rem 0.7rem;">
<div style="color: rgba(255,255,255,0.5); font-size: 0.7rem; text-transform: uppercase;">Hardware</div>
<div style="color: white; font-weight: 700; font-size: 1.1rem;">{hw_display}</div>
                    </div>
<div style="flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 0.6rem 0.7rem;">
<div style="color: rgba(255,255,255,0.5); font-size: 0.7rem; text-transform: uppercase;">Replicas</div>
<div style="color: white; font-weight: 700; font-size: 1.1rem;">{replicas}</div>
                        </div>
<div style="flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 0.6rem 0.7rem;">
<div style="color: rgba(255,255,255,0.5); font-size: 0.65rem; text-transform: uppercase;">RPS/Rep</div>
<div style="color: white; font-weight: 700; font-size: 1.1rem;">{rps_per_replica:.1f}</div>
                    </div>
                        </div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
<div style="display: flex; gap: 1.5rem;">
<div style="display: flex; align-items: center; gap: 0.4rem;">
<span style="color: {color}; font-size: 1rem;">⏱</span>
<span style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">{metric_label}</span>
<span style="color: white; font-weight: 700; font-size: 1rem;">{metric_value:.0f}{metric_unit}</span>
                    </div>
<div style="display: flex; align-items: center; gap: 0.4rem;">
<span style="color: {color}; font-size: 1rem;">⚡</span>
<span style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">TPS ({percentile_display})</span>
<span style="color: white; font-weight: 700; font-size: 1rem;">{throughput_tps:.0f} tok/s</span>
                        </div>
                    </div>
<div style="text-align: right;">
<div style="color: {color}; font-size: 2.5rem; font-weight: 800; line-height: 1;">{highlight_value:.0f}</div>
<div style="color: rgba(255,255,255,0.4); font-size: 0.7rem; text-transform: uppercase;">Score</div>
                        </div>
                    </div>
<div style="display: flex; justify-content: space-between; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.95rem;">
<span style="color: rgba(255,255,255,0.6);">Acc {scores["accuracy"]:.0f}</span>
<span style="color: rgba(255,255,255,0.6);">Lat {scores["latency"]:.0f}</span>
<span style="color: rgba(255,255,255,0.6);">Cost {scores["cost"]:.0f}</span>
<span style="color: #ffffff; font-weight: 700;">Final: {final_score:.1f}</span>
                    </div>
<div style="display: flex; align-items: center; justify-content: space-between; margin-top: 0.75rem;">
<div style="display: flex; align-items: center; gap: 0.3rem;">
{'<span style="display: inline-flex; align-items: center; gap: 0.3rem; background: #10B981; color: white; font-size: 0.7rem; font-weight: 600; padding: 0.25rem 0.5rem; border-radius: 4px;"><span style="font-size: 0.65rem;">✓</span> Validated</span>' if not is_estimated else '<span style="display: inline-flex; align-items: center; gap: 0.3rem; background: rgba(255,165,0,0.8); color: white; font-size: 0.7rem; font-weight: 600; padding: 0.25rem 0.5rem; border-radius: 4px;"><span style="font-size: 0.65rem;">◐</span> Estimated</span>'}
                    </div>
<div style="display: flex; align-items: center; gap: 0.5rem;">
<span style="color: rgba(255,255,255,0.3); font-size: 0.7rem;">◀</span>
{dots_html}
<span style="color: rgba(255,255,255,0.3); font-size: 0.7rem;">▶</span>
</div>
</div>
</div>'''
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Small arrow buttons inside card footer area
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col1:
                if st.button("◀", key=f"prev_{category_key}", use_container_width=True):
                    # IMPORTANT: Reset dialog states to prevent table from opening
                    st.session_state.show_full_table_dialog = False
                    st.session_state.show_category_dialog = False
                    st.session_state.show_winner_dialog = False
                    # Use session state directly to get latest value
                    st.session_state[idx_key] = (st.session_state[idx_key] - 1) % total
            with btn_col3:
                if st.button("▶", key=f"next_{category_key}", use_container_width=True):
                    # IMPORTANT: Reset dialog states to prevent table from opening
                    st.session_state.show_full_table_dialog = False
                    st.session_state.show_category_dialog = False
                    st.session_state.show_winner_dialog = False
                    # Use session state directly to get latest value
                    st.session_state[idx_key] = (st.session_state[idx_key] + 1) % total
    
    # Render 4 carousel cards: 2 on top row, 2 on bottom row
    col1, col2 = st.columns(2)
    render_carousel_card("Balanced", "#EE0000", top5_balanced, "final", "balanced", col1)
    render_carousel_card("Best Accuracy", "#EE0000", top5_accuracy, "accuracy", "accuracy", col2)
    
    col3, col4 = st.columns(2)
    render_carousel_card("Best Latency", "#EE0000", top5_latency, "latency", "latency", col3)
    render_carousel_card("Best Cost", "#EE0000", top5_cost, "cost", "cost", col4)
    
    # Show info if limited models available
    total_available = len(recommendations)
    if total_available <= 2:
        use_case_display = use_case.replace('_', ' ').title() if use_case else "this task"
        st.markdown(f'''
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 0.75rem; margin-top: 1rem;">
            <span style="color: #a5b4fc; font-size: 0.85rem;">
                Only {total_available} model(s) have benchmarks for {use_case_display}
            </span>
    </div>
        ''', unsafe_allow_html=True)
    


def render_score_bar(label: str, icon: str, score: float, bar_class: str, contribution: float):
    """Render score bar like Artificial Analysis - number ABOVE the bar."""
    import math
    # Handle NaN values
    if math.isnan(score) if isinstance(score, float) else False:
        score = 50  # Default to 50 if NaN
    if math.isnan(contribution) if isinstance(contribution, float) else False:
        contribution = 0
    
    score = max(0, min(100, score))  # Clamp to 0-100
    
    # Color based on bar class
    bar_colors = {
        "score-bar-quality": "#38ef7d",
        "score-bar-latency": "#667eea", 
        "score-bar-cost": "#f5576c",
        "score-bar-capacity": "#8b5cf6",
    }
    bar_color = bar_colors.get(bar_class, "#38ef7d")
    
    st.markdown(f"""
    <div style="margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: rgba(255,255,255,0.9); font-weight: 600; font-size: 1rem;">{icon} {label}</span>
            <span style="font-size: 1.1rem;">
                <strong style="color: white; font-size: 1.3rem;">{score:.0f}</strong>
                <span style="color: rgba(255,255,255,0.5);">→</span>
                <span style="color: #38ef7d; font-weight: 700;">+{contribution:.1f}</span>
            </span>
        </div>
        <div style="position: relative; height: 28px;">
            <!-- Score number above bar (Artificial Analysis style) -->
            <div style="position: absolute; left: {min(score, 95)}%; transform: translateX(-50%); top: -2px; z-index: 10;">
                <span style="background: {bar_color}; color: white; font-weight: 700; font-size: 0.9rem; padding: 0.2rem 0.5rem; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{score:.0f}%</span>
            </div>
            <!-- Bar background -->
            <div style="position: absolute; top: 8px; left: 0; right: 0; height: 12px; background: rgba(255,255,255,0.1); border-radius: 6px; overflow: hidden;">
                <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {bar_color}cc, {bar_color}); border-radius: 6px; transition: width 0.5s ease;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_slo_cards(use_case: str, user_count: int, priority: str = "balanced", hardware: str = None):
    """Render SLO and workload impact cards with editable fields.
    
    SLO defaults are set to MAX (showing all configs by default).
    User drags slider LEFT to tighten SLOs and filter down configs.
    Priority affects the MAX - low_latency has tighter max values.
    """
    # Get research data for token config and priority factors
    research_data = load_research_slo_ranges()
    use_case_data = research_data.get('slo_ranges', {}).get(use_case, {}) if research_data else {}
    token_config = use_case_data.get('token_config', {'prompt': 512, 'output': 256})
    
    # Get actual benchmark ranges for this use case's token config
    benchmark_ranges = get_benchmark_ranges_for_token_config(token_config['prompt'], token_config['output'])
    
    # Get priority adjustment factors
    priority_adjustments = research_data.get('priority_adjustments', {}) if research_data else {}
    priority_factor = priority_adjustments.get(priority, {})
    ttft_factor = priority_factor.get('ttft_factor', 1.0)
    itl_factor = priority_factor.get('itl_factor', 1.0)
    e2e_factor = priority_factor.get('e2e_factor', 1.0)
    
    # Get selected percentile (default P95)
    percentile_key = st.session_state.get('slo_percentile', 'p95')
    
    # Calculate MAX values (adjusted by priority) - these are the DEFAULTS
    ttft_max_raw = benchmark_ranges.get(f'ttft_{percentile_key}_max', 270000)
    itl_max_raw = benchmark_ranges.get(f'itl_{percentile_key}_max', 430)
    e2e_max_raw = benchmark_ranges.get(f'e2e_{percentile_key}_max', 300000)
    
    ttft_default = int(ttft_max_raw * ttft_factor)
    itl_default = int(itl_max_raw * itl_factor)
    e2e_default = int(e2e_max_raw * e2e_factor)

    # Fetch expected RPS from backend using research-based workload patterns
    rps_data = fetch_expected_rps(use_case, user_count)
    if rps_data:
        estimated_qps = int(rps_data.get("expected_rps", 1))
    else:
        # Fallback to simple heuristic if API fails
        estimated_qps = max(1, user_count // 50)

    # Track if use_case or user_count changed - if so, reset custom_qps to use new default
    last_use_case = st.session_state.get("_last_rps_use_case")
    last_user_count = st.session_state.get("_last_rps_user_count")
    if last_use_case != use_case or last_user_count != user_count:
        # Use case or user count changed - reset to new calculated default
        st.session_state.custom_qps = None
        st.session_state._last_rps_use_case = use_case
        st.session_state._last_rps_user_count = user_count
        # Clear the widget key to force re-render with new value
        if "edit_qps" in st.session_state:
            del st.session_state["edit_qps"]

    # Use custom values if set, otherwise use MAX as default (shows all configs)
    # Ensure all values are integers for slider compatibility
    ttft = int(st.session_state.custom_ttft) if st.session_state.custom_ttft else ttft_default
    itl = int(st.session_state.custom_itl) if st.session_state.custom_itl else itl_default
    e2e = int(st.session_state.custom_e2e) if st.session_state.custom_e2e else e2e_default
    qps = int(st.session_state.custom_qps) if st.session_state.custom_qps else estimated_qps
    
    # Section header - Technical Specification
    st.markdown("""
    <div class="section-header" style="background: #000000; border: 1px solid rgba(255,255,255,0.2);">
        Set Technical Specification
    </div>
    """, unsafe_allow_html=True)
    
    # benchmark_ranges already fetched above for defaults
    
    # Benchmark ranges table removed - user found it confusing
    
    # Create 4 columns for all cards in one row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="slo-card">
            <div class="slo-header">
                <span class="slo-title">SLO Targets</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # CSS for SLO inputs - white borders, white text, red arrows
        st.markdown("""
        <style>
            /* Percentile selector label - WHITE */
            .stSelectbox label {
                color: white !important;
                font-weight: 600 !important;
            }
            .stSelectbox p {
                color: white !important;
            }
            /* Percentile selector box - WHITE border */
            .stSelectbox > div > div {
                background: rgba(0,0,0,0.5) !important;
                border: 2px solid rgba(255,255,255,0.6) !important;
                border-radius: 8px !important;
            }
            .stSelectbox > div > div:hover {
                border-color: #EE0000 !important;
            }
            .stSelectbox [data-baseweb="select"] > div {
                background: rgba(0,0,0,0.5) !important;
                border: 2px solid rgba(255,255,255,0.6) !important;
                color: white !important;
            }
            .stSelectbox [data-baseweb="select"] span {
                color: white !important;
            }
            /* Dropdown arrow - RED */
            .stSelectbox svg {
                fill: #EE0000 !important;
                color: #EE0000 !important;
            }
            .stSelectbox [data-baseweb="select"] svg {
                fill: #EE0000 !important;
            }
            /* Number inputs - WHITE border */
            .stNumberInput > div > div > input {
                background: rgba(0,0,0,0.5) !important;
                border: 2px solid rgba(255,255,255,0.6) !important;
                border-radius: 8px !important;
                color: white !important;
            }
            .stNumberInput > div > div > input:focus {
                border-color: #EE0000 !important;
            }
            /* Number input +/- buttons - RED */
            .stNumberInput button {
                color: #EE0000 !important;
            }
            .stNumberInput button svg {
                fill: #EE0000 !important;
            }
            /* Dropdown menu styling */
            [data-baseweb="menu"] {
                background: #1a1a1a !important;
            }
            [data-baseweb="menu"] li {
                color: white !important;
            }
            [data-baseweb="menu"] li:hover {
                background: rgba(238, 0, 0, 0.3) !important;
            }
        </style>
        """, unsafe_allow_html=True)

        # Load research data and get use-case specific ranges
        research_data = load_research_slo_ranges()
        use_case_ranges = research_data.get('slo_ranges', {}).get(use_case, {}) if research_data else {}
        token_config = use_case_ranges.get('token_config', {'prompt': 512, 'output': 256})
        prompt_tokens = token_config.get('prompt', 512)
        output_tokens = token_config.get('output', 256)
        benchmark_ranges = get_benchmark_ranges_for_token_config(prompt_tokens, output_tokens)

        # Percentile options
        percentile_options = ["P50", "P90", "P95", "P99"]
        percentile_map = {"P50": "p50", "P90": "p90", "P95": "p95", "P99": "p99"}
        reverse_map = {"p50": "P50", "p90": "P90", "p95": "P95", "p99": "P99"}

        # Initialize percentile states for each metric
        if 'ttft_percentile' not in st.session_state:
            st.session_state.ttft_percentile = "p95"
        if 'itl_percentile' not in st.session_state:
            st.session_state.itl_percentile = "p95"
        if 'e2e_percentile' not in st.session_state:
            st.session_state.e2e_percentile = "p95"

        # Track previous SLO values to detect changes
        prev_ttft = st.session_state.get("_last_ttft")
        prev_itl = st.session_state.get("_last_itl")
        prev_e2e = st.session_state.get("_last_e2e")

        # Fetch use-case specific SLO defaults from backend API
        slo_defaults = fetch_slo_defaults(use_case)
        default_ttft = slo_defaults["ttft_ms"]["default"] if slo_defaults else 500
        default_itl = slo_defaults["itl_ms"]["default"] if slo_defaults else 50
        default_e2e = slo_defaults["e2e_ms"]["default"] if slo_defaults else 10000

        # Helper function to get range for a metric and percentile
        def get_metric_range(metric: str, percentile_key: str) -> tuple:
            if metric == "ttft":
                return (
                    int(benchmark_ranges.get(f'ttft_{percentile_key}_min', 15)),
                    int(benchmark_ranges.get(f'ttft_{percentile_key}_max', 270000))
                )
            elif metric == "itl":
                return (
                    int(benchmark_ranges.get(f'itl_{percentile_key}_min', 3)),
                    int(benchmark_ranges.get(f'itl_{percentile_key}_max', 430))
                )
            else:  # e2e
                return (
                    int(benchmark_ranges.get(f'e2e_{percentile_key}_min', 800)),
                    int(benchmark_ranges.get(f'e2e_{percentile_key}_max', 300000))
                )

        # === TTFT ===
        ttft_min, ttft_max = get_metric_range("ttft", st.session_state.ttft_percentile)
        if 'input_ttft' not in st.session_state:
            st.session_state.input_ttft = default_ttft

        st.markdown('<div style="margin-top: 0.5rem; margin-bottom: 0.25rem;"><span style="color: #EE0000; font-weight: 700; font-size: 0.95rem;">TTFT (Time to First Token)</span></div>', unsafe_allow_html=True)
        ttft_val_col, ttft_pct_col = st.columns([2, 1])
        with ttft_val_col:
            st.number_input("TTFT value", min_value=ttft_min, max_value=ttft_max, step=100, key="input_ttft", label_visibility="collapsed")
            st.session_state.custom_ttft = st.session_state.input_ttft
        with ttft_pct_col:
            ttft_pct_display = reverse_map.get(st.session_state.ttft_percentile, "P95")
            selected_ttft_pct = st.selectbox("TTFT percentile", percentile_options, index=percentile_options.index(ttft_pct_display), key="ttft_pct_selector", label_visibility="collapsed")
            st.session_state.ttft_percentile = percentile_map[selected_ttft_pct]
        st.markdown(f'<div style="font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: -0.5rem;">Range: {ttft_min:,} - {ttft_max:,} ms</div>', unsafe_allow_html=True)

        # === ITL ===
        itl_min, itl_max = get_metric_range("itl", st.session_state.itl_percentile)
        if 'input_itl' not in st.session_state:
            st.session_state.input_itl = default_itl

        st.markdown('<div style="margin-top: 1rem; margin-bottom: 0.25rem;"><span style="color: #EE0000; font-weight: 700; font-size: 0.95rem;">ITL (Inter-Token Latency)</span></div>', unsafe_allow_html=True)
        itl_val_col, itl_pct_col = st.columns([2, 1])
        with itl_val_col:
            st.number_input("ITL value", min_value=itl_min, max_value=itl_max, step=10, key="input_itl", label_visibility="collapsed")
            st.session_state.custom_itl = st.session_state.input_itl
        with itl_pct_col:
            itl_pct_display = reverse_map.get(st.session_state.itl_percentile, "P95")
            selected_itl_pct = st.selectbox("ITL percentile", percentile_options, index=percentile_options.index(itl_pct_display), key="itl_pct_selector", label_visibility="collapsed")
            st.session_state.itl_percentile = percentile_map[selected_itl_pct]
        st.markdown(f'<div style="font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: -0.5rem;">Range: {itl_min:,} - {itl_max:,} ms</div>', unsafe_allow_html=True)

        # === E2E ===
        e2e_min, e2e_max = get_metric_range("e2e", st.session_state.e2e_percentile)
        if 'input_e2e' not in st.session_state:
            st.session_state.input_e2e = default_e2e

        st.markdown('<div style="margin-top: 1rem; margin-bottom: 0.25rem;"><span style="color: #EE0000; font-weight: 700; font-size: 0.95rem;">E2E (End-to-End Latency)</span></div>', unsafe_allow_html=True)
        e2e_val_col, e2e_pct_col = st.columns([2, 1])
        with e2e_val_col:
            st.number_input("E2E value", min_value=e2e_min, max_value=e2e_max, step=1000, key="input_e2e", label_visibility="collapsed")
            st.session_state.custom_e2e = st.session_state.input_e2e
        with e2e_pct_col:
            e2e_pct_display = reverse_map.get(st.session_state.e2e_percentile, "P95")
            selected_e2e_pct = st.selectbox("E2E percentile", percentile_options, index=percentile_options.index(e2e_pct_display), key="e2e_pct_selector", label_visibility="collapsed")
            st.session_state.e2e_percentile = percentile_map[selected_e2e_pct]
        st.markdown(f'<div style="font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: -0.5rem;">Range: {e2e_min:,} - {e2e_max:,} ms</div>', unsafe_allow_html=True)

        # Check if SLO values changed - if so, clear recommendation cache
        curr_ttft = st.session_state.get("custom_ttft")
        curr_itl = st.session_state.get("custom_itl")
        curr_e2e = st.session_state.get("custom_e2e")

        if (curr_ttft != prev_ttft or curr_itl != prev_itl or curr_e2e != prev_e2e):
            # SLO values changed - invalidate recommendation cache
            if st.session_state.get("recommendation_result"):
                st.session_state.recommendation_result = None
            # Store current values for next comparison
            st.session_state._last_ttft = curr_ttft
            st.session_state._last_itl = curr_itl
            st.session_state._last_e2e = curr_e2e
    
    with col2:
        st.markdown("""
        <div class="slo-card">
            <div class="slo-header">
                <span class="slo-title">Workload Profile</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Load token config and workload data from research
        research_data = load_research_slo_ranges()
        use_case_ranges = research_data.get('slo_ranges', {}).get(use_case, {}) if research_data else {}
        token_config = use_case_ranges.get('token_config', {'prompt': 512, 'output': 256})
        prompt_tokens = token_config.get('prompt', 512)
        output_tokens = token_config.get('output', 256)

        workload_data = load_research_workload_patterns()
        pattern = workload_data.get('workload_distributions', {}).get(use_case, {}) if workload_data else {}
        peak_mult = pattern.get('peak_multiplier', 2.0)

        # Store workload profile values in session state for Recommendations tab
        st.session_state.spec_prompt_tokens = prompt_tokens
        st.session_state.spec_output_tokens = output_tokens
        st.session_state.spec_peak_multiplier = peak_mult

        # 1. Editable QPS - support up to 10M QPS for enterprise scale
        # Get research-based default QPS for this use case
        default_qps = estimated_qps  # This is the research-based default
        new_qps = st.number_input("Expected RPS", value=min(qps, 10000000), min_value=1, max_value=10000000, step=1, key="edit_qps", label_visibility="collapsed")
        st.markdown(f'<div style="font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: -0.75rem; margin-bottom: 0.5rem;">Expected RPS: <span style="color: white; font-weight: 700; font-size: 1rem;">{new_qps}</span> <span style="color: rgba(255,255,255,0.4); font-size: 0.75rem;">(default: {default_qps})</span></div>', unsafe_allow_html=True)

        # Store the actual QPS value shown to user (not just custom override)
        st.session_state.spec_expected_qps = new_qps

        if new_qps != qps:
            st.session_state.custom_qps = new_qps
        
        # RPS change warning - show implications of changing from research-based default
        if new_qps > default_qps * 2:
            qps_ratio = new_qps / max(default_qps, 1)
            st.markdown(f'''
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 0.6rem; margin: 0.5rem 0;">
                <div style="color: #ef4444; font-weight: 600; font-size: 0.85rem;">High RPS Warning ({qps_ratio:.1f}x default)</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-top: 0.3rem;">
                    • Requires <strong style="color: #f59e0b;">{int(qps_ratio)}x more GPU replicas</strong><br/>
                    • Estimated cost increase: <strong style="color: #ef4444;">~{int((qps_ratio-1)*100)}%</strong><br/>
                    • Consider load balancing or queue-based architecture
                </div>
            </div>
            ''', unsafe_allow_html=True)
        elif new_qps > default_qps * 1.5:
            qps_ratio = new_qps / max(default_qps, 1)
            st.markdown(f'''
            <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 0.5rem; margin: 0.5rem 0;">
                <div style="color: #f59e0b; font-weight: 600; font-size: 0.8rem;">Elevated RPS ({qps_ratio:.1f}x default)</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; margin-top: 0.2rem;">
                    May need additional replicas. Cost ~{int((qps_ratio-1)*100)}% higher.
                </div>
            </div>
            ''', unsafe_allow_html=True)
        elif new_qps < default_qps * 0.5 and default_qps > 1:
            st.markdown(f'''
            <div style="background: rgba(56, 239, 125, 0.1); border: 1px solid rgba(56, 239, 125, 0.3); border-radius: 8px; padding: 0.5rem; margin: 0.5rem 0;">
                <div style="color: #38ef7d; font-weight: 600; font-size: 0.8rem;">Low RPS - Cost Savings Possible</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; margin-top: 0.2rem;">
                    Single replica may suffice. Consider smaller GPU or spot instances.
                </div>
            </div>
            ''', unsafe_allow_html=True)

        # 2-4. Fixed workload values with inline descriptions (like datasets)
        st.markdown(f"""
        <div style="margin-top: 0.5rem; background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: 8px;">
            <div style="padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: rgba(255,255,255,0.95); font-size: 0.95rem; font-weight: 500;">Mean Input Tokens</span>
                    <span style="color: white; font-weight: 700; font-size: 1.1rem; background: #000000; padding: 3px 10px; border-radius: 4px;">{prompt_tokens}</span>
            </div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Average input length per request (research-based for {use_case.replace('_', ' ')})</div>
            </div>
            <div style="padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: rgba(255,255,255,0.95); font-size: 0.95rem; font-weight: 500;">Mean Output Tokens</span>
                    <span style="color: white; font-weight: 700; font-size: 1.1rem; background: #000000; padding: 3px 10px; border-radius: 4px;">{output_tokens}</span>
                </div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Average output length generated per request</div>
            </div>
            <div style="padding: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: rgba(255,255,255,0.95); font-size: 0.95rem; font-weight: 500;">Peak Multiplier</span>
                    <span style="color: white; font-weight: 700; font-size: 1.1rem; background: #000000; padding: 3px 10px; border-radius: 4px;">{peak_mult}x</span>
                </div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Capacity buffer for traffic spikes (user behavior patterns)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. Informational messages from research data - black background, no emojis
        workload_messages = get_workload_insights(use_case, new_qps, user_count)
        
        for icon, color, text, severity in workload_messages[:3]:  # Limit to 3 for space
            st.markdown(f'<div style="font-size: 0.85rem; color: white; padding: 0.4rem 0.5rem; line-height: 1.4; background: #000000; border-radius: 6px; margin: 4px 0;">{text}</div>', unsafe_allow_html=True)
    
    with col3:
        # Accuracy Benchmarks - show which benchmarks are used for this use case
        # Each entry: (name, weight, color, tooltip_description) - all black background
        # Updated based on USE_CASE_METHODOLOGY.md (December 2024)
        # Focus on TOP 3 benchmarks: τ²-Bench (91%), LiveCodeBench (84%), MMLU-Pro (83%), GPQA (82%)
        TASK_DATASETS = {
            "chatbot_conversational": [
                ("τ²-Bench", 45, "Conversational AI agentic workflow (114 tasks)"),
                ("MMLU-Pro", 35, "General knowledge (12,032 questions)"),
                ("GPQA", 20, "Scientific reasoning (198 questions)"),
            ],
            "code_completion": [
                ("LiveCodeBench", 45, "Code benchmark (315 questions)"),
                ("τ²-Bench", 35, "Agentic code assistance (114 tasks)"),
                ("MMLU-Pro", 20, "Knowledge for context"),
            ],
            "code_generation_detailed": [
                ("LiveCodeBench", 40, "Code generation (315 questions)"),
                ("τ²-Bench", 40, "Agentic reasoning (114 tasks)"),
                ("GPQA", 20, "Scientific reasoning for explanations"),
            ],
            "translation": [
                ("τ²-Bench", 45, "Language agentic tasks (114 tasks)"),
                ("MMLU-Pro", 35, "Language understanding (12,032 questions)"),
                ("GPQA", 20, "Reasoning"),
            ],
            "content_generation": [
                ("τ²-Bench", 45, "Creative agentic workflow (114 tasks)"),
                ("MMLU-Pro", 35, "General knowledge for facts"),
                ("GPQA", 20, "Reasoning"),
            ],
            "summarization_short": [
                ("τ²-Bench", 45, "Summarization agentic (114 tasks)"),
                ("MMLU-Pro", 35, "Comprehension (12,032 questions)"),
                ("GPQA", 20, "Reasoning"),
            ],
            "document_analysis_rag": [
                ("τ²-Bench", 50, "RAG is agentic workflow - DOMINANT (114 tasks)"),
                ("GPQA", 30, "Scientific reasoning for factual answers"),
                ("MMLU-Pro", 20, "Knowledge retrieval"),
            ],
            "long_document_summarization": [
                ("τ²-Bench", 50, "Long doc handling is agentic (114 tasks)"),
                ("MMLU-Pro", 30, "Knowledge for understanding"),
                ("GPQA", 20, "Reasoning"),
            ],
            "research_legal_analysis": [
                ("τ²-Bench", 55, "Research analysis is agentic reasoning - CRITICAL"),
                ("GPQA", 25, "Scientific reasoning (198 questions)"),
                ("MMLU-Pro", 20, "Knowledge (12,032 questions)"),
            ],
        }
        
        datasets = TASK_DATASETS.get(use_case, TASK_DATASETS["chatbot_conversational"])
        
        st.markdown("""
        <div class="slo-card">
            <div class="slo-header">
                <span class="slo-title">Accuracy Benchmarks</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display datasets with weights - black background, white text for visibility
        datasets_html = '<div style="background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;">'
        for item in datasets:
            name = item[0]
            weight = item[1]
            tooltip = item[2] if len(item) > 2 else ""
            datasets_html += f'''<div style="padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: rgba(255,255,255,0.95); font-size: 0.95rem; font-weight: 500;">{name}</span>
                    <span style="color: white; font-weight: 700; font-size: 0.95rem; background: #000000; padding: 3px 10px; border-radius: 4px;">{weight}%</span>
            </div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">{tooltip}</div>
            </div>'''
        datasets_html += '</div>'
        datasets_html += '<div style="font-size: 0.75rem; color: white; font-weight: 600; margin-top: 0.5rem;">Weights from Artificial Analysis Intelligence Index</div>'
        st.markdown(datasets_html, unsafe_allow_html=True)

    with col4:
        # PRIORITIES card - interactive priority dropdowns with weights
        st.markdown("""
        <div class="slo-card">
            <div class="slo-header">
                <span class="slo-title">Priorities</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Load priority weights config from backend (cached)
        priority_config = fetch_priority_weights()
        # priority_config structure: {priority_weights: {accuracy: {low: 3, ...}, ...}, defaults: {...}}
        priority_weights_map = priority_config.get("priority_weights", {}) if priority_config else {}
        defaults_config = priority_config.get("defaults", {}) if priority_config else {}
        logger.debug(f"Priority config loaded: weights_map keys={list(priority_weights_map.keys())}, defaults={defaults_config}")

        # Priority options for dropdowns
        priority_options = ["Low", "Medium", "High"]
        priority_to_value = {"Low": "low", "Medium": "medium", "High": "high"}
        value_to_priority = {"low": "Low", "medium": "Medium", "high": "High"}

        # Helper to get weight for a priority level
        def get_weight_for_priority(dimension: str, priority_level: str) -> int:
            pw = priority_weights_map.get(dimension, {})
            default_weights = defaults_config.get("weights", {"accuracy": 5, "cost": 4, "latency": 2, "complexity": 2})
            weight = pw.get(priority_level, default_weights.get(dimension, 5))
            logger.info(f"get_weight_for_priority({dimension}, {priority_level}): pw={pw}, weight={weight}")
            return weight

        # Get extraction result to check for LLM-detected priorities
        extraction = st.session_state.get('extraction_result', {})

        # Initialize session state for priorities if not set
        if 'accuracy_priority' not in st.session_state:
            st.session_state.accuracy_priority = extraction.get('accuracy_priority', 'medium')
        if 'cost_priority' not in st.session_state:
            st.session_state.cost_priority = extraction.get('cost_priority', 'medium')
        if 'latency_priority' not in st.session_state:
            st.session_state.latency_priority = extraction.get('latency_priority', 'medium')
        # Note: complexity_priority removed from UI (backend still extracts it)

        # Initialize weights based on priorities
        if 'weight_accuracy' not in st.session_state:
            st.session_state.weight_accuracy = get_weight_for_priority("accuracy", st.session_state.accuracy_priority)
        if 'weight_cost' not in st.session_state:
            st.session_state.weight_cost = get_weight_for_priority("cost", st.session_state.cost_priority)
        if 'weight_latency' not in st.session_state:
            st.session_state.weight_latency = get_weight_for_priority("latency", st.session_state.latency_priority)
        # Note: weight_complexity removed from UI (hardcoded to 0 in get_enhanced_recommendation)

        # Container for the priority rows
        st.markdown('<div style="background: rgba(255,255,255,0.03); padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem;">', unsafe_allow_html=True)

        # Row 1: Accuracy
        acc_col1, acc_col2, acc_col3 = st.columns([2, 1.5, 1])
        with acc_col1:
            st.markdown('<span style="color: rgba(255,255,255,0.95); font-size: 0.9rem; font-weight: 500; line-height: 2.5;">Accuracy</span>', unsafe_allow_html=True)
        with acc_col2:
            acc_priority_display = value_to_priority.get(st.session_state.accuracy_priority, "Medium")
            new_acc_priority = st.selectbox("Accuracy", priority_options, index=priority_options.index(acc_priority_display), key="acc_priority_select", label_visibility="collapsed")
            new_acc_priority_val = priority_to_value[new_acc_priority]
            if new_acc_priority_val != st.session_state.accuracy_priority:
                st.session_state.accuracy_priority = new_acc_priority_val
                st.session_state.weight_accuracy = get_weight_for_priority("accuracy", new_acc_priority_val)
        with acc_col3:
            new_acc_weight = st.number_input("Accuracy weight", min_value=0, max_value=10, value=st.session_state.weight_accuracy, key="acc_weight_input", label_visibility="collapsed")
            if new_acc_weight != st.session_state.weight_accuracy:
                st.session_state.weight_accuracy = new_acc_weight

        # Row 2: Cost
        cost_col1, cost_col2, cost_col3 = st.columns([2, 1.5, 1])
        with cost_col1:
            st.markdown('<span style="color: rgba(255,255,255,0.95); font-size: 0.9rem; font-weight: 500; line-height: 2.5;">Cost</span>', unsafe_allow_html=True)
        with cost_col2:
            cost_priority_display = value_to_priority.get(st.session_state.cost_priority, "Medium")
            new_cost_priority = st.selectbox("Cost", priority_options, index=priority_options.index(cost_priority_display), key="cost_priority_select", label_visibility="collapsed")
            new_cost_priority_val = priority_to_value[new_cost_priority]
            if new_cost_priority_val != st.session_state.cost_priority:
                st.session_state.cost_priority = new_cost_priority_val
                st.session_state.weight_cost = get_weight_for_priority("cost", new_cost_priority_val)
        with cost_col3:
            new_cost_weight = st.number_input("Cost weight", min_value=0, max_value=10, value=st.session_state.weight_cost, key="cost_weight_input", label_visibility="collapsed")
            if new_cost_weight != st.session_state.weight_cost:
                st.session_state.weight_cost = new_cost_weight

        # Row 3: Latency
        lat_col1, lat_col2, lat_col3 = st.columns([2, 1.5, 1])
        with lat_col1:
            st.markdown('<span style="color: rgba(255,255,255,0.95); font-size: 0.9rem; font-weight: 500; line-height: 2.5;">Latency</span>', unsafe_allow_html=True)
        with lat_col2:
            lat_priority_display = value_to_priority.get(st.session_state.latency_priority, "Medium")
            new_lat_priority = st.selectbox("Latency", priority_options, index=priority_options.index(lat_priority_display), key="lat_priority_select", label_visibility="collapsed")
            new_lat_priority_val = priority_to_value[new_lat_priority]
            if new_lat_priority_val != st.session_state.latency_priority:
                st.session_state.latency_priority = new_lat_priority_val
                st.session_state.weight_latency = get_weight_for_priority("latency", new_lat_priority_val)
        with lat_col3:
            new_lat_weight = st.number_input("Latency weight", min_value=0, max_value=10, value=st.session_state.weight_latency, key="lat_weight_input", label_visibility="collapsed")
            if new_lat_weight != st.session_state.weight_latency:
                st.session_state.weight_latency = new_lat_weight

        # Note: Complexity row removed from UI but scoring logic preserved in backend
        # Complexity weight is hardcoded to 0 in get_enhanced_recommendation()

        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# WINNER DETAILS DIALOG
# =============================================================================

@st.dialog("Winner Details", width="large")
def show_winner_details_dialog():
    """Show winner details in a modal dialog."""
    winner = st.session_state.get('balanced_winner') or st.session_state.get('winner_recommendation')
    priority = st.session_state.get('winner_priority', 'balanced')
    extraction = st.session_state.get('winner_extraction', {})
    
    if not winner:
        st.warning("No winner data available.")
        return
    
    # Render the winner details
    _render_winner_details(winner, priority, extraction)
    
    # Close button
    if st.button("Close", key="close_dialog_btn", use_container_width=True):
        st.session_state.show_winner_dialog = False
        st.rerun()


# =============================================================================
# CATEGORY EXPLORATION DIALOG
# =============================================================================

@st.dialog("Top 5 Models", width="large")
def show_category_dialog():
    """Show top 5 models for a category with performance data."""
    category = st.session_state.get('explore_category', 'balanced')
    
    # Get use case for context
    use_case = st.session_state.get("detected_use_case", "chatbot_conversational")
    
    # Token config per use case (for throughput calculation)
    USE_CASE_OUTPUT_TOKENS = {
        "chatbot_conversational": 256,
        "code_completion": 256,
        "translation": 256,
        "content_creation": 256,
        "code_generation_detailed": 1024,
        "summarization_short": 512,
        "document_analysis_rag": 512,
        "long_document_summarization": 1536,
        "research_legal_analysis": 1536,
    }
    output_tokens = USE_CASE_OUTPUT_TOKENS.get(use_case, 256)
    
    # Category config - theme colors (no emojis)
    category_config = {
        "balanced": {"title": "Balanced - Top 5", "color": "#EE0000", "field": "final", "top5_key": "top5_balanced"},
        "accuracy": {"title": "Best Accuracy - Top 5", "color": "#EE0000", "field": "accuracy", "top5_key": "top5_accuracy"},
        "latency": {"title": "Best Latency - Top 5", "color": "#EE0000", "field": "latency", "top5_key": "top5_latency"},
        "cost": {"title": "Best Cost - Top 5", "color": "#EE0000", "field": "cost", "top5_key": "top5_cost"},
    }
    
    config = category_config.get(category, category_config["balanced"])
    top5_list = st.session_state.get(config["top5_key"], [])
    
    # Theme CSS
    st.markdown("""
    <style>
        [data-testid="stDialog"] > div {
            background: #000000 !important;
        }
        [data-testid="stDialog"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stDialog"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stDialog"] [data-testid="stMarkdownContainer"] div {
            color: #ffffff !important;
        }
        [data-testid="stDialog"] [data-testid="stButton"] button {
            background: #000000 !important;
            border: 1px solid #EE0000 !important;
            color: white !important;
        }
        [data-testid="stDialog"] [data-testid="stButton"] button:hover {
            background: #EE0000 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div style="background: #EE0000; padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <h3 style="color: #ffffff; margin: 0; font-size: 1.3rem;">{config['title']}</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Use case: <strong style="color: white;">{format_use_case_name(use_case)}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
        
    if not top5_list:
        st.warning("No models available for this category.")
        if st.button("Close", key="close_cat_dialog_empty"):
            st.session_state.show_category_dialog = False
            st.rerun()
        return
    
    # Helper to get scores
    def get_model_scores(rec):
        backend_scores = rec.get("scores", {}) or {}
        ui_breakdown = rec.get("score_breakdown", {}) or {}
        model_name = rec.get('model_name', 'Unknown')
        raw_aa = rec.get('raw_aa_accuracy', 0)
        if not raw_aa:
            raw_aa = get_raw_aa_accuracy(model_name, use_case)
        return {
            "accuracy": raw_aa,
            "latency": backend_scores.get("latency_score", ui_breakdown.get("latency_score", 0)),
            "cost": backend_scores.get("price_score", ui_breakdown.get("cost_score", 0)),
            "complexity": backend_scores.get("complexity_score", ui_breakdown.get("capacity_score", 0)),
            "final": backend_scores.get("balanced_score", rec.get("final_score", 0)),
        }
    
    # Helper to calculate throughput: output_tokens / (E2E_ms / 1000) = output_tokens * 1000 / E2E_ms
    def calc_throughput(e2e_ms, output_toks):
        if isinstance(e2e_ms, str) or e2e_ms is None or e2e_ms <= 0:
            return None
        return output_toks * 1000 / e2e_ms  # tok/s
    
    # Render each model in the top 5
    for i, rec in enumerate(top5_list):
        scores = get_model_scores(rec)
        model_name = format_display_name(rec.get('model_name', 'Unknown'))
        gpu_cfg = rec.get('gpu_config', {}) or {}
        hw_type = gpu_cfg.get('gpu_type', rec.get('hardware', 'H100'))
        hw_count = gpu_cfg.get('gpu_count', rec.get('hardware_count', 1))
        hw_display = f"{hw_count}x {hw_type}"
        tp = gpu_cfg.get('tensor_parallel', 1)
        
        # Get selected percentile from session state
        selected_percentile = st.session_state.get('slo_percentile', 'p95')
        percentile_suffix = selected_percentile  # mean, p90, p95, p99
        percentile_label = selected_percentile.upper() if selected_percentile != 'mean' else 'Mean'
        
        # Get benchmark metrics with all percentiles (from backend)
        benchmark_metrics = rec.get('benchmark_metrics', {}) or {}
        
        # Get values for selected percentile
        ttft = benchmark_metrics.get(f'ttft_{percentile_suffix}', rec.get('predicted_ttft_p95_ms', 'N/A'))
        itl = benchmark_metrics.get(f'itl_{percentile_suffix}', rec.get('predicted_itl_p95_ms', 'N/A'))
        e2e = benchmark_metrics.get(f'e2e_{percentile_suffix}', rec.get('predicted_e2e_p95_ms', 'N/A'))
        tps = benchmark_metrics.get(f'tps_{percentile_suffix}', 0)
        
        # Use TPS from benchmark if available, else calculate from E2E
        if tps and tps > 0:
            throughput_display = f"{tps:.0f} tok/s"
        else:
            throughput = calc_throughput(e2e, output_tokens)
            throughput_display = f"{throughput:.0f} tok/s" if throughput else "N/A"
        
        highlight_score = scores.get(config["field"], 0)
        # Theme rank colors: red for #1, gray shades for others
        rank_colors = ["#EE0000", "#666666", "#555555", "#444444", "#333333"]
        rank_color = rank_colors[i] if i < 5 else "#333333"
        rank_text_color = "#ffffff"  # Always white text
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(30,30,30,0.95), rgba(20,20,20,0.95)); 
                    border: 1px solid #333333; border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="background: {rank_color}; color: {rank_text_color}; font-weight: 800; padding: 0.25rem 0.6rem; 
                                border-radius: 6px; font-size: 0.85rem;">#{i+1}</span>
                    <div>
                        <div style="color: white; font-weight: 700; font-size: 1.05rem;">{model_name}</div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">{hw_display} (TP={tp})</div>
        </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #ffffff; font-size: 1.75rem; font-weight: 800;">{highlight_score:.0f}</div>
                    <div style="color: rgba(255,255,255,0.4); font-size: 0.65rem; text-transform: uppercase;">FINAL</div>
                </div>
            </div>
            <div style="display: flex; gap: 1rem; margin-bottom: 0.75rem; padding: 0.5rem; 
                        background: rgba(0,0,0,0.4); border-radius: 8px;">
                <span style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">Acc: {scores['accuracy']:.0f}</span>
                <span style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">Lat: {scores['latency']:.0f}</span>
                <span style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">Cost: {scores['cost']:.0f}</span>
                <span style="color: #ffffff; font-size: 0.8rem; font-weight: 600;">Final: {scores['final']:.1f}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem;">
                <div style="background: #EE0000; padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.65rem;">TTFT ({percentile_label})</div>
                    <div style="color: #ffffff; font-weight: 700; font-size: 0.9rem;">{ttft if isinstance(ttft, str) else f'{ttft:.0f}ms'}</div>
                </div>
                <div style="background: #EE0000; padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.65rem;">ITL ({percentile_label})</div>
                    <div style="color: #ffffff; font-weight: 700; font-size: 0.9rem;">{itl if isinstance(itl, str) else f'{itl:.0f}ms'}</div>
                </div>
                <div style="background: #EE0000; padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.65rem;">E2E ({percentile_label})</div>
                    <div style="color: #ffffff; font-weight: 700; font-size: 0.9rem;">{e2e if isinstance(e2e, str) else f'{e2e:.0f}ms'}</div>
                </div>
                <div style="background: #EE0000; padding: 0.4rem; border-radius: 6px; text-align: center;">
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.65rem;">Throughput</div>
                    <div style="color: #ffffff; font-weight: 700; font-size: 0.9rem;">{throughput_display}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Close button
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("Close", key="close_cat_dialog", use_container_width=True):
        st.session_state.show_category_dialog = False
        st.rerun()


@st.dialog("All Recommendation Options", width="large")
def show_full_table_dialog():
    """Show the full recommendations table with all categories."""
    ranked_response = st.session_state.get('ranked_response')
    
    if not ranked_response:
        st.warning("No recommendations available. Please run the recommendation process first.")
        if st.button("Close", key="close_full_table_empty"):
            st.session_state.show_full_table_dialog = False
            st.rerun()
        return
    
    # Dark theme styling for dialog
    st.markdown("""<style>[data-testid="stDialog"],[data-testid="stDialog"] > div,[data-testid="stDialog"] > div > div {background: #0d1117 !important;}[data-testid="stDialog"] .stMarkdown,[data-testid="stDialog"] p,[data-testid="stDialog"] span,[data-testid="stDialog"] th,[data-testid="stDialog"] td {color: #f0f6fc !important;}</style>""", unsafe_allow_html=True)
    
    # Header
    st.markdown('<div style="background: linear-gradient(135deg, #EE0000, #cc0000); padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;"><h2 style="color: white; margin: 0; font-size: 1.5rem;">Configuration Options</h2><p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">All viable deployment configurations ranked by category</p></div>', unsafe_allow_html=True)
    
    total_configs = ranked_response.get("total_configs_evaluated", 0)
    configs_after_filters = ranked_response.get("configs_after_filters", 0)
    
    st.markdown(f'<div style="color: #ffffff; margin-bottom: 1rem; font-size: 0.9rem;">Evaluated <span style="color: #06b6d4; font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="color: #10b981; font-weight: 600;">{configs_after_filters}</span> unique options</div>', unsafe_allow_html=True)
    
    # Define categories
    categories = [
        ("balanced", "Balanced", "#EE0000"),
        ("best_accuracy", "Best Accuracy", "#ffffff"),
        ("lowest_cost", "Lowest Cost", "#f59e0b"),
        ("lowest_latency", "Lowest Latency", "#ffffff"),
    ]
    
    # Helper function to format GPU config
    def format_gpu_config(gpu_config: dict) -> str:
        if not isinstance(gpu_config, dict):
            return "Unknown"
        gpu_type = gpu_config.get("gpu_type", "Unknown")
        gpu_count = gpu_config.get("gpu_count", 1)
        tp = gpu_config.get("tensor_parallel", 1)
        replicas = gpu_config.get("replicas", 1)
        return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"
    
    # Build table rows
    all_rows = []
    for cat_key, cat_name, cat_color in categories:
        recs = ranked_response.get(cat_key, [])
        if not recs:
            all_rows.append(f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);"><td style="padding: 0.75rem 0.5rem;"><span style="color: {cat_color}; font-weight: 600;">{cat_name}</span></td><td colspan="7" style="padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.5); font-style: italic;">No configurations found</td></tr>')
        else:
            for i, rec in enumerate(recs[:5]):  # Show top 5 per category
                model_name = format_display_name(rec.get("model_name", "Unknown"))
                gpu_config = rec.get("gpu_config", {})
                gpu_str = format_gpu_config(gpu_config)
                ttft = rec.get("predicted_ttft_p95_ms", 0)
                cost = rec.get("cost_per_month_usd", 0)
                scores = rec.get("scores", {}) or {}
                accuracy = scores.get("accuracy_score", 0)
                balanced = scores.get("balanced_score", 0)
                meets_slo = rec.get("meets_slo", False)
                slo_icon = "Yes" if meets_slo else "No"
                
                cat_display = f'<span style="color: {cat_color}; font-weight: 600;">{cat_name}</span> (+{len(recs)-1})' if i == 0 else ""
                
                row = f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);"><td style="padding: 0.75rem 0.5rem;">{cat_display}</td><td style="padding: 0.75rem 0.5rem; color: white; font-weight: 500;">{model_name}</td><td style="padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem;">{gpu_str}</td><td style="padding: 0.75rem 0.5rem; text-align: right; color: #06b6d4;">{ttft:.0f}ms</td><td style="padding: 0.75rem 0.5rem; text-align: right; color: #f59e0b;">${cost:,.0f}</td><td style="padding: 0.75rem 0.5rem; text-align: center; color: #10b981;">{accuracy:.0f}</td><td style="padding: 0.75rem 0.5rem; text-align: center; color: #8b5cf6;">{balanced:.1f}</td><td style="padding: 0.75rem 0.5rem; text-align: center;">{slo_icon}</td></tr>'
                all_rows.append(row)
    
    # Table header
    header = '<thead><tr style="border-bottom: 2px solid rgba(255,255,255,0.2);"><th style="text-align: left; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Category</th><th style="text-align: left; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Model</th><th style="text-align: left; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">GPU Config</th><th style="text-align: right; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">TTFT</th><th style="text-align: right; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Cost/mo</th><th style="text-align: center; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Acc</th><th style="text-align: center; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Score</th><th style="text-align: center; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">SLO</th></tr></thead>'
    
    # Render table
    table_html = f'<table style="width: 100%; border-collapse: collapse; background: rgba(13, 17, 23, 0.95); border-radius: 8px;">{header}<tbody>{"".join(all_rows)}</tbody></table>'
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Close button
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Close", key="close_full_table_dialog", use_container_width=True):
        st.session_state.show_full_table_dialog = False
        st.rerun()


def render_options_list_inline():
    """Render the expanded options list content inline on the page."""
    ranked_response = st.session_state.get('ranked_response')

    if not ranked_response:
        st.warning("No recommendations available. Please run the recommendation process first.")
        return

    # Header
    st.markdown('<div style="background: linear-gradient(135deg, #EE0000, #cc0000); padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; margin-top: 1rem;"><h2 style="color: white; margin: 0; font-size: 1.5rem;">Configuration Options</h2><p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">All viable deployment configurations ranked by category</p></div>', unsafe_allow_html=True)

    total_configs = ranked_response.get("total_configs_evaluated", 0)
    configs_after_filters = ranked_response.get("configs_after_filters", 0)

    st.markdown(f'<div style="color: #ffffff; margin-bottom: 1rem; font-size: 0.9rem;">Evaluated <span style="color: #06b6d4; font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="color: #10b981; font-weight: 600;">{configs_after_filters}</span> unique options</div>', unsafe_allow_html=True)

    # Define categories
    categories = [
        ("balanced", "Balanced", "#EE0000"),
        ("best_accuracy", "Best Accuracy", "#ffffff"),
        ("lowest_cost", "Lowest Cost", "#f59e0b"),
        ("lowest_latency", "Lowest Latency", "#ffffff"),
    ]

    # Helper function to format GPU config
    def format_gpu_config(gpu_config: dict) -> str:
        if not isinstance(gpu_config, dict):
            return "Unknown"
        gpu_type = gpu_config.get("gpu_type", "Unknown")
        gpu_count = gpu_config.get("gpu_count", 1)
        tp = gpu_config.get("tensor_parallel", 1)
        replicas = gpu_config.get("replicas", 1)
        return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"

    # Build table rows
    all_rows = []
    for cat_key, cat_name, cat_color in categories:
        recs = ranked_response.get(cat_key, [])
        if not recs:
            all_rows.append(f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);"><td style="padding: 0.75rem 0.5rem;"><span style="color: {cat_color}; font-weight: 600;">{cat_name}</span></td><td colspan="7" style="padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.5); font-style: italic;">No configurations found</td></tr>')
        else:
            for i, rec in enumerate(recs[:5]):  # Show top 5 per category
                model_name = format_display_name(rec.get("model_name", "Unknown"))
                gpu_config = rec.get("gpu_config", {})
                gpu_str = format_gpu_config(gpu_config)
                ttft = rec.get("predicted_ttft_p95_ms", 0)
                cost = rec.get("cost_per_month_usd", 0)
                scores = rec.get("scores", {}) or {}
                accuracy = scores.get("accuracy_score", 0)
                balanced = scores.get("balanced_score", 0)
                meets_slo = rec.get("meets_slo", False)
                slo_icon = "Yes" if meets_slo else "No"

                cat_display = f'<span style="color: {cat_color}; font-weight: 600;">{cat_name}</span> (+{len(recs)-1})' if i == 0 else ""

                row = f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);"><td style="padding: 0.75rem 0.5rem;">{cat_display}</td><td style="padding: 0.75rem 0.5rem; color: white; font-weight: 500;">{model_name}</td><td style="padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem;">{gpu_str}</td><td style="padding: 0.75rem 0.5rem; text-align: right; color: #06b6d4;">{ttft:.0f}ms</td><td style="padding: 0.75rem 0.5rem; text-align: right; color: #f59e0b;">${cost:,.0f}</td><td style="padding: 0.75rem 0.5rem; text-align: center; color: #10b981;">{accuracy:.0f}</td><td style="padding: 0.75rem 0.5rem; text-align: center; color: #8b5cf6;">{balanced:.1f}</td><td style="padding: 0.75rem 0.5rem; text-align: center;">{slo_icon}</td></tr>'
                all_rows.append(row)

    # Table header
    header = '<thead><tr style="border-bottom: 2px solid rgba(255,255,255,0.2);"><th style="text-align: left; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Category</th><th style="text-align: left; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Model</th><th style="text-align: left; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">GPU Config</th><th style="text-align: right; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">TTFT</th><th style="text-align: right; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Cost/mo</th><th style="text-align: center; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Acc</th><th style="text-align: center; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">Score</th><th style="text-align: center; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;">SLO</th></tr></thead>'

    # Render table
    table_html = f'<table style="width: 100%; border-collapse: collapse; background: rgba(13, 17, 23, 0.95); border-radius: 8px;">{header}<tbody>{"".join(all_rows)}</tbody></table>'

    st.markdown(table_html, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Show dialogs if triggered
    # Only show ONE dialog at a time (Streamlit limitation)
    if st.session_state.show_winner_dialog is True and st.session_state.balanced_winner is not None:
        # Reset other dialogs
        st.session_state.show_category_dialog = False
        st.session_state.show_full_table_dialog = False
        show_winner_details_dialog()
    elif st.session_state.get('show_category_dialog') is True:
        # Reset other dialogs
        st.session_state.show_winner_dialog = False
        st.session_state.show_full_table_dialog = False
        show_category_dialog()
    elif st.session_state.get('show_full_table_dialog') is True:
        # Reset other dialogs
        st.session_state.show_winner_dialog = False
        st.session_state.show_category_dialog = False
        show_full_table_dialog()
    
    # Load models
    if st.session_state.models_df is None:
        st.session_state.models_df = load_206_models()
    models_df = st.session_state.models_df
    
    # Default priority
    priority = "balanced"
    
    # Main Content - Compact hero
    render_hero()
    
    # Tab-based navigation (4 tabs)
    tab1, tab2, tab3, tab4 = st.tabs(["Define Use Case", "Technical Specification", "Recommendations", "About"])
    
    with tab1:
        render_use_case_input_tab(priority, models_df)
    
    with tab2:
        render_technical_specs_tab(priority, models_df)
    
    with tab3:
        render_results_tab(priority, models_df)

    with tab4:
        render_about_section(models_df)

    
def render_use_case_input_tab(priority: str, models_df: pd.DataFrame):
    """Tab 1: Use case input interface."""

    def clear_dialog_states():
        """Clear all dialog and expanded states when starting a new use case."""
        st.session_state.show_full_table_dialog = False
        st.session_state.show_category_dialog = False
        st.session_state.show_winner_dialog = False
        st.session_state.show_options_list_expanded = False

    st.markdown('<div class="section-header">Describe your use case or select from 9 predefined scenarios</div>', unsafe_allow_html=True)

    # Row 1: 5 task buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Chat Completion", use_container_width=True, key="task_chat"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.user_input = "Customer service chatbot for 30 users."

    with col2:
        if st.button("Code Completion", use_container_width=True, key="task_code"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.user_input = "IDE code completion tool for 30 developers."

    with col3:
        if st.button("Document Q&A", use_container_width=True, key="task_rag"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.user_input = "Document Q&A system for enterprise knowledge base, 30 users."

    with col4:
        if st.button("Summarization", use_container_width=True, key="task_summ"):
            clear_dialog_states()
            # With priority (cost-effective) to show filtering
            st.session_state.user_input = "News article summarization for 30 users, cost-effective solution preferred."

    with col5:
        if st.button("Legal Analysis", use_container_width=True, key="task_legal"):
            clear_dialog_states()
            # With priority (accuracy) to show filtering
            st.session_state.user_input = "Legal document analysis for 30 lawyers, accuracy is critical."

    # Row 2: 4 more task buttons
    col6, col7, col8, col9 = st.columns(4)

    with col6:
        if st.button("Translation", use_container_width=True, key="task_trans"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.user_input = "Multi-language translation service for 30 users."

    with col7:
        if st.button("Content Generation", use_container_width=True, key="task_content"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.user_input = "Content generation tool for marketing team, 30 users."

    with col8:
        if st.button("Long Doc Summary", use_container_width=True, key="task_longdoc"):
            clear_dialog_states()
            # With priority (accuracy) to show filtering
            st.session_state.user_input = "Long document summarization for research papers, 30 researchers, accuracy matters."

    with col9:
        if st.button("Code Generation", use_container_width=True, key="task_codegen"):
            clear_dialog_states()
            # Simple prompt - no priority, no hardware = show all configs
            st.session_state.user_input = "Full code generation tool for implementing features, 30 developers."
    
    # Input area with validation
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_input = st.text_area(
        "Your requirements:",
        value=st.session_state.user_input,
        height=120,
        max_chars=2000,  # Corporate standard: limit input length
        placeholder="Describe your LLM use case in natural language...\n\nExample: I need a chatbot for customer support with 30 users. Low latency is important, and we have H100 GPUs available.",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show character count - white text
    char_count = len(user_input) if user_input else 0
    st.markdown(f'<div style="text-align: right; font-size: 0.75rem; color: white; margin-top: -0.5rem;">{char_count}/2000 characters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 1, 2])
    with col1:
        # Disable button if input is too short
        analyze_disabled = len(user_input.strip()) < 10 if user_input else True
        analyze_clicked = st.button("Analyze & Recommend", type="primary", use_container_width=True, disabled=analyze_disabled)
        if analyze_disabled and user_input and len(user_input.strip()) < 10:
            st.caption("Please enter at least 10 characters")
    with col2:
        if st.button("Clear", use_container_width=True):
            # Complete session state reset
            for key in ['user_input', 'extraction_result', 'recommendation_result', 
                       'extraction_approved', 'slo_approved', 'edited_extraction',
                       'custom_ttft', 'custom_itl', 'custom_e2e', 'custom_qps', 'used_priority']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.user_input = ""
            st.rerun()
    
    # Input validation before analysis
    if analyze_clicked and user_input and len(user_input.strip()) >= 10:
        st.session_state.user_input = user_input
        # Reset workflow state
        st.session_state.extraction_approved = None
        st.session_state.slo_approved = None
        st.session_state.recommendation_result = None
        st.session_state.edited_extraction = None
        
        # Show progress bar for better UX
        progress_container = st.empty()
        with progress_container:
            progress_bar = st.progress(0, text="Initializing extraction...")
            
        try:
            progress_bar.progress(20, text="Analyzing input text...")
            extraction = extract_business_context(user_input)
            progress_bar.progress(80, text="Extraction complete!")
            
            if extraction:
                # Clear old recommendation data when new extraction is done
                st.session_state.recommendation_result = None
                st.session_state.extraction_approved = None
                st.session_state.slo_approved = None
                st.session_state.edited_extraction = None
                st.session_state.ranked_response = None

                # Reset priority weights from extraction (LLM-detected priorities)
                # Note: complexity_priority and weight_complexity removed from UI
                for key in ['accuracy_priority', 'cost_priority', 'latency_priority',
                           'weight_accuracy', 'weight_cost', 'weight_latency']:
                    if key in st.session_state:
                        del st.session_state[key]

                # Store new extraction
                st.session_state.extraction_result = extraction

                # Initialize priorities and weights immediately from extraction
                # This ensures weights are set before the recommendation API is called
                priority_config = fetch_priority_weights()
                pw_map = priority_config.get("priority_weights", {}) if priority_config else {}
                defaults_cfg = priority_config.get("defaults", {}) if priority_config else {}
                default_weights = defaults_cfg.get("weights", {"accuracy": 5, "cost": 4, "latency": 2})

                # Set priorities from extraction (LLM-detected) or default to medium
                # Note: complexity_priority removed from UI (backend still extracts it)
                st.session_state.accuracy_priority = extraction.get('accuracy_priority', 'medium')
                st.session_state.cost_priority = extraction.get('cost_priority', 'medium')
                st.session_state.latency_priority = extraction.get('latency_priority', 'medium')

                # Set weights based on priorities using the JSON config mapping
                # Note: weight_complexity removed from UI (hardcoded to 0 in API call)
                st.session_state.weight_accuracy = pw_map.get("accuracy", {}).get(st.session_state.accuracy_priority, default_weights["accuracy"])
                st.session_state.weight_cost = pw_map.get("cost", {}).get(st.session_state.cost_priority, default_weights["cost"])
                st.session_state.weight_latency = pw_map.get("latency", {}).get(st.session_state.latency_priority, default_weights["latency"])

                logger.info(f"Initialized priorities from extraction: accuracy={st.session_state.accuracy_priority}, "
                           f"cost={st.session_state.cost_priority}, latency={st.session_state.latency_priority}")
                logger.info(f"Initialized weights: accuracy={st.session_state.weight_accuracy}, "
                           f"cost={st.session_state.weight_cost}, latency={st.session_state.weight_latency}")
                st.session_state.used_priority = extraction.get("priority", priority)
                st.session_state.detected_use_case = extraction.get("use_case", "chatbot_conversational")
                progress_bar.progress(100, text="Ready!")
            else:
                st.error("Could not extract business context. Please try rephrasing your input.")
                progress_bar.empty()
                
        except Exception as e:
            st.error(f"An error occurred during analysis. Please try again.")
            progress_bar.empty()
        finally:
            # Clean up progress bar after brief delay
            import time
            time.sleep(0.5)
            progress_container.empty()
    
    # Get the priority that was actually used
    used_priority = st.session_state.get("used_priority", priority)
    
    # Show extraction with approval if extraction exists but not approved
    if st.session_state.extraction_result and st.session_state.extraction_approved is None:
        render_extraction_with_approval(st.session_state.extraction_result, models_df)
        return
    
    # If editing, show edit form
    if st.session_state.extraction_approved == False:
        render_extraction_edit_form(st.session_state.extraction_result, models_df)
        return
    
    # If approved, show message to proceed to Technical Specifications tab
    if st.session_state.extraction_approved == True:
        render_extraction_result(st.session_state.extraction_result, used_priority)
        
        # Left-aligned completion banner and button
        col_btns, col_space = st.columns([2, 2])
        with col_btns:
            st.markdown("""
            <div style="background: #EE0000; color: white; padding: 0.75rem 1rem; border-radius: 8px; font-size: 1rem; margin-bottom: 0.75rem;">
                <strong>Step 1 Complete</strong> · Go to Technical Specification
            </div>
            """, unsafe_allow_html=True)
            if st.button("Next Tab →", key="next_tab_1", type="primary", use_container_width=True):
                import streamlit.components.v1 as components
                components.html("""
                <script>
                    const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs.length > 1) tabs[1].click();
                </script>
                """, height=0)


def render_technical_specs_tab(priority: str, models_df: pd.DataFrame):
    """Tab 2: Technical Specification (SLO targets and workload settings)."""
    used_priority = st.session_state.get("used_priority", priority)
    
    # Check if extraction is approved first
    if not st.session_state.extraction_approved:
        st.markdown("""
        <div style="background: #1a1a1a; color: white; padding: 1.5rem; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.2);">
            <strong style="font-size: 1.1rem;">Complete Step 1 First</strong><br>
            <span style="font-size: 0.95rem; color: rgba(255,255,255,0.8);">Go to the <strong>Define Use Case</strong> tab to describe your use case and approve the extraction.</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    final_extraction = st.session_state.edited_extraction or st.session_state.extraction_result or {}
    
    # Show SLO section
    render_slo_with_approval(final_extraction, used_priority, models_df)
    
    # If SLO approved, show navigation message
    if st.session_state.slo_approved == True:
        # Left-aligned completion banner and button
        col_btns2, col_space2 = st.columns([2, 2])
        with col_btns2:
            st.markdown("""
            <div style="background: #EE0000; color: white; padding: 0.75rem 1rem; border-radius: 8px; font-size: 1rem; margin-bottom: 0.75rem;">
                <strong>Step 2 Complete</strong> · Go to Recommendations
            </div>
            """, unsafe_allow_html=True)
            if st.button("Next Tab →", key="next_tab_2", type="primary", use_container_width=True):
                import streamlit.components.v1 as components
                components.html("""
                <script>
                    const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs.length > 2) tabs[2].click();
                </script>
                """, height=0)


def render_results_tab(priority: str, models_df: pd.DataFrame):
    """Tab 3: Results display - Best Model Recommendations."""
    used_priority = st.session_state.get("used_priority", priority)
    
    # Check if SLO is approved
    if not st.session_state.slo_approved:
        if not st.session_state.extraction_approved:
            st.markdown("""
            <div style="background: #1a1a1a; color: white; padding: 1.5rem; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.2);">
                <strong style="font-size: 1.1rem;">Complete Previous Steps First</strong><br>
                <span style="font-size: 0.95rem; color: rgba(255,255,255,0.8);">1. Go to <strong>Define Use Case</strong> tab to describe your use case<br>
                2. Then go to <strong>Technical Specification</strong> tab to set your SLO targets</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #1a1a1a; color: white; padding: 1.5rem; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.2);">
                <strong style="font-size: 1.1rem;">Complete Step 2 First</strong><br>
                <span style="font-size: 0.95rem; color: rgba(255,255,255,0.8);">Go to the <strong>Technical Specification</strong> tab to set your SLO targets and workload parameters.</span>
            </div>
            """, unsafe_allow_html=True)
        return
    
    final_extraction = st.session_state.edited_extraction or st.session_state.extraction_result or {}
    
    # ALWAYS regenerate recommendations to ensure fresh SLO filtering
    # Clear any cached results
    st.session_state.recommendation_result = None
    
    # Generate recommendations
    # FORCE clear cache to ensure fresh data every time
    st.session_state.recommendation_result = None
    st.session_state.pop('ranked_response', None)
    
    if True:  # Always regenerate
        # Get all specification values from session state (set in Tech Specs tab)
        # These are the EXACT values the user sees on the Technical Specification tab
        use_case = final_extraction.get("use_case", "chatbot_conversational")
        user_count = final_extraction.get("user_count", 1000)

        # Get SLO targets from session state (set by number_input widgets)
        ttft_target = st.session_state.get("custom_ttft") or st.session_state.get("input_ttft") or 500
        itl_target = st.session_state.get("custom_itl") or st.session_state.get("input_itl") or 50
        e2e_target = st.session_state.get("custom_e2e") or st.session_state.get("input_e2e") or 10000

        # Get QPS from session state - this is the value shown in the Expected RPS input
        qps_target = st.session_state.get("spec_expected_qps") or st.session_state.get("custom_qps") or 1

        # Get token config from session state (set by render_slo_cards)
        prompt_tokens = st.session_state.get("spec_prompt_tokens", 512)
        output_tokens = st.session_state.get("spec_output_tokens", 256)

        # Get percentile from session state (default to p95)
        # Note: Currently we use a single percentile for all metrics for the backend query
        # The UI allows per-metric percentiles but the backend uses one
        percentile = st.session_state.get("slo_percentile", "p95")

        business_context = {
            "use_case": use_case,
            "user_count": user_count,
            "priority": used_priority,
            "hardware_preference": final_extraction.get("hardware"),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "expected_qps": float(qps_target),
            "ttft_p95_target_ms": int(ttft_target),
            "itl_p95_target_ms": int(itl_target),
            "e2e_p95_target_ms": int(e2e_target),
            "percentile": percentile,
        }
        with st.spinner(f"Scoring {len(models_df)} models with MCDM..."):
            recommendation = get_enhanced_recommendation(business_context)
        if recommendation:
            st.session_state.recommendation_result = recommendation
        
    if st.session_state.recommendation_result:
            render_recommendation_result(st.session_state.recommendation_result, used_priority, final_extraction)


def render_extraction_result(extraction: dict, priority: str):
    """Render beautiful extraction results."""
    st.markdown('<div class="section-header">Extracted Business Context</div>', unsafe_allow_html=True)
    
    use_case = extraction.get("use_case", "unknown")
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware")
    
    st.markdown(f"""
    <div class="extraction-card">
        <div class="extraction-grid">
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Use Case</div>
                    <div class="extraction-value">{use_case.replace("_", " ").title() if use_case else "Unknown"}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Expected Users</div>
                    <div class="extraction-value">{user_count:,}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Hardware</div>
                    <div class="extraction-value">{hardware if hardware else "Any GPU"}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_extraction_with_approval(extraction: dict, models_df: pd.DataFrame):
    """Render extraction results with YES/NO approval buttons."""
    st.markdown('<div class="section-header">Extracted Business Context</div>', unsafe_allow_html=True)
    
    use_case = extraction.get("use_case", "unknown")
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware")
    
    st.markdown(f"""
    <div class="extraction-card" style="border: 2px solid #EE0000;">
        <div class="extraction-grid">
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Use Case</div>
                    <div class="extraction-value">{use_case.replace("_", " ").title() if use_case else "Unknown"}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Expected Users</div>
                    <div class="extraction-value">{user_count:,}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div>
                    <div class="extraction-label">Hardware</div>
                    <div class="extraction-value">{hardware if hardware else "Any GPU"}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Approval question
    st.markdown("""
    <div style="background: #000000; 
                padding: 1.25rem; border-radius: 1rem; margin: 1.5rem 0; text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.2);">
        <p style="color: white; font-size: 1.2rem; font-weight: 600; margin: 0;">
            Is this extraction correct?
        </p>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 0.5rem;">
            Verify the extracted business context before proceeding to recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Yes, Continue", type="primary", use_container_width=True, key="approve_extraction"):
            st.session_state.extraction_approved = True
            st.rerun()
    with col2:
        if st.button("No, Edit", use_container_width=True, key="edit_extraction"):
            st.session_state.extraction_approved = False
            st.rerun()
    with col3:
        if st.button("Start Over", use_container_width=True, key="restart"):
            st.session_state.extraction_result = None
            st.session_state.extraction_approved = None
            st.session_state.recommendation_result = None
            st.session_state.user_input = ""
            # Reset priority weights (complexity removed from UI)
            for key in ['accuracy_priority', 'cost_priority', 'latency_priority',
                       'weight_accuracy', 'weight_cost', 'weight_latency']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def render_extraction_edit_form(extraction: dict, models_df: pd.DataFrame):
    """Render editable form for extraction correction."""
    st.markdown('<div class="section-header">Edit Business Context</div>', unsafe_allow_html=True)
    
    # CSS to make form inputs visible
    st.markdown("""
    <style>
        /* Edit form - make ALL labels white */
        .stSelectbox label, .stNumberInput label {
            color: white !important;
            font-weight: 600 !important;
        }
        /* Selectbox styling */
        .stSelectbox > div > div {
            color: white !important;
        }
        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(102, 126, 234, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(102, 126, 234, 0.4) !important;
        }
        .stSelectbox [data-baseweb="select"] span {
            color: white !important;
        }
        /* Number input styling - match selectbox */
        .stNumberInput > div > div > input {
            color: white !important;
            background: rgba(102, 126, 234, 0.2) !important;
            border: 1px solid rgba(102, 126, 234, 0.4) !important;
        }
        .stNumberInput button {
            background: rgba(102, 126, 234, 0.3) !important;
            color: white !important;
            border: 1px solid rgba(102, 126, 234, 0.4) !important;
        }
        .stNumberInput button:hover {
            background: rgba(102, 126, 234, 0.5) !important;
        }
        /* Dropdown menu items */
        [data-baseweb="menu"] {
            background: #1a1a2e !important;
        }
        [data-baseweb="menu"] li {
            color: white !important;
        }
        [data-baseweb="menu"] li:hover {
            background: rgba(102, 126, 234, 0.3) !important;
        }
    </style>
    <div style="background: rgba(56, 239, 125, 0.1); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #38ef7d;">
        <p style="color: white; margin: 0;">Review and adjust the extracted values below, then click "Apply Changes" to continue.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use case dropdown
    use_cases = [
        "chatbot_conversational", "code_completion", "code_generation_detailed",
        "document_analysis_rag", "summarization_short", "long_document_summarization",
        "translation", "content_generation", "research_legal_analysis"
    ]
    use_case_labels = {
        "chatbot_conversational": "Chatbot / Conversational AI",
        "code_completion": "Code Completion (IDE autocomplete)",
        "code_generation_detailed": "Code Generation (full implementations)",
        "document_analysis_rag": "Document RAG / Q&A",
        "summarization_short": "Short Summarization (<10 pages)",
        "long_document_summarization": "Long Document Summarization (10+ pages)",
        "translation": "Translation",
        "content_generation": "Content Generation",
        "research_legal_analysis": "Research / Legal Analysis"
    }
    
    current_use_case = extraction.get("use_case", "chatbot_conversational")
    current_idx = use_cases.index(current_use_case) if current_use_case in use_cases else 0
    
    col1, col2 = st.columns(2)
    with col1:
        new_use_case = st.selectbox(
            "Use Case",
            use_cases,
            index=current_idx,
            format_func=lambda x: use_case_labels.get(x, x),
            key="edit_use_case"
        )
        
        new_user_count = st.number_input(
            "User Count",
            min_value=1,
            max_value=1000000,
            value=extraction.get("user_count", 1000),
            step=100,
            key="edit_user_count"
        )
    
    with col2:
        priorities = ["balanced", "low_latency", "cost_saving", "high_accuracy", "high_throughput"]
        priority_labels = {
            "balanced": "Balanced",
            "low_latency": "Low Latency",
            "cost_saving": "Cost Saving",
            "high_accuracy": "High Accuracy",
            "high_throughput": "High Throughput"
        }
        current_priority = extraction.get("priority", "balanced")
        priority_idx = priorities.index(current_priority) if current_priority in priorities else 0
        
        new_priority = st.selectbox(
            "Priority",
            priorities,
            index=priority_idx,
            format_func=lambda x: priority_labels.get(x, x),
            key="edit_priority"
        )
        
        hardware_options = ["Any GPU", "H100", "A100", "A10G", "L4", "T4"]
        current_hardware = extraction.get("hardware") or "Any GPU"
        hw_idx = hardware_options.index(current_hardware) if current_hardware in hardware_options else 0
        
        new_hardware = st.selectbox(
            "Hardware",
            hardware_options,
            index=hw_idx,
            key="edit_hardware"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Changes", type="primary", use_container_width=True, key="apply_edit"):
            st.session_state.edited_extraction = {
                "use_case": new_use_case,
                "user_count": new_user_count,
                "priority": new_priority,
                "hardware": new_hardware if new_hardware != "Any GPU" else None
            }
            st.session_state.used_priority = new_priority
            st.session_state.extraction_approved = True
            st.rerun()
    with col2:
        if st.button("🔙 Cancel", use_container_width=True, key="cancel_edit"):
            st.session_state.extraction_approved = None
            st.rerun()


def render_slo_with_approval(extraction: dict, priority: str, models_df: pd.DataFrame):
    """Render SLO section with approval to proceed to recommendations."""
    use_case = extraction.get("use_case", "chatbot_conversational")
    user_count = extraction.get("user_count", 1000)
    hardware = extraction.get("hardware")
    
    # SLO and Impact Cards - all 4 cards in one row
    render_slo_cards(use_case, user_count, priority, hardware)
    
    # ==========================================================================
    # VALIDATE SLO VALUES - Use case-specific ranges (same as sliders)
    # ==========================================================================
    research_data = load_research_slo_ranges()
    
    # Get use-case specific token config and benchmark ranges
    use_case_data = research_data.get('slo_ranges', {}).get(use_case, {}) if research_data else {}
    token_config = use_case_data.get('token_config', {'prompt': 512, 'output': 256})
    benchmark_ranges = get_benchmark_ranges_for_token_config(token_config['prompt'], token_config['output'])
    
    # Get selected percentile (same as sliders)
    percentile_key = st.session_state.get('slo_percentile', 'p95')
    
    # Get ranges for this specific percentile and use case
    ttft_min = int(benchmark_ranges.get(f'ttft_{percentile_key}_min', 15))
    ttft_max = int(benchmark_ranges.get(f'ttft_{percentile_key}_max', 270000))
    itl_min = int(benchmark_ranges.get(f'itl_{percentile_key}_min', 3))
    itl_max = int(benchmark_ranges.get(f'itl_{percentile_key}_max', 430))
    e2e_min = int(benchmark_ranges.get(f'e2e_{percentile_key}_min', 800))
    e2e_max = int(benchmark_ranges.get(f'e2e_{percentile_key}_max', 300000))
    
    # Get current SLO values from session state (these are from the sliders)
    current_ttft = int(st.session_state.get('edit_ttft', ttft_max))
    current_itl = int(st.session_state.get('edit_itl', itl_max))
    current_e2e = int(st.session_state.get('edit_e2e', e2e_max))
    
    # Validate - values from sliders should always be valid since sliders enforce bounds
    # But check anyway in case of edge cases
    validation_errors = []
    if current_ttft < ttft_min or current_ttft > ttft_max:
        validation_errors.append(f"TTFT must be between {ttft_min:,}-{ttft_max:,}ms")
    if current_itl < itl_min or current_itl > itl_max:
        validation_errors.append(f"ITL must be between {itl_min}-{itl_max}ms")
    if current_e2e < e2e_min or current_e2e > e2e_max:
        validation_errors.append(f"E2E must be between {e2e_min:,}-{e2e_max:,}ms")
    
    is_valid = len(validation_errors) == 0
    
    # Proceed button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not is_valid:
            # Show validation errors
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.5); 
                        border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem; text-align: center;">
                <div style="color: #ef4444; font-weight: 600; font-size: 0.9rem;">⚠️ Invalid SLO Values</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem; margin-top: 0.25rem;">
                    {' • '.join(validation_errors)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Button disabled if invalid
        if st.button("Generate Recommendations", type="primary", use_container_width=True, key="generate_recs", disabled=not is_valid):
            st.session_state.slo_approved = True
            st.rerun()


def render_recommendation_result(result: dict, priority: str, extraction: dict):
    """Render beautiful recommendation results with Top 5 table."""
    
    # === Use the ALREADY FETCHED data from result ===
    # The result was fetched with correct SLO filters from get_enhanced_recommendation
    # DO NOT call fetch_ranked_recommendations again - it would use wrong default values!
    
    # Gather data from extraction (handle None case)
    if extraction is None:
        extraction = {}
    use_case = extraction.get("use_case", "chatbot_conversational")
    user_count = extraction.get("user_count", 1000)

    # Use the result directly - it already has correct SLO-filtered data
    ranked_response = result
    
    if ranked_response:
        # NOTE: Table removed from main view - now only shown in "List Options" expandable section
        # render_ranked_recommendations(ranked_response)

        # Store ranked response for winner details and for the List Options section
        st.session_state.ranked_response = ranked_response
        
        # Get the Balanced winner for the Explore button
        balanced_recs = ranked_response.get("balanced", [])
        if balanced_recs:
            winner = balanced_recs[0]
            recommendations = balanced_recs
        else:
            # Fallback to any available recommendations
            for cat in ["best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
                if ranked_response.get(cat):
                    winner = ranked_response[cat][0]
                    recommendations = ranked_response[cat]
                    break
            else:
                st.warning("No recommendations found.")
                return
    else:
        st.warning("Could not fetch ranked recommendations from backend. Ensure the backend is running.")
        st.session_state.ranked_response = None
        recommendations = result.get("recommendations", [])
        if not recommendations:
            st.warning("No recommendations found. Try adjusting your requirements.")
            return
        winner = recommendations[0]
    
    # Store winner for explore dialog
    st.session_state.winner_recommendation = winner
    st.session_state.winner_priority = priority
    st.session_state.winner_extraction = extraction
    
    # Render the 4 "Best" cards with Explore button
    st.markdown("---")
    
    # Get all recommendations for the cards
    all_recs = []
    for cat in ["balanced", "best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
        cat_recs = st.session_state.ranked_response.get(cat, []) if st.session_state.ranked_response else []
        all_recs.extend(cat_recs)
    
    # Remove duplicates by model+hardware
    seen = set()
    unique_recs = []
    for rec in all_recs:
        model = rec.get("model_name", "")
        gpu_cfg = rec.get("gpu_config", {}) or {}
        hw = f"{gpu_cfg.get('gpu_type', 'H100')}x{gpu_cfg.get('gpu_count', 1)}"
        key = f"{model}_{hw}"
        if key not in seen:
            seen.add(key)
            unique_recs.append(rec)
    
    if unique_recs:
        render_top5_table(unique_recs, priority)
    
    # === LIST OPTIONS SECTION (inline expandable) ===
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        # Toggle button text based on expanded state
        is_expanded = st.session_state.get('show_options_list_expanded', False)
        button_text = "Hide Option List" if is_expanded else "List Options"
        if st.button(button_text, key="list_options_btn", use_container_width=True):
            # Toggle the expanded state
            st.session_state.show_options_list_expanded = not is_expanded
            st.rerun()

    # Render inline expanded content if expanded
    if st.session_state.get('show_options_list_expanded', False):
        render_options_list_inline()

    # === MODIFY SLOs & RE-RUN SECTION ===
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.1), rgba(166, 124, 0, 0.05)); 
                padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(255, 255, 255, 0.2); margin-top: 1rem; background: #000000;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="color: white; font-weight: 700; font-size: 1rem;">Want Different Results?</span>
        </div>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0;">
            Adjust SLO targets above to find models with different latency/performance trade-offs. 
            Stricter SLOs = fewer models, Relaxed SLOs = more options.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # Show buttons to go back or start new case
    col_back, col_new, col_spacer = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back to Technical Specs", key="back_to_slo", type="primary", use_container_width=True):
            # Reset dialog states to prevent popups
            st.session_state.show_category_dialog = False
            st.session_state.show_full_table_dialog = False
            st.session_state.show_winner_dialog = False
            st.session_state.show_options_list_expanded = False
            st.session_state.explore_category = None
            # Keep slo_approved as True - user can adjust SLOs and click "Generate Recommendations" again
            # The Recommendations tab will regenerate with new values when they return
            # Use JavaScript to switch to Technical Specification tab
            import streamlit.components.v1 as components
            components.html("""
            <script>
                const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs.length > 1) tabs[1].click();
            </script>
            """, height=0)
    
    with col_new:
        if st.button("New Case", key="new_case_btn", type="secondary", use_container_width=True):
            # Complete session state reset - start fresh from Tab 1
            keys_to_clear = [
                'user_input', 'extraction_result', 'recommendation_result',
                'extraction_approved', 'slo_approved', 'edited_extraction',
                'custom_ttft', 'custom_itl', 'custom_e2e', 'custom_qps', 'used_priority',
                'ranked_response', 'show_category_dialog', 'show_full_table_dialog',
                'show_winner_dialog', 'explore_category', 'detected_use_case',
                'top5_balanced', 'top5_accuracy', 'top5_latency', 'top5_cost', 'top5_simplest',
                'show_options_list_expanded'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # Set flag to switch to Tab 1 after rerun
            st.session_state.switch_to_tab1 = True
            st.rerun()
    
    # Check if we need to switch to Tab 1 (after New Case button)
    if st.session_state.get('switch_to_tab1', False):
        st.session_state.switch_to_tab1 = False
        import streamlit.components.v1 as components
        components.html("""
        <script>
            setTimeout(function() {
                const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs.length > 0) tabs[0].click();
            }, 100);
        </script>
        """, height=0)


def _render_winner_details(winner: dict, priority: str, extraction: dict):
    """Render detailed winner information inside the expander."""
    
    # Dark theme styling for popup dialog - including header
    st.markdown("""
    <style>
        /* Dialog container - force dark background everywhere */
        [data-testid="stDialog"],
        [data-testid="stDialog"] > div,
        [data-testid="stDialog"] > div > div,
        [data-testid="stDialog"] [data-testid="stVerticalBlock"],
        [data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"],
        div[data-modal-container="true"],
        div[data-modal-container="true"] > div {
            background: #0d1117 !important;
            background-color: #0d1117 !important;
        }
        
        /* Dialog header area */
        [data-testid="stDialog"] header,
        [data-testid="stDialog"] [data-testid="stModalHeader"],
        [role="dialog"] > div:first-child,
        [role="dialog"] header {
            background: #0d1117 !important;
            background-color: #0d1117 !important;
        }
        
        /* Dialog title */
        [data-testid="stDialog"] [data-testid="stModalHeader"] span,
        [role="dialog"] header span {
            color: #D4AF37 !important;
        }
        
        /* All text in dialog */
        [data-testid="stDialog"] .stMarkdown,
        [data-testid="stDialog"] p, 
        [data-testid="stDialog"] span,
        [data-testid="stDialog"] div,
        [data-testid="stDialog"] h1, 
        [data-testid="stDialog"] h2, 
        [data-testid="stDialog"] h3,
        [data-testid="stDialog"] label {
            color: #f0f6fc !important;
        }
        
        /* Close button */
        [data-testid="stDialog"] button[kind="secondary"],
        [data-testid="stDialog"] [data-testid="stBaseButton-secondary"] {
            background: rgba(212, 175, 55, 0.2) !important;
            color: #D4AF37 !important;
            border: 1px solid rgba(212, 175, 55, 0.4) !important;
        }
        
        /* Section headers in dialog */
        [data-testid="stDialog"] .section-header {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(6, 182, 212, 0.2)) !important;
            color: #f0f6fc !important;
        }
    </style>
            """, unsafe_allow_html=True)
            
    # Handle both backend format (scores) and UI format (score_breakdown)
    backend_scores = winner.get("scores", {}) or {}
    ui_breakdown = winner.get("score_breakdown", {}) or {}
    breakdown = {
        "quality_score": backend_scores.get("accuracy_score", ui_breakdown.get("quality_score", 0)),
        "latency_score": backend_scores.get("latency_score", ui_breakdown.get("latency_score", 0)),
        "cost_score": backend_scores.get("price_score", ui_breakdown.get("cost_score", 0)),
        "capacity_score": backend_scores.get("complexity_score", ui_breakdown.get("capacity_score", 0)),
    }
    
    # === 📋 FINAL RECOMMENDATION BOX (Schema-Aligned Clean Format) ===
    st.markdown('<div class="section-header" style="background: #1a1a1a; border: 1px solid rgba(255,255,255,0.2);">Final Recommendation</div>', unsafe_allow_html=True)
    
    # Extract data for clean display - handle both backend and UI formats
    model_name = winner.get("model_name", "Unknown Model")
    
    # Get hardware config - backend returns gpu_config object
    gpu_config = winner.get("gpu_config", {}) or {}
    hardware = gpu_config.get("gpu_type", winner.get("hardware", "H100"))
    hw_count = gpu_config.get("gpu_count", winner.get("hardware_count", 1))
    tp = gpu_config.get("tensor_parallel", 1)
    replicas = gpu_config.get("replicas", 1)
    
    # Get final score - backend uses balanced_score in scores
    backend_scores = winner.get("scores", {}) or {}
    final_score = backend_scores.get("balanced_score", winner.get("final_score", 0))
    quality_score = breakdown.get("quality_score", 0)
    
    # Get SLO data - backend returns predicted_* fields directly on winner
    # Try backend format first (predicted_ttft_p95_ms), then benchmark_slo format
    ttft_p95 = winner.get("predicted_ttft_p95_ms", 0)
    itl_p95 = winner.get("predicted_itl_p95_ms", 0)
    e2e_p95 = winner.get("predicted_e2e_p95_ms", 0)
    throughput_qps = winner.get("predicted_throughput_qps", 0)
    
    # Fallback to benchmark_slo if backend fields empty
    if not ttft_p95:
        benchmark_slo = winner.get("benchmark_slo", {})
        slo_actual = benchmark_slo.get("slo_actual", {}) if benchmark_slo else {}
        throughput_data = benchmark_slo.get("throughput", {}) if benchmark_slo else {}
        ttft_p95 = slo_actual.get("ttft_p95_ms", slo_actual.get("ttft_mean_ms", 0))
        itl_p95 = slo_actual.get("itl_p95_ms", slo_actual.get("itl_mean_ms", 0))
        e2e_p95 = slo_actual.get("e2e_p95_ms", slo_actual.get("e2e_mean_ms", 0))
        throughput_qps = throughput_data.get("tokens_per_sec", 0) / 100 if throughput_data.get("tokens_per_sec") else 0
    
    # Format for display
    ttft_display = f"{int(ttft_p95)}" if ttft_p95 and ttft_p95 > 0 else "—"
    itl_display = f"{int(itl_p95)}" if itl_p95 and itl_p95 > 0 else "—"
    e2e_display = f"{int(e2e_p95)}" if e2e_p95 and e2e_p95 > 0 else "—"
    max_rps = f"{throughput_qps:.1f}" if throughput_qps and throughput_qps > 0 else "—"
    
    # Schema-aligned recommendation box - Build HTML without comments
    # All models now have benchmark data (filtered to valid models only)
    benchmark_status = "<strong style='color: #10b981;'>Verified</strong> - Real benchmark data"
    priority_text = priority.replace('_', ' ').title()
    
    # Build hardware display text
    hw_display = f"{hw_count}x {hardware}"
    if tp > 1 and replicas > 1:
        hw_display += f" (TP={tp}, R={replicas})"
    
    rec_html = f'''<div style="background: #1a1a2e; padding: 2rem; border-radius: 1.25rem; border: 2px solid rgba(16, 185, 129, 0.4); margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid rgba(16, 185, 129, 0.3);">
        <span style="font-size: 2.5rem;"></span>
        <div>
            <h2 style="margin: 0; color: #10b981; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;">RECOMMENDATION</h2>
            <p style="margin: 0.25rem 0 0 0; color: #9ca3af; font-size: 0.85rem;">Based on {priority_text} optimization</p>
                    </div>
                    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
        <div style="display: flex; flex-direction: column; gap: 1.25rem;">
            <div style="background: #16213e; padding: 1rem; border-radius: 0.75rem; border-left: 4px solid #10b981;">
                <p style="margin: 0 0 0.5rem 0; color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Model</p>
                <p style="margin: 0; color: #f0f6fc; font-size: 1.25rem; font-weight: 700;">{model_name}</p>
                <p style="margin: 0.25rem 0 0 0; color: #9ca3af; font-size: 0.8rem;">Quality Score: <span style="color: #10b981; font-weight: 700;">{quality_score:.0f}%</span></p>
                    </div>
            <div style="background: #16213e; padding: 1rem; border-radius: 0.75rem; border-left: 4px solid #06b6d4;">
                <p style="margin: 0 0 0.5rem 0; color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">Hardware Configuration</p>
                <p style="margin: 0; color: #f0f6fc; font-size: 1.25rem; font-weight: 700;">{hw_display}</p>
                    </div>
                </div>
        <div style="background: #16213e; padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(139, 92, 246, 0.4);">
            <p style="margin: 0 0 1rem 0; color: #8b5cf6; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">Expected SLO (p95)</p>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <span style="color: #d1d5db; font-size: 0.85rem;">Max RPS</span>
                    <span style="color: #8b5cf6; font-weight: 800; font-size: 1.1rem;">{max_rps}</span>
            </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <span style="color: #d1d5db; font-size: 0.85rem;">TTFT (p95)</span>
                    <span style="color: #f59e0b; font-weight: 800; font-size: 1.1rem;">{ttft_display}<span style="font-size: 0.7rem; color: #9ca3af;"> ms</span></span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <span style="color: #d1d5db; font-size: 0.85rem;">ITL (p95)</span>
                    <span style="color: #ec4899; font-weight: 800; font-size: 1.1rem;">{itl_display}<span style="font-size: 0.7rem; color: #9ca3af;"> ms</span></span>
                    </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0;">
                    <span style="color: #d1d5db; font-size: 0.85rem;">E2E (p95)</span>
                    <span style="color: #06b6d4; font-weight: 800; font-size: 1.1rem;">{e2e_display}<span style="font-size: 0.7rem; color: #9ca3af;"> ms</span></span>
                    </div>
                    </div>
                    </div>
                </div>
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 2px solid rgba(212, 175, 55, 0.3); display: flex; justify-content: space-between; align-items: center;">
        <div style="color: #9ca3af; font-size: 0.85rem;"><strong style="color: #10b981;">Verified</strong> - Real benchmark data</div>
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="color: #d1d5db; font-size: 0.9rem;">Final Score:</span>
            <span style="background: linear-gradient(135deg, #D4AF37, #F4E4BA); color: #1a1a2e; padding: 0.5rem 1.25rem; border-radius: 0.5rem; font-weight: 900; font-size: 1.5rem; box-shadow: 0 4px 12px rgba(212, 175, 55, 0.3);">{final_score:.1f}</span>
            </div>
        </div>
</div>'''
    
    st.markdown(rec_html, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown('<div class="section-header" style="background: #1a1a1a; border: 1px solid rgba(255,255,255,0.2);">Winner Details: Score Breakdown</div>', unsafe_allow_html=True)
    
    # Add explanation for the score notation - golden styled
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.1), rgba(166, 124, 0, 0.05)); padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #D4AF37;">
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">
            <strong style="color: #D4AF37;">Score Format:</strong> <code style="background: rgba(212, 175, 55, 0.2); color: #F4E4BA; padding: 0.1rem 0.4rem; border-radius: 0.25rem;">87 → +17.4</code> means the model scored <strong>87/100</strong> in this category, contributing <strong style="color: #38ef7d;">+17.4 points</strong> to the final weighted score.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f'<h3 style="color: white; font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem; background: linear-gradient(135deg, #D4AF37, #F4E4BA, #D4AF37); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 2px 10px rgba(212, 175, 55, 0.3);">{winner.get("model_name", "Unknown")}</h3>', unsafe_allow_html=True)
        
        # Get weights based on priority
        priority_weights = {
            "balanced": {"accuracy": 0.30, "latency": 0.30, "cost": 0.25, "capacity": 0.15},
            "low_latency": {"accuracy": 0.15, "latency": 0.50, "cost": 0.15, "capacity": 0.20},
            "cost_saving": {"accuracy": 0.20, "latency": 0.15, "cost": 0.50, "capacity": 0.15},
            "high_accuracy": {"accuracy": 0.50, "latency": 0.20, "cost": 0.15, "capacity": 0.15},
            "high_throughput": {"accuracy": 0.15, "latency": 0.15, "cost": 0.15, "capacity": 0.55},
        }
        weights = priority_weights.get(priority, priority_weights["balanced"])
        
        # Calculate contributions
        q_score = breakdown.get("quality_score", 0)
        l_score = breakdown.get("latency_score", 0)
        c_score = breakdown.get("cost_score", 0)
        cap_score = breakdown.get("capacity_score", 0)
        
        q_contrib = q_score * weights["accuracy"]
        l_contrib = l_score * weights["latency"]
        c_contrib = c_score * weights["cost"]
        cap_contrib = cap_score * weights["capacity"]
        
        render_score_bar("Accuracy", "", q_score, "score-bar-accuracy", q_contrib)
        render_score_bar("Latency", "", l_score, "score-bar-latency", l_contrib)
        render_score_bar("Cost", "", c_score, "score-bar-cost", c_contrib)
        render_score_bar("Capacity", "", cap_score, "score-bar-capacity", cap_contrib)
    
    with col2:
        st.markdown('<h3 style="color: white;">Why This Model?</h3>', unsafe_allow_html=True)
        
        # Get use case from extraction context for use-case-specific summary
        use_case = extraction.get('use_case', 'chatbot_conversational')
        model_name = winner.get('model_name', 'Unknown')
        
        # Use-case specific model summaries
        use_case_context = {
            "code_completion": f"excels at real-time code suggestions with fast TTFT, ideal for IDE integrations where developer productivity depends on instant completions",
            "code_generation_detailed": f"provides detailed, well-documented code generation with strong reasoning capabilities, suitable for complex software engineering tasks",
            "chatbot_conversational": f"delivers natural, engaging conversations with consistent response quality, perfect for customer service and interactive applications",
            "translation": f"handles multilingual translation tasks with high accuracy, supporting document localization and cross-language communication",
            "content_generation": f"creates compelling marketing copy and creative content with style consistency and brand voice alignment",
            "summarization_short": f"efficiently compresses documents while preserving key insights, ideal for quick document digestion",
            "document_analysis_rag": f"excels at RAG-based document Q&A with accurate information retrieval and coherent answer synthesis",
            "long_document_summarization": f"handles long-context documents (10K+ tokens) with comprehensive summarization maintaining narrative flow",
            "research_legal_analysis": f"provides thorough analysis for legal and research documents requiring high accuracy and nuanced understanding",
        }
        
        # Model-specific traits
        model_traits = {
            "deepseek": "Known for exceptional coding and math performance at lower cost.",
            "qwen": "Top performer on reasoning benchmarks with strong multilingual support.",
            "llama": "Industry standard with excellent instruction-following.",
            "gemma": "Lightweight yet powerful, optimized for efficiency.",
            "mistral": "Efficient MoE architecture balancing speed and quality.",
            "doubao": "ByteDance model excelling in code tasks.",
            "kimi": "Moonshot AI model with strong reasoning.",
            "phi": "Microsoft's compact model for edge deployment.",
            "gpt-oss": "Open-source model benchmarked on diverse tasks.",
        }
        
        # Build use-case specific summary
        use_case_desc = use_case_context.get(use_case, "optimized for your specific task requirements")
        model_trait = ""
        for key, trait in model_traits.items():
            if key.lower() in model_name.lower():
                model_trait = trait
                break
        
        summary = f"<strong>{model_name}</strong> {use_case_desc}."
        if model_trait:
            summary += f"<br><br><em>{model_trait}</em>"
        
        # Display model summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(56, 239, 125, 0.1), rgba(102, 126, 234, 0.1)); padding: 1rem; border-radius: 0.75rem; margin-bottom: 1rem; border-left: 4px solid #38ef7d;">
            <p style="color: rgba(255,255,255,0.95); margin: 0; font-size: 0.95rem; line-height: 1.6;">{summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        pros = winner.get("pros", ["Top Quality", "Fast Responses"])
        cons = winner.get("cons", [])
        
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">Strengths:</p>', unsafe_allow_html=True)
        pros_html = '<div style="display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 1rem;">'
        for pro in pros:
            pros_html += f'<span class="tag tag-pro">{pro}</span>'
        pros_html += '</div>'
        st.markdown(pros_html, unsafe_allow_html=True)
        
        if cons:
            st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">Trade-offs:</p>', unsafe_allow_html=True)
            cons_html = '<div style="display: flex; flex-direction: column; gap: 0.4rem;">'
            for con in cons:
                cons_html += f'<span class="tag tag-con">{con}</span>'
            cons_html += '</div>'
            st.markdown(cons_html, unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(212, 175, 55, 0.3); margin: 1rem 0;">', unsafe_allow_html=True)
        st.markdown(f"""
        <p style="color: white;"><strong style="color: #D4AF37;">Final Score:</strong> <code style="background: linear-gradient(135deg, #D4AF37, #F4E4BA); color: #1a1a2e; padding: 0.35rem 0.75rem; border-radius: 0.25rem; font-weight: 700; font-size: 1.1rem;">{winner.get('final_score', 0):.1f}/100</code></p>
        <p style="color: rgba(212, 175, 55, 0.8); font-style: italic;">Based on {priority.replace('_', ' ').title()} priority weighting</p>
        """, unsafe_allow_html=True)
    
    # Display benchmark SLO data - use backend fields or benchmark_slo
    # Get predicted values from backend OR from benchmark_slo
    benchmark_slo = winner.get("benchmark_slo", {}) or {}
    gpu_config = winner.get("gpu_config", {}) or {}
    
    # Get SLO values - prioritize backend's predicted_* fields, fallback to benchmark_slo
    ttft_p95_val = winner.get("predicted_ttft_p95_ms") or benchmark_slo.get("slo_actual", {}).get("ttft_p95_ms", 0)
    itl_p95_val = winner.get("predicted_itl_p95_ms") or benchmark_slo.get("slo_actual", {}).get("itl_p95_ms", 0)
    e2e_p95_val = winner.get("predicted_e2e_p95_ms") or benchmark_slo.get("slo_actual", {}).get("e2e_p95_ms", 0)
    throughput_qps_val = winner.get("predicted_throughput_qps") or (benchmark_slo.get("throughput", {}).get("tokens_per_sec", 0) / 100 if benchmark_slo.get("throughput", {}).get("tokens_per_sec") else 0)
    
    # Get traffic profile from winner or result
    traffic_profile = winner.get("traffic_profile", {}) or {}
    prompt_tokens_val = traffic_profile.get("prompt_tokens", benchmark_slo.get("token_config", {}).get("prompt", 512))
    output_tokens_val = traffic_profile.get("output_tokens", benchmark_slo.get("token_config", {}).get("output", 256))
    
    # Get hardware info
    hw_type_val = gpu_config.get("gpu_type", benchmark_slo.get("hardware", "H100"))
    hw_count_val = gpu_config.get("gpu_count", benchmark_slo.get("hardware_count", 1))
    tp_val = gpu_config.get("tensor_parallel", 1)
    replicas_val = gpu_config.get("replicas", 1)
    
    # Show benchmark box if we have any SLO data
    if ttft_p95_val or itl_p95_val or e2e_p95_val:
        st.markdown("---")
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(16, 185, 129, 0.1)); border: 1px solid rgba(99, 102, 241, 0.2);">
            Real Benchmark SLOs (Actual Achievable Performance)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.05)); padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #6366f1;">
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">
                <strong style="color: #6366f1;">Benchmarks:</strong> Real measured values from vLLM simulation.
                Hardware: <strong style="color: #10b981;">{hw_count_val}x {hw_type_val}</strong> | 
                Token Config: <strong style="color: #f59e0b;">{prompt_tokens_val}→{output_tokens_val}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use the values we already extracted
        slo_actual = benchmark_slo.get("slo_actual", {})
        throughput = benchmark_slo.get("throughput", {})
        token_config = benchmark_slo.get("token_config", {})
        hardware = hw_type_val
        hw_count = hw_count_val
        
        col1, col2, col3 = st.columns(3)
        
        # Use our extracted values with fallback to slo_actual
        ttft_p95_show = ttft_p95_val or slo_actual.get('ttft_p95_ms', 0)
        itl_p95_show = itl_p95_val or slo_actual.get('itl_p95_ms', 0)
        e2e_p95_show = e2e_p95_val or slo_actual.get('e2e_p95_ms', 0)
        tps_show = throughput_qps_val * 100 if throughput_qps_val else throughput.get('tokens_per_sec', 0)
        
        with col1:
            st.markdown(f"""
            <div style="background: var(--bg-card); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(99, 102, 241, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">⏱️</span>
                    <span style="color: #6366f1; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">TTFT</span>
                </div>
                <div style="text-align: center;">
                    <p style="color: #10b981; font-weight: 800; font-size: 2rem; margin: 0;">{int(ttft_p95_show) if ttft_p95_show else 'N/A'}<span style="font-size: 1rem; color: rgba(255,255,255,0.5);">ms</span></p>
                    <p style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin: 0.25rem 0 0 0;">p95 latency</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: var(--bg-card); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1rem; color: white; font-weight: 700;">TTFT</span>
                    <span style="color: #10b981; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">ITL</span>
                </div>
                <div style="text-align: center;">
                    <p style="color: #f59e0b; font-weight: 800; font-size: 2rem; margin: 0;">{int(itl_p95_show) if itl_p95_show else 'N/A'}<span style="font-size: 1rem; color: rgba(255,255,255,0.5);">ms</span></p>
                    <p style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin: 0.25rem 0 0 0;">inter-token latency</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: var(--bg-card); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(245, 158, 11, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">🏁</span>
                    <span style="color: #f59e0b; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">E2E</span>
                </div>
                <div style="text-align: center;">
                    <p style="color: #ec4899; font-weight: 800; font-size: 2rem; margin: 0;">{int(e2e_p95_show) if e2e_p95_show else 'N/A'}<span style="font-size: 1rem; color: rgba(255,255,255,0.5);">ms</span></p>
                    <p style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin: 0.25rem 0 0 0;">end-to-end</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Throughput and Config row
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(99, 102, 241, 0.05)); padding: 1rem; border-radius: 0.75rem; text-align: center; border: 1px solid rgba(139, 92, 246, 0.2);">
                <span style="font-size: 1.25rem;">🚀</span>
                <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Throughput</p>
                <p style="color: #8b5cf6; font-weight: 800; font-size: 1.5rem; margin: 0;">{int(tps_show) if tps_show else 'N/A'} <span style="font-size: 0.8rem;">tok/s</span></p>
            </div>
            <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(14, 165, 233, 0.05)); padding: 1rem; border-radius: 0.75rem; text-align: center; border: 1px solid rgba(6, 182, 212, 0.2);">
                <span style="font-size: 1rem; color: white; font-weight: 700;">HW</span>
                <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Hardware</p>
                <p style="color: #06b6d4; font-weight: 800; font-size: 1.25rem; margin: 0;">{hw_count_val}x {hw_type_val}</p>
            </div>
            <div style="background: linear-gradient(135deg, rgba(244, 114, 182, 0.1), rgba(236, 72, 153, 0.05)); padding: 1rem; border-radius: 0.75rem; text-align: center; border: 1px solid rgba(244, 114, 182, 0.2);">
                <span style="font-size: 1rem; color: white; font-weight: 700;">Tokens</span>
                <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Token Config</p>
                <p style="color: #f472b6; font-weight: 700; font-size: 1rem; margin: 0;">{prompt_tokens_val} → {output_tokens_val}</p>
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(16, 185, 129, 0.08); border-radius: 0.5rem; border: 1px solid rgba(16, 185, 129, 0.2);">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem; text-align: center;">
                <strong style="color: #10b981;">Data Source:</strong> vLLM Simulation Benchmarks | 
                <strong style="color: #6366f1;">Model:</strong> {winner.get('model_name', 'Unknown')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    # All recommendations now come from valid models with both AA + benchmark data
    # No need to show "No benchmark data" warning


if __name__ == "__main__":
    main()
