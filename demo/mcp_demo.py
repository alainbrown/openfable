"""MCP agent demo — an LLM uses OpenFable as a tool to ingest and query documents."""

import asyncio
import logging
import os

from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

# Suppress noisy library logs so the demo output is clean.
logging.getLogger("mcp_use").setLevel(logging.CRITICAL)
logging.getLogger("mcp_use.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

OPENFABLE_URL = os.environ.get("OPENFABLE_URL", "http://localhost:8000")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ── Document 1: Q1 primary research report ────────────────────────────────────
# Deep nested structure; the measurement-conditions detail in §2.2.1 is the
# "buried fact" the low-budget drill-down query will target.
DOCUMENT_Q1 = """\
# Project Chimera: Q1 2026 Research Report

## Section 1: Executive Summary
Project Chimera is a multi-year initiative to develop bio-digital interfaces capable
of seamless data transfer between biological neural networks and silicon-based storage.
Q1 marked the successful transition from ex-vivo to in-vivo murine trials.

## Section 2: Neural Mapping Protocols

### 2.1 Synthetic Synapse Integration
Graphene-oxide scaffolding introduced in Q4 2025 has raised the synthetic-synapse
integration rate by 40% over the previous polymer baseline. Murine subjects show
stable signal fidelity after 30-day implantation with no measurable inflammatory response.

### 2.2 Latency and Throughput

#### 2.2.1 Measurement Methodology
All latency figures in this report were obtained under isoflurane anaesthesia
(1.5–2.0% v/v) using a stereotaxic frame. Measurements were NOT taken from
freely-moving or awake subjects. This constraint was necessary to eliminate
motion artefacts during the current calibration phase.

#### 2.2.2 Current Performance Figures
Under the controlled anaesthesia protocol, round-trip data-retrieval latency from
the hippocampus-equivalent interface is **12 ms** (mean across n=24 subjects,
95 % CI: 10.8–13.2 ms). Throughput is 4.2 Mbit/s sustained.

#### 2.2.3 Q4 Target
The engineering team projects a reduction to **under 5 ms** by end of Q4 2026,
contingent on the Phase 3 graphene deposition upgrade and a move to awake-subject
calibration protocols.

### 2.3 Bandwidth Roadmap
Phase 3 targets 10 Mbit/s sustained throughput. Achieving this requires the custom
ASIC currently in tape-out at TSMC (expected delivery: September 2026).

## Section 3: Ethical Considerations and Safety

### 3.1 Bio-Safety Standards
All experiments comply with the 2025 International Bio-Ethics Accords. Any permanent
alteration to a biological host requires sign-off from the Secondary Review Committee
(SRC). The SRC convenes bi-weekly.

### 3.2 Data Privacy
Raw neural signatures are encrypted at point-of-origin using AES-256-GCM with
quantum-resistant key encapsulation (CRYSTALS-Kyber-1024). The "Ghost Protocol"
pipeline ensures no unencrypted signatures reach persistent storage.

## Section 4: Budget and Resource Allocation

### 4.1 Spend to Date
The project has consumed **65 %** of its $50 M allocation ($32.5 M).
The single largest line item was the acquisition of High-Fidelity Neural Scanners
in January 2026 ($8.1 M, 25 % over the original equipment budget).

### 4.2 Projected Overrun Risk
Current burn rate projects total spend at $53–55 M if Phase 3 hardware arrives
on schedule. A further delay of 8+ weeks would push total spend above $57 M.
The programme office has flagged this as a medium risk requiring CFO review.
"""

# ── Document 2: Independent Risk & Safety Audit (Q2 2026) ────────────────────
# Partially contradicts the Q1 report: awake-subject latency is 31 ms, not 12 ms,
# and the SRC has a backlog that threatens the bio-ethics compliance posture.
DOCUMENT_AUDIT = """\
# Project Chimera: Independent Risk & Safety Audit — Q2 2026

Prepared by: External Assurance Group (EAG)
Commissioned by: Project Chimera Steering Committee

## Section 1: Audit Scope and Findings Overview
EAG conducted a six-week review of Project Chimera covering experimental methodology,
regulatory compliance, and financial risk. Three significant findings are documented below.

## Section 2: Latency Claims — Measurement Conditions Discrepancy

### 2.1 Anaesthesia vs. Awake-Subject Results
The Q1 2026 report cites a 12 ms round-trip latency. EAG replicated the measurement
protocol and confirmed the 12 ms figure under anaesthesia. However, EAG also conducted
trials on awake, freely-moving subjects using the same hardware.

**Awake-subject latency: 31 ms** (mean, n=18, 95 % CI: 28.4–33.6 ms).

The 2.6× gap is attributed to motion artefacts and the absence of the stereotaxic
frame used in anaesthetised trials. The Q1 report does not mention awake-subject
results and does not flag the anaesthesia dependency in its executive summary.

### 2.2 Q4 Target Feasibility
EAG considers the sub-5 ms target for Q4 2026 **highly aggressive** given the
awake-subject baseline of 31 ms. Achieving it would require both the Phase 3 ASIC
and a novel motion-compensation algorithm not yet in development. EAG recommends
revising the Q4 target to 15 ms and publishing a corrected latency baseline
that includes awake-subject conditions.

## Section 3: Bio-Ethics Compliance

### 3.1 Secondary Review Committee Backlog
As of 1 April 2026, the SRC has **47 pending experiment-approval requests**,
representing 11 weeks of backlog at its current bi-weekly cadence. Three Phase 3
experiments scheduled for May 2026 cannot legally proceed without SRC sign-off.
EAG recommends increasing SRC meeting frequency to weekly until the backlog clears.

### 3.2 Ghost Protocol Audit Trail
EAG found no gaps in the Ghost Protocol encryption pipeline. Neural data handling
is compliant with the 2025 Bio-Ethics Accords.

## Section 4: Financial Risk Assessment

### 4.1 Revised Overrun Projection
Incorporating the ASIC delivery risk and potential SRC-caused schedule slippage,
EAG projects total programme spend of **$56–60 M** against the $50 M budget —
a 12–20 % overrun. This is materially higher than the programme office's own
estimate of $53–55 M and warrants immediate CFO and board review.
"""


async def main():
    config = {"mcpServers": {"openfable": {"url": f"{OPENFABLE_URL}/v1/mcp/sse"}}}
    client = MCPClient.from_dict(config)

    llm = ChatOpenAI(model="gpt-5.4-nano")
    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    try:
        # ── Step 1: Ingest Document 1 ─────────────────────────────────────────
        print("--- Step 1: Ingest Project Chimera Q1 Report ---\n")
        result = await agent.run(
            f"Ingest the following document using the create document tool:\n\n{DOCUMENT_Q1}",
        )
        print(f"Agent: {result}\n")

        # ── Step 2: Ingest Document 2 ─────────────────────────────────────────
        print("--- Step 2: Ingest Independent Risk & Safety Audit ---\n")
        result = await agent.run(
            f"Ingest the following document using the create document tool:\n\n{DOCUMENT_AUDIT}",
        )
        print(f"Agent: {result}\n")

        # ── Step 3: Cross-document synthesis ──────────────────────────────────
        # Requires retrieving matching sections from both documents and reconciling
        # contradictory claims — impossible without real multi-document retrieval.
        print("--- Step 3: Cross-Document Synthesis (token_budget=4000) ---\n")
        print("Query: Compare the Q1 latency claims with the audit findings.")
        print("       Are there contradictions? What do they mean for the Q4 target?\n")
        result = await agent.run(
            "Using the query tool with token_budget=4000, compare Project Chimera's "
            "Q1 2026 latency figures with the findings in the independent risk audit. "
            "Are there any contradictions? What are the implications for the Q4 target?",
        )
        print(f"Agent: {result}\n")

        # ── Step 4: Surgical drill-down — low budget ───────────────────────────
        # The answer is buried in §2.2.1 of the Q1 doc, three heading levels deep.
        # A low budget should surface just that leaf node and nothing else.
        print("--- Step 4: Surgical Drill-Down (token_budget=150) ---\n")
        print("Query: What were the exact measurement conditions for the 12 ms figure?\n")
        result = await agent.run(
            "Using the query tool with token_budget=150, what were the exact "
            "measurement conditions used to obtain the 12 ms latency figure in "
            "the Q1 report?",
        )
        print(f"Agent: {result}\n")

        # ── Step 5: Same question, higher budget ───────────────────────────────
        # More context flows in — surrounding sections reveal why the measurement
        # conditions matter (the Q4 target depends on awake-subject calibration).
        print("--- Step 5: Same Question, Higher Budget (token_budget=2000) ---\n")
        print("Query: Same question — but now with enough budget to see why it matters.\n")
        result = await agent.run(
            "Using the query tool with token_budget=2000, what were the exact "
            "measurement conditions used to obtain the 12 ms latency figure in "
            "the Q1 report, and what broader context surrounds that methodology?",
        )
        print(f"Agent: {result}\n")

        # ── Step 6: Question the corpus is silent on ───────────────────────────
        # Demonstrates the system doesn't hallucinate — it returns "I don't know"
        # rather than inventing an answer.
        print("--- Step 6: Out-of-Corpus Question (token_budget=2000) ---\n")
        print("Query: Something plausible but not in either document.\n")
        result = await agent.run(
            "Using the query tool with token_budget=2000, what animal species other "
            "than mice have been used in Project Chimera trials, and what were their results?",
        )
        print(f"Agent: {result}\n")

    finally:
        if client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(main())
