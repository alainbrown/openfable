"""E2E test: LLM agent connects to OpenFable's MCP server.

Uses mcp-use to wire an LLM directly to the MCP SSE endpoint.
The agent discovers tools, ingests synthetic documents, and answers
questions — exactly how Claude Desktop or any MCP client would work.

Documents contain fictional facts that prove retrieval, not memorization.
"""

import asyncio
import os

import pytest
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

API_URL = os.environ.get("OPENFABLE_API_URL", "http://localhost:8000")
MCP_SSE_URL = API_URL + "/v1/mcp/sse"
OLLAMA_URL = os.environ.get("OPENFABLE_LITELLM_BASE_URL", "http://ollama:11434")

# ---------------------------------------------------------------------------
# Synthetic documents
# ---------------------------------------------------------------------------

DOCUMENT_XENOBIOLOGY = """\
# Xenobiology Research at Haldane Station

In 2031, Dr. Yuki Tanaka discovered the Cephalon organism in the subsurface
ocean of Europa during the Haldane Station expedition. The organism is the first
known example of silicate-based metabolism outside Earth.

## Morphology

Cephalon colonies consist of approximately 10^8 individual cells arranged in
a radial lattice. Each cell measures 0.4 micrometers in diameter. The colony
communicates via bioluminescent pulses emitted at a wavelength of 340 nanometers.

## Metabolic Pathway

Unlike terrestrial life, Cephalon reduces dissolved silicates using hydrogen
sulfide as an electron donor. The net reaction produces crystalline silicon
dioxide and water. Dr. Tanaka named this pathway "lithotrophic silicification"
and published the findings in the journal Astrobiology Frontiers, volume 12.
"""

DOCUMENT_MATERIALS = """\
# Veridium: A Next-Generation Structural Alloy

Veridium is a ruthenium-osmium lattice alloy developed in 2028 by Dr. Emil
Volkov at Kazan Federal University. It was designed for extreme-environment
aerospace applications.

## Mechanical Properties

Veridium exhibits a tensile strength of 4,200 MPa — roughly three times
that of conventional titanium alloys. Its melting point is 3,847 degrees
Celsius. The alloy maintains structural integrity under radiation doses
exceeding 500 Gray, making it suitable for deep-space hull construction.

## Applications

The first large-scale deployment of Veridium was the Tiangong-4 space station
hull, manufactured in 2030. The hull panels are 2.3 millimeters thick and
weigh 40% less than equivalent steel panels. Veridium is produced exclusively
at the Volkov Advanced Materials Laboratory in Kazan, Russia.
"""

DOCUMENT_TREATY = """\
# The Treaty of Meridian

The Treaty of Meridian was signed on March 15, 2033 in Reykjavik, Iceland
by representatives of 14 Arctic and near-Arctic nations. The treaty established
the Arctic Sovereignty Framework, a binding international agreement governing
resource extraction, shipping lanes, and environmental protection north of
the 70th parallel.

## Negotiation

Negotiations were led by Ambassador Clara Nystrom of Sweden over a period of
18 months. The key sticking point was subsea mining rights in the Lomonosov
Ridge region, which was resolved by creating a shared-revenue zone administered
by the newly formed Polar Commons Authority.

## Ratification

By December 2035, 127 nations had ratified the treaty. The Polar Commons
Authority is headquartered in Tromso, Norway and has an annual operating
budget of 840 million euros. Its first chair was Dr. Anika Bergsdottir
of Iceland.
"""

# ---------------------------------------------------------------------------
# Agent helper
# ---------------------------------------------------------------------------


def run_agent(prompt, max_steps=10):
    """Create an MCP agent, connect to OpenFable, run a prompt."""
    async def _run():
        config = {
            "mcpServers": {
                "openfable": {"url": MCP_SSE_URL}
            }
        }
        client = MCPClient.from_dict(config)
        llm = ChatOpenAI(
            model="qwen3:8b",
            base_url=f"{OLLAMA_URL}/v1",
            api_key="ollama",
        )
        agent = MCPAgent(llm=llm, client=client, max_steps=max_steps)
        try:
            return await agent.run(prompt, max_steps=max_steps)
        finally:
            if client.sessions:
                await client.close_all_sessions()

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMCPAgent:
    """LLM agent ingests and queries documents via MCP."""

    def test_00_clean_database(self, clean_db):
        pass

    def test_01_ingest_xenobiology(self):
        result = run_agent(
            f"Ingest the following document using the create document tool:\n\n{DOCUMENT_XENOBIOLOGY}",
        )
        assert result is not None

    def test_02_ingest_materials(self):
        result = run_agent(
            f"Ingest the following document using the create document tool:\n\n{DOCUMENT_MATERIALS}",
        )
        assert result is not None

    def test_03_ingest_treaty(self):
        result = run_agent(
            f"Ingest the following document using the create document tool:\n\n{DOCUMENT_TREATY}",
        )
        assert result is not None

    def test_04_query_xenobiology(self):
        result = run_agent(
            "Search the indexed documents: Who discovered the Cephalon organism and on which moon?",
        )
        lower = result.lower()
        assert "tanaka" in lower, f"Expected 'Tanaka' in: {result}"
        assert "europa" in lower, f"Expected 'Europa' in: {result}"

    def test_05_query_materials(self):
        result = run_agent(
            "Search the indexed documents: What is the tensile strength of Veridium and who developed it?",
        )
        lower = result.lower()
        assert "4,200" in result or "4200" in result, f"Expected '4200' in: {result}"
        assert "volkov" in lower, f"Expected 'Volkov' in: {result}"

    def test_06_query_treaty(self):
        result = run_agent(
            "Search the indexed documents: Where was the Treaty of Meridian signed and who led negotiations?",
        )
        lower = result.lower()
        assert "reykjavik" in lower, f"Expected 'Reykjavik' in: {result}"
        assert "nystrom" in lower or "nyström" in lower, f"Expected 'Nystrom' in: {result}"
