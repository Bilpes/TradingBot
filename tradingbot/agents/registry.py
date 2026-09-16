"""Sector-specific agent profiles with customizable indicators.

Different sectors behave differently, and using one global parameter set for all
of them is the kind of simplification that looks tidy and costs money. Banking
names trend hard and punish RSI contrarianism; FMCG is defensive and mean
reverts; metals are high-beta and need volume confirmation; telecom is
news-driven.

A profile is pure data -- a list of ``AgentSpec`` plus a ``ConsensusConfig`` --
so it can be overridden from a JSON file without touching code:

    python -m tradingbot.cli sectors-config --out profiles.json
    # edit profiles.json
    python -m tradingbot.cli backtest --profiles profiles.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Type

from tradingbot.agents.base import Agent
from tradingbot.agents.macd import MACDAgent
from tradingbot.agents.meanrev import MeanReversionAgent
from tradingbot.agents.momentum import CrossSectionalMomentumAgent
from tradingbot.agents.moving_average import MovingAverageAgent
from tradingbot.agents.rsi import RSIAgent
from tradingbot.agents.sector import SectorRotationAgent
from tradingbot.agents.sentiment import SentimentAgent
from tradingbot.agents.trend import TrendAgent
from tradingbot.agents.volatility import VolatilityRegimeAgent
from tradingbot.agents.volume import VolumeSpikeAgent
from tradingbot.core.consensus import ConsensusConfig
from tradingbot.markets.nse import nse_sector_of

AGENT_TYPES: Dict[str, Type[Agent]] = {
    "rsi": RSIAgent,
    "macd": MACDAgent,
    "moving_average": MovingAverageAgent,
    "volume_spike": VolumeSpikeAgent,
    "momentum": CrossSectionalMomentumAgent,
    "trend": TrendAgent,
    "mean_reversion": MeanReversionAgent,
    "sector_rotation": SectorRotationAgent,
    "vol_regime": VolatilityRegimeAgent,
    "sentiment": SentimentAgent,
}


@dataclass(frozen=True)
class AgentSpec:
    agent: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.agent not in AGENT_TYPES:
            raise ValueError(
                f"unknown agent {self.agent!r}; known: {sorted(AGENT_TYPES)}"
            )

    def build(self) -> Agent:
        instance = AGENT_TYPES[self.agent](**self.params)
        instance.weight = self.weight
        return instance


@dataclass(frozen=True)
class SectorProfile:
    sector: str
    agents: List[AgentSpec]
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    note: str = ""

    def build_agents(self, sector_map: Callable[[str], str] | None = None) -> List[Agent]:
        """Instantiate the profile's agents.

        ``sector_map`` is injected into any agent that accepts one. It cannot
        live in the JSON profile (callables are not serialisable), and the
        default belongs to the US universe -- running an NSE book without
        overriding it silently classifies every symbol as "Unknown", which
        collapses the sector cross-section to a single column.
        """
        agents = []
        for spec in self.agents:
            agent = spec.build()
            if sector_map is not None and hasattr(agent, "sector_map"):
                agent.sector_map = sector_map
            agents.append(agent)
        return agents

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector": self.sector,
            "note": self.note,
            "consensus": asdict(self.consensus),
            "agents": [{"agent": a.agent, "weight": a.weight, "params": a.params}
                       for a in self.agents],
        }


def _spec(agent: str, weight: float, **params: Any) -> AgentSpec:
    return AgentSpec(agent=agent, weight=weight, params=dict(params))


# Baseline every sector starts from. Sector rotation and the volatility gate are
# present everywhere: rotation is the cross-sector view, and the gate is what
# keeps the whole system from sizing up into a stressed tape.
_BASE: List[AgentSpec] = [
    _spec("sector_rotation", 1.1),
    _spec("vol_regime", 1.0),
]


_SECTOR_NOTES = {
    "Banking": "Rate-sensitive; trends persist, RSI contrarianism is dangerous.",
    "Financial Services": "NBFCs and insurers; credit-cycle driven, momentum-led.",
    "IT": "USD/INR and NASDAQ-linked; momentum and trend dominate.",
    "Auto": "Cyclical; medium-length trends, MACD confirmation.",
    "Pharma": "Defensive; mean reverts more often than it trends.",
    "FMCG": "Low-beta defensive; fade extremes, distrust momentum.",
    "Energy": "Commodity-driven; needs volume confirmation on breakouts.",
    "Metal": "High-beta cyclical; wide stops, volume matters most.",
    "Realty": "High-beta, rate-sensitive; fade extremes with volume.",
    "Infra": "Slow, persistent trends; moving averages lead.",
    "Consumer Durables": "Steady trends with mild mean reversion.",
    "Telecom": "News/event-driven; sentiment and volume carry more weight.",
}


def _profile(sector: str, *specs: AgentSpec) -> SectorProfile:
    return SectorProfile(sector=sector, agents=[*_BASE, *specs],
                         note=_SECTOR_NOTES.get(sector, ""))


DEFAULT_PROFILES: Dict[str, SectorProfile] = {
    p.sector: p
    for p in [
        _profile("Banking",
                 _spec("macd", 1.2), _spec("moving_average", 1.1),
                 _spec("momentum", 1.0, lookback=126, skip=21),
                 _spec("rsi", 0.5), _spec("volume_spike", 0.6)),
        _profile("IT",
                 _spec("momentum", 1.3, lookback=126, skip=21), _spec("moving_average", 1.2),
                 _spec("macd", 1.0), _spec("trend", 0.9), _spec("rsi", 0.4)),
        _profile("Auto",
                 _spec("macd", 1.2), _spec("moving_average", 1.1), _spec("momentum", 0.9),
                 _spec("volume_spike", 0.7), _spec("rsi", 0.5)),
        _profile("Pharma",
                 _spec("rsi", 1.2, window=14, overbought=70, oversold=30),
                 _spec("mean_reversion", 1.1), _spec("macd", 0.8),
                 _spec("moving_average", 0.8), _spec("sentiment", 0.7)),
        _profile("FMCG",
                 _spec("mean_reversion", 1.3), _spec("rsi", 1.2),
                 _spec("moving_average", 0.9), _spec("macd", 0.6),
                 _spec("volume_spike", 0.4)),
        _profile("Energy",
                 _spec("momentum", 1.2), _spec("volume_spike", 1.1),
                 _spec("macd", 1.0), _spec("trend", 0.9), _spec("rsi", 0.5)),
        _profile("Metal",
                 _spec("momentum", 1.3), _spec("volume_spike", 1.2),
                 _spec("macd", 1.0), _spec("trend", 0.9), _spec("rsi", 0.4)),
        _profile("Realty",
                 _spec("rsi", 1.2), _spec("volume_spike", 1.1),
                 _spec("mean_reversion", 0.9), _spec("macd", 0.8),
                 _spec("moving_average", 0.7)),
        _profile("Infra",
                 _spec("moving_average", 1.3), _spec("momentum", 1.1),
                 _spec("trend", 1.0), _spec("macd", 0.8), _spec("rsi", 0.4)),
        _profile("Financial Services",
                 _spec("momentum", 1.2, lookback=126, skip=21), _spec("macd", 1.1),
                 _spec("moving_average", 1.0), _spec("trend", 0.9),
                 _spec("volume_spike", 0.7), _spec("rsi", 0.4)),
        _profile("Consumer Durables",
                 _spec("moving_average", 1.1), _spec("rsi", 1.0),
                 _spec("macd", 0.9), _spec("mean_reversion", 0.8),
                 _spec("volume_spike", 0.5)),
        _profile("Telecom",
                 _spec("sentiment", 1.2), _spec("volume_spike", 1.1),
                 _spec("macd", 0.9), _spec("momentum", 0.8), _spec("rsi", 0.7)),
    ]
}


def profile_for(sector: str) -> SectorProfile:
    """Profile for a sector, falling back to a balanced default."""
    if sector in DEFAULT_PROFILES:
        return DEFAULT_PROFILES[sector]
    return _profile(sector,
                    _spec("macd", 1.0), _spec("moving_average", 1.0),
                    _spec("rsi", 0.8), _spec("momentum", 0.8),
                    _spec("volume_spike", 0.6))


def profile_for_symbol(symbol: str) -> SectorProfile:
    return profile_for(nse_sector_of(symbol))


def load_profiles(path: str | Path) -> Dict[str, SectorProfile]:
    """Load sector profiles from JSON, validating every agent name and param."""
    raw = json.loads(Path(path).read_text())
    out: Dict[str, SectorProfile] = {}
    for sector, body in raw.items():
        agents = [
            AgentSpec(agent=a["agent"], weight=float(a.get("weight", 1.0)),
                      params=dict(a.get("params", {})))
            for a in body.get("agents", [])
        ]
        if not agents:
            raise ValueError(f"profile {sector!r} defines no agents")
        cfg = ConsensusConfig(**body.get("consensus", {}))
        out[sector] = SectorProfile(sector=sector, agents=agents, consensus=cfg,
                                    note=body.get("note", ""))
    return out


def dump_profiles(path: str | Path, profiles: Dict[str, SectorProfile] | None = None) -> Path:
    fp = Path(path)
    fp.write_text(json.dumps(
        {s: p.to_dict() for s, p in (profiles or DEFAULT_PROFILES).items()},
        indent=2, sort_keys=True,
    ) + "\n")
    return fp
