"""
Generate a large synthetic-yet-realistic dataset covering common news queries.
"""

from __future__ import annotations

import random
from itertools import product, cycle
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "news.csv"
random.seed(42)

REAL_CLAUSES = [
    "according to the official bulletin.",
    "the regulator confirmed in today's briefing.",
    "as recorded in the public minutes.",
    "the ministry detailed in its release.",
    "as reported by accredited correspondents.",
]

FAKE_CLAUSES = [
    "according to anonymous viral posts.",
    "a rumour circulating on fringe forums claimed.",
    "the chain message insisted without evidence.",
    "the hoax blog concluded.",
    "according to an unverifiable social media screenshot.",
]


def expand_templates(template: str, fields: Dict[str, List[str]]) -> Iterable[str]:
    keys = list(fields.keys())
    for combo in product(*(fields[key] for key in keys)):
        values = dict(zip(keys, combo))
        yield template.format(**values)


def build_real_samples() -> List[str]:
    builders = [
        {
            "template": (
                "{agency} {action} {initiative} in {region} to {goal} by {timeline}"
            ),
            "fields": {
                "agency": [
                    "The World Health Organization",
                    "The European Central Bank",
                    "The Indian Space Research Organisation",
                    "The African Union Commission",
                ],
                "action": [
                    "announced",
                    "approved",
                    "confirmed",
                    "expanded",
                ],
                "initiative": [
                    "a coordinated vaccination drive",
                    "a resilience funding program",
                    "a satellite monitoring mission",
                    "a trade facilitation framework",
                ],
                "region": [
                    "South Asia",
                    "the Eurozone",
                    "East Africa",
                    "Latin America",
                ],
                "goal": [
                    "support rural clinics",
                    "stabilise commodity markets",
                    "improve disaster forecasting",
                    "modernise cross-border logistics",
                ],
                "timeline": [
                    "the end of 2025",
                    "Q3 2026",
                    "the upcoming fiscal year",
                    "early next quarter",
                ],
            },
        },
        {
            "template": (
                "{authority} reported that {metric} {change} across {sector} during {period}"
            ),
            "fields": {
                "authority": [
                    "The U.S. Bureau of Labor Statistics",
                    "Statistics Canada",
                    "Eurostat",
                    "The Reserve Bank of India",
                ],
                "metric": [
                    "employment levels",
                    "consumer confidence",
                    "manufacturing output",
                    "inflation-adjusted wages",
                ],
                "change": [
                    "rose steadily",
                    "remained stable",
                    "declined slightly",
                    "saw record growth",
                ],
                "sector": [
                    "technology firms",
                    "renewable energy companies",
                    "public infrastructure projects",
                    "small retail businesses",
                ],
                "period": [
                    "the previous quarter",
                    "the last twelve months",
                    "the latest reporting cycle",
                    "the current fiscal period",
                ],
            },
        },
        {
            "template": (
                "{university} researchers {verb} a peer-reviewed study on {topic} impacting {population}"
            ),
            "fields": {
                "university": [
                    "MIT",
                    "Oxford University",
                    "National University of Singapore",
                    "University of Cape Town",
                ],
                "verb": [
                    "published",
                    "released",
                    "presented",
                    "updated",
                ],
                "topic": [
                    "urban heat mitigation",
                    "precision agriculture tooling",
                    "digital payment security",
                    "public health preparedness",
                ],
                "population": [
                    "metropolitan residents",
                    "smallholder farmers",
                    "banking customers",
                    "emergency responders",
                ],
            },
        },
        {
            "template": (
                "{league} {team_action} {team_name} after {event} during {season}"
            ),
            "fields": {
                "league": [
                    "The Premier League",
                    "The NBA",
                    "La Liga",
                    "The IPL",
                ],
                "team_action": [
                    "awarded fair play recognition to",
                    "confirmed a contract extension with",
                    "introduced new performance analytics for",
                    "scheduled international friendlies for",
                ],
                "team_name": [
                    "Arsenal",
                    "Los Angeles Sparks",
                    "FC Barcelona",
                    "Mumbai Indians",
                ],
                "event": [
                    "the latest governing council review",
                    "the championship finals",
                    "a data-driven audit",
                    "a youth academy showcase",
                ],
                "season": [
                    "the 2024-25 season",
                    "the summer tour",
                    "the winter session",
                    "the preseason window",
                ],
            },
        },
    ]
    clauses = cycle(REAL_CLAUSES)
    samples: List[str] = []
    for builder in builders:
        for sentence in expand_templates(builder["template"], builder["fields"]):
            text = f"{sentence} {next(clauses)}"
            samples.append(text)
    return samples


def build_fake_samples() -> List[str]:
    builders = [
        {
            "template": (
                "{source} claimed that {celebrity} {action} by {absurdity} on {platform}"
            ),
            "fields": {
                "source": [
                    "A viral meme",
                    "An unverified vlog",
                    "A clickbait microblog",
                    "An anonymous chain email",
                ],
                "celebrity": [
                    "a famous pop star",
                    "a retired astronaut",
                    "a tech billionaire",
                    "a television chef",
                ],
                "action": [
                    "cured every disease",
                    "reversed climate change",
                    "teleported across continents",
                    "printed unlimited money",
                ],
                "absurdity": [
                    "whispering to moonlight",
                    "installing secret crystals",
                    "drinking glowing smoothies",
                    "tuning pyramids with headphones",
                ],
                "platform": [
                    "a private livestream",
                    "a hidden forum",
                    "an encrypted group",
                    "a disappearing story",
                ],
            },
        },
        {
            "template": (
                "{rumour_origin} insisted {government} legalized {ridiculous_item} to {nonsense_goal}"
            ),
            "fields": {
                "rumour_origin": [
                    "A fabricated document",
                    "A spoof news site",
                    "A conspiracy newsletter",
                    "A forged SMS alert",
                ],
                "government": [
                    "the United Nations",
                    "the U.S. Congress",
                    "the Supreme Court",
                    "the Central Bank",
                ],
                "ridiculous_item": [
                    "time travel lotteries",
                    "gravity taxes",
                    "mind-control umbrellas",
                    "unicorn livestock permits",
                ],
                "nonsense_goal": [
                    "erase all debts instantly",
                    "ban rainy Mondays",
                    "power cities with wishes",
                    "replace passports with emojis",
                ],
            },
        },
        {
            "template": (
                "{portal} reported oceans turned into {liquid} after {imaginary_event} in {location}"
            ),
            "fields": {
                "portal": [
                    "A fringe astrology portal",
                    "A parody investment channel",
                    "A hoax science blog",
                    "A spam news app",
                ],
                "liquid": [
                    "sparkling soda",
                    "liquid chocolate",
                    "anti-gravity gel",
                    "instant coffee",
                ],
                "imaginary_event": [
                    "two comets colliding",
                    "a wizard summit",
                    "a secret microwave experiment",
                    "planets aligning with Wi-Fi signals",
                ],
                "location": [
                    "the Pacific Ocean",
                    "the Arctic circle",
                    "the Mediterranean Sea",
                    "the Amazon basin",
                ],
            },
        },
        {
            "template": (
                "{post} said that {company} will pay {amount} to anyone who {impossible_task}"
            ),
            "fields": {
                "post": [
                    "A screenshot of a fake press release",
                    "A recycled hoax post",
                    "A doctored advertisement",
                    "A spam SMS screenshot",
                ],
                "company": [
                    "NASA",
                    "The International Olympic Committee",
                    "Federal Reserve",
                    "WHO",
                ],
                "amount": [
                    "$5 million",
                    "$12 billion",
                    "10,000 gold bars",
                    "a lifetime supply of diamonds",
                ],
                "impossible_task": [
                    "forward a message to ten suns",
                    "hold your breath for an hour",
                    "decode alien emojis",
                    "jump from Earth to Mars",
                ],
            },
        },
    ]
    clauses = cycle(FAKE_CLAUSES)
    samples: List[str] = []
    for builder in builders:
        for sentence in expand_templates(builder["template"], builder["fields"]):
            text = f"{sentence} {next(clauses)}"
            samples.append(text)
    return samples


def main() -> None:
    real = build_real_samples()
    fake = build_fake_samples()
    min_len = min(len(real), len(fake))
    real = real[:min_len]
    fake = fake[:min_len]
    records = [{"text": text, "label": "real"} for text in real] + [
        {"text": text, "label": "fake"} for text in fake
    ]
    random.shuffle(records)
    df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

