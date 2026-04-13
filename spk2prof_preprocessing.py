import csv
import re
from pathlib import Path

in_path = Path("./spk2prof")
out_path = in_path.parent / "spk2prof_clean.csv"

SINGER_KW = {
    "singer", "musician", "rapper", "songwriter", "composer",
    "vocalist", "pianist", "guitarist", "drummer", "dj", "playback"
}

ACTOR_KW = {
    "actor", "actress", "film", "director", "producer",
    "screenwriter", "filmmaker", "model", "comedian",
    "voice", "voiceover"
}

POLITICIAN_KW = {
    "politician", "minister", "president", "prime", "senator",
    "parliament", "mp", "governor", "mayor", "diplomat",
    "party", "king", "queen", "prince", "princess", "royal"
}

ATHLETE_KW = {
    "athlete", "football", "footballer", "soccer", "boxer", "boxing",
    "cricket", "cricketer", "tennis", "golfer", "wrestler",
    "racing", "driver", "formula", "ski", "skier", "snowboarder",
    "runner", "gymnast", "mma", "coach", "manager",
    "basketball", "baseball", "hockey",
    "rugby", "badminton", "swimmer"
}

SPORT_POSITION_KW = {
    "striker", "midfielder", "defender", "goalkeeper",
    "centre", "center", "back", "winger",
    "forward"
}

MEDIA_PHRASES = {
    "tv", "television",
    "tv personality", "television personality",
    "tv_personality", "television_personality",
    "presenter", "host", "broadcaster",
    "pundit", "commentator", "news correspondent",
    "anchor"
}

BUSINESS_KW = {
    "businessman", "businesswoman", "entrepreneur",
    "executive", "chairman"
}

EDU_SCI_KW = {
    "teacher", "professor", "academic", "scientist"
}

LEGAL_KW = {
    "lawyer", "advocate", "attorney", "barrister", "solicitor",
    "prosecutor", "jurist", "judge", "legal"
}

JOURNALIST_KW = {"journalist"}

BANKER_KW = {"banker"}

INVESTOR_KW = {"investor"}

WRITER_KW = {"writer"}

CHAMPION_KW = {"champion"}

TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> set:
    text = text.lower().replace("_", " ")
    return set(TOKEN_RE.findall(text))


def contains_media(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in MEDIA_PHRASES)


def assign_category(prof: str):
    if prof == "unk":
        return "unk", 0

    tokens = tokenize(prof)

    has_media = contains_media(prof)

    has_actor = bool(tokens & ACTOR_KW)
    has_singer = bool(tokens & SINGER_KW)
    has_politician = bool(tokens & POLITICIAN_KW)
    has_athlete = bool((tokens & ATHLETE_KW) or (tokens & SPORT_POSITION_KW))
    has_business = bool(tokens & BUSINESS_KW)
    has_edu_sci = bool(tokens & EDU_SCI_KW)
    has_legal = bool(tokens & LEGAL_KW)

    has_journalist = bool(tokens & JOURNALIST_KW)
    has_banker = bool(tokens & BANKER_KW)
    has_investor = bool(tokens & INVESTOR_KW)
    has_writer = bool(tokens & WRITER_KW)
    has_champion = bool(tokens & CHAMPION_KW)

    # Media assignment only when TV cues exist and none of these are present
    # actor, singer, politician, comedian, model
    if has_media and (not has_actor) and (not has_singer) and (not has_politician):
        return "Media and TV personalities without acting", 1

    # journalist to Media, only if no actor, singer, politician, businessman
    if has_journalist and (not has_actor) and (not has_singer) and (not has_politician) and (not has_business):
        return "Media and TV personalities without acting", 1

    # Actor beats singer if both appear
    if has_actor:
        return "Actors", 1

    if has_singer:
        return "Singers and musicians", 1

    if has_politician:
        return "Politicians", 1

    # champion to Athletes, only if no actor, singer, politician, businessman
    if has_champion and (not has_actor) and (not has_singer) and (not has_politician) and (not has_business):
        return "Athletes", 1

    # swimmer
    if ("swimmer" in tokens) and (not has_actor) and (not has_singer) and (not has_politician) and (not has_business):
        return "Athletes", 1

    if has_athlete:
        return "Athletes", 1

    # business rules
    if has_business and has_politician:
        return "Politicians", 1

    if has_business:
        return "Other", 1

    # education and science roles
    if has_edu_sci:
        return "Other", 1

    # Legal roles, only when no other category cues exist
    if has_legal and (not has_media) and (not has_actor) and (not has_singer) and (not has_politician) and (not has_athlete):
        return "Other", 1

    # Banker to Other, only if no actor, singer, politician, athlete, media
    if has_banker and (not has_media) and (not has_actor) and (not has_singer) and (not has_politician) and (not has_business) and (not has_athlete):
        return "Other", 1

    # Investor to Other, only if no actor, singer, politician, athlete, media
    if has_investor and (not has_media) and (not has_actor) and (not has_singer) and (not has_politician) and (not has_business) and (not has_athlete):
        return "Other", 1

    # Writer to Other, only if no actor, singer, politician, athlete, media
    if has_writer and (not has_media) and (not has_actor) and (not has_singer) and (not has_politician) and (not has_business) and (not has_athlete):
        return "Other", 1

    # otherwise leave unassigned 
    return "", 0


def main():
    rows_out = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            spk_id, prof = line.split(maxsplit=1)

            category, assigned = assign_category(prof)

            rows_out.append({
                "spk_id": spk_id,
                "scrape_prof": prof,
                "assigned_category": category,
                "is_assigned": assigned
            })

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["spk_id", "scrape_prof", "assigned_category", "is_assigned"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
