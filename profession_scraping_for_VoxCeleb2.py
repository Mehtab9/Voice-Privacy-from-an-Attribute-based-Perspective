from collections import Counter, OrderedDict
import re
import time

import pandas as pd
import spacy
import wikipedia
import wptools
from tqdm import tqdm
from wikipedia import DisambiguationError


def wiki_infobox(title: str) -> dict:
    try:
        page = wptools.page(title, silent=True).get_parse()
        infobox = page.data.get("infobox", {}) or {}
    except Exception:
        infobox = {}
    return infobox


def _clean_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        x = " ".join([str(t) for t in x])

    s = str(x)

    # remove common wiki reference markers
    s = re.sub(r"\[[0-9]+\]", "", s)

    # normalize separators and whitespace
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = s.replace("|", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_templates_and_tags(s: str) -> str:
    if not s:
        return ""

    # convert common html breaks to commas
    s = s.replace("<br>", ",").replace("<br/>", ",").replace("<br />", ",")

    # remove html tags
    s = re.sub(r"</?[^>]+>", " ", s)

    # remove common list templates
    s = re.sub(r"\b(hlist|flatlist|plainlist|ubl|unbulleted list)\b", " ", s, flags=re.IGNORECASE)

    # normalize underscores to spaces
    s = s.replace("_", " ")

    # remove braces and brackets leftovers
    s = s.replace("[", " ").replace("]", " ")
    s = s.replace("{", " ").replace("}", " ")

    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_value_to_list(val) -> list[str]:
    s = _clean_text(val)
    s = _strip_templates_and_tags(s)
    if not s:
        return []

    # normalize conjunctions
    s = s.replace(" and ", ",")
    s = s.replace(" & ", ",")

    # split on commas
    parts = [p.strip() for p in s.split(",") if p.strip()]

    out: list[str] = []
    for p in parts:
        p = re.sub(r"\(.*?\)", "", p).strip()
        p = re.sub(r"[^a-zA-Z\s]", " ", p).lower().strip()
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def _first_two_sentences(text: str) -> str:
    if not text:
        return ""
    # wikipedia.summary is usually short, still keep it safe
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(chunks[:2]).strip()


def _is_nationality_only_token(s: str) -> bool:
    if not s:
        return True
    t = s.strip().lower()
    if " " in t:
        return False
    # very common nationality and demonym noise tokens seen in summaries
    demonyms = {
        "american", "british", "english", "scottish", "welsh", "irish",
        "french", "german", "italian", "spanish", "portuguese", "dutch",
        "belgian", "swiss", "austrian", "polish", "czech", "slovak", "hungarian",
        "russian", "ukrainian", "belarusian", "yugoslav", "serbian", "croatian",
        "bosnian", "albanian", "greek", "turkish", "bulgarian", "romanian",
        "swedish", "norwegian", "danish", "finnish", "icelandic",
        "canadian", "australian", "new", "zealand", 
        "mexican", "brazilian", "argentinian", "chilean", "colombian", "peruvian",
        "egyptian", "moroccan", "algerian", "tunisian", "libyan", "sudanese",
        "nigerian", "ghanaian", "kenyan", "ethiopian", "somali", "south", "african",
        "saudi", "emirati", "qatari", "kuwaiti", "bahraini", "oman", "yemeni",
        "iranian", "iraqi", "syrian", "lebanese", "jordanian", "palestinian",
        "pakistani", "indian", "bangladeshi", "sri", "lankan", "nepali",
        "chinese", "japanese", "korean", "taiwanese", "thai", "vietnamese",
        "indonesian", "malaysian", "singaporean", "filipino",
        "israeli", "afghan", "kazakh", "uzbek",
        "pakistan", "india", "egypt", "jordan", "algeria"
    }
    return t in demonyms


def _is_junk_token(s: str) -> bool:
    if not s:
        return True
    t = s.strip().lower()

    # remove template leak or obvious placeholders
    if any(x in t for x in ["hlist", "flatlist", "plainlist", "ubl"]):
        return True


    if len(t.split()) > 10:
        return True

    # discard nationality only tokens
    if _is_nationality_only_token(t):
        return True

    if len(t) < 3:
        return True

    # discard single word tokens that look like names, keep common job words by excluding them
    keep_singletons = {
        "actor", "actress", "singer", "musician", "rapper", "songwriter", "composer",
        "politician", "minister", "senator", "president", "governor", "diplomat",
        "journalist", "author", "writer", "director", "producer", "filmmaker",
        "presenter", "host", "model",
        "athlete", "footballer", "boxer", "cricketer", "golfer", "tennis",
        "coach", "manager", "driver",
        "businessman", "businesswoman", "entrepreneur", "engineer", "economist", "banker",
        "scientist", "researcher", "professor", "academic"
    }
    if " " not in t and t not in keep_singletons:

        return True

    return False


def _pick_best_profession(cands: list[str]) -> str:
    cands = [c for c in cands if c and not _is_junk_token(c)]
    if not cands:
        return "unk"
    return Counter(cands).most_common(1)[0][0]


def _extract_from_summary(text: str) -> list[str]:
    """
    Extract short role phrases from the first two sentences.
    Keeps profession words, avoids filtering them out.
    """
    lead = _first_two_sentences(text)
    if not lead:
        return []

    lead = _strip_templates_and_tags(lead)
    low = lead.lower()

    # common patterns
    patterns = [
        r"\b(is|was)\s+(an?|the)\s+([^\.]{0,120})[\.!]",
        r"\b(is|was)\s+([^\.]{0,120})[\.!]",
    ]

    chunks: list[str] = []
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            chunk = m.group(m.lastindex)
            chunks.append(chunk)
            break

    if not chunks:
        return []

    chunk = chunks[0]
    chunk = re.sub(r"\bfrom\b.*", "", chunk).strip()
    chunk = re.sub(r"\(.*?\)", "", chunk).strip()
    chunk = re.sub(r"[^a-z\s]", " ", chunk)
    chunk = re.sub(r"\s+", " ", chunk).strip()

    # split into parts 
    parts = [p.strip() for p in re.split(r",| and |;| or ", chunk) if p.strip()]
    parts = [_strip_templates_and_tags(p) for p in parts]
    parts = [p for p in parts if p and not _is_junk_token(p)]
    return parts


if __name__ == "__main__":
    # Read metadata
    vox2 = pd.read_csv("./vox2_meta.csv")
    vgg2 = pd.read_csv(
        "./vggface2_meta.csv",
        quotechar='"',
        skipinitialspace=True,
    )

    #  column cleanup
    vox2.columns = vox2.columns.str.strip()
    vgg2.columns = vgg2.columns.str.strip()

    vgg_id_to_name = {k: v.strip() for k, v in zip(vgg2["Class_ID"].values, vgg2["Name"].values)}
    vox2_ids_dict = {k.strip(): v.strip() for k, v in zip(vox2["VoxCeleb2 ID"].values, vox2["VGGFace2 ID"].values)}

    vox2_id_to_name = {}
    for spk_id, vgg_id in vox2_ids_dict.items():
        if vgg_id in vgg_id_to_name:
            vox2_id_to_name[spk_id] = vgg_id_to_name[vgg_id]

    vox2_name_to_id = {name: spkid for spkid, name in vox2_id_to_name.items()}
    vox2_names = list(vox2_id_to_name.values())
    

    vox2_professions = OrderedDict({name: [] for name in vox2_names})

    nlp = spacy.load("en_core_web_sm")

    # Infobox keys that are most likely to contain roles
    infobox_keys = [
        "occupation",
        "occupations",
        "profession",
        "job",
        "title",
        "positions",
        "position",
        "notable_works",
        "known for",
        "known_for",
    ]

    for name in tqdm(vox2_professions, desc="Scraping professions"):
        print(f"Processing: {name}", flush=True)
        qname = " ".join(name.split("_"))

        # Resolve a Wikipedia page title and summary
        page_title = None
        text = ""

        try:
            text = wikipedia.summary(qname, auto_suggest=True)
            page_title = qname
        except Exception:
            search = wikipedia.search(qname, results=5)
            if not search:
                vox2_professions[name] = []
                continue

            for cand in search:
                try:
                    text = wikipedia.summary(cand, auto_suggest=False)
                    page_title = cand
                    break
                except DisambiguationError:
                    continue
                except Exception:
                    continue

        if not page_title:
            vox2_professions[name] = []
            continue

        # Infobox extraction first
        infobox = wiki_infobox(page_title)

        prof_cands: list[str] = []
        for k in infobox_keys:
            if k in infobox and infobox.get(k):
                prof_cands.extend(_split_value_to_list(infobox.get(k)))

        prof_cands = [c for c in prof_cands if c and not _is_junk_token(c)]
        if prof_cands:
            vox2_professions[name] = prof_cands
            time.sleep(0.12)
            continue

        #  Summary based extraction
        parts = _extract_from_summary(text)
        if parts:
            vox2_professions[name] = parts
            time.sleep(0.12)
            continue

        #
        lead = _first_two_sentences(text)
        doc = nlp(lead)
        noun_cands: list[str] = []
        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN"}:
                t = tok.text.lower().strip()
                t = re.sub(r"[^a-z\s]", "", t).strip()
                t = re.sub(r"\s+", " ", t).strip()
                if t and not _is_junk_token(t):
                    noun_cands.append(t)

        vox2_professions[name] = noun_cands
        time.sleep(0.12)

    vox2_prof_final = OrderedDict({})
    for name in vox2_professions:
        vox2_prof_final[name] = _pick_best_profession(vox2_professions[name])

    # Write output
    out_path = "./spk2prof"
    with open(out_path, "w") as fp:
        for name, prof in vox2_prof_final.items():
            spk_id = vox2_name_to_id[name]
            prof_norm = "_".join(prof.split())
            fp.write(f"{spk_id} {prof_norm}\n")

    print(f"Done. Wrote: {out_path}")

