#!/bin/bash
set -euo pipefail

REPO_MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_DIR="${REPO_MODEL_DIR}/tmp_sap_smoke"
mkdir -p "${TMP_DIR}"

# Load OPENAI_API_KEY from .env.local if present.
if [ -f "${REPO_MODEL_DIR}/.env.local" ]; then
    set -a; . "${REPO_MODEL_DIR}/.env.local"; set +a
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY is not set"
    exit 1
fi

# Resolve a python3 that has the project deps installed. Prefer $PYTHON override,
# then the active conda env (if any), then plain python3. Errors if none qualify.
PY=""
for cand in "${PYTHON:-}" "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python3}" "python3"; do
    [ -z "${cand}" ] && continue
    if "${cand}" -c "import transformers" >/dev/null 2>&1; then
        PY="${cand}"; break
    fi
done
if [ -z "${PY}" ]; then
    echo "ERROR: no python3 on PATH has the project dependencies (transformers, etc.)."
    echo "       Activate your conda env first, e.g.:  conda activate aac39"
    exit 1
fi

# macOS + conda Python doesn't ship a usable CA bundle for urllib/requests.
CERTIFI_PATH="$("${PY}" -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
if [ -n "${CERTIFI_PATH}" ]; then
    export SSL_CERT_FILE="${CERTIFI_PATH}"
    export REQUESTS_CA_BUNDLE="${CERTIFI_PATH}"
fi

LLM_MODEL="gpt-4.1-mini"
FENSE_SBERT="paraphrase-TinyBERT-L6-v2"
FENSE_ECHECKER="echecker_clotho_audiocaps_tiny"

cat > "${TMP_DIR}/gen_captions_encoder_reranked.json" << 'JSON'
[
  {
    "idx": 0,
    "audio_file": "synthetic_000.wav",
    "true_captions": [
      "a dog barks loudly as a car drives by on the street",
      "traffic passes in the background while dogs bark",
      "dogs are barking and vehicles pass in the distance",
      "a dog barks repeatedly while cars go past outside",
      "a barking dog is heard as a vehicle drives past"
    ],
    "generated_captions": [
      {
        "text": "a dog is barking while a car is passing in the street",
        "audio_text_similarity": 0.98,
        "source": null
      },
      {
        "text": "dogs are barking with traffic noise in the background",
        "audio_text_similarity": 0.96,
        "source": null
      },
      {
        "text": "a dog barks several times as a vehicle drives past on a road",
        "audio_text_similarity": 0.95,
        "source": null
      },
      {
        "text": "loud barking from a dog mixes with the sound of a passing car",
        "audio_text_similarity": 0.94,
        "source": null
      },
      {
        "text": "a small dog is yapping outdoors while traffic moves nearby",
        "audio_text_similarity": 0.92,
        "source": null
      },
      {
        "text": "people are talking quietly as a vehicle is moving by",
        "audio_text_similarity": 0.91,
        "source": null
      },
      {
        "text": "a dog howls at the moon in a quiet neighborhood at night",
        "audio_text_similarity": 0.83,
        "source": null
      },
      {
        "text": "a siren is sounding and footsteps are echoing nearby",
        "audio_text_similarity": 0.75,
        "source": null
      },
      {
        "text": "rain is falling heavily on a tin roof with thunder in the distance",
        "audio_text_similarity": 0.62,
        "source": null
      },
      {
        "text": "a cat meows softly while a clock ticks on the wall",
        "audio_text_similarity": 0.55,
        "source": null
      }
    ]
  },
  {
    "idx": 1,
    "audio_file": "synthetic_001.wav",
    "true_captions": [
      "rain falls steadily on a rooftop with occasional thunder",
      "a heavy rainstorm with rumbling thunder in the distance",
      "rain pours down and thunder rolls in the background",
      "steady rainfall is accompanied by distant thunderclaps",
      "thunder rumbles as rain drums against a surface"
    ],
    "generated_captions": [
      {
        "text": "rain is falling steadily while thunder rumbles in the distance",
        "audio_text_similarity": 0.97,
        "source": null
      },
      {
        "text": "heavy rain pours down with occasional thunder",
        "audio_text_similarity": 0.96,
        "source": null
      },
      {
        "text": "a thunderstorm with steady rain and distant thunderclaps",
        "audio_text_similarity": 0.95,
        "source": null
      },
      {
        "text": "rainfall drums on a surface as thunder rolls overhead",
        "audio_text_similarity": 0.93,
        "source": null
      },
      {
        "text": "water is running from a faucet into a metal sink",
        "audio_text_similarity": 0.70,
        "source": null
      },
      {
        "text": "applause from a large crowd fills an indoor venue",
        "audio_text_similarity": 0.58,
        "source": null
      },
      {
        "text": "ocean waves crash against a rocky shore repeatedly",
        "audio_text_similarity": 0.66,
        "source": null
      },
      {
        "text": "fireworks explode in the night sky during a celebration",
        "audio_text_similarity": 0.52,
        "source": null
      }
    ]
  },
  {
    "idx": 2,
    "audio_file": "synthetic_002.wav",
    "true_captions": [
      "a person types on a keyboard in a quiet office",
      "someone is typing quickly on a mechanical keyboard",
      "keyboard keys clack as a person types at a desk",
      "rapid typing sounds with the occasional mouse click",
      "fingers tap on keyboard keys in a silent room"
    ],
    "generated_captions": [
      {
        "text": "a person is typing on a keyboard in a quiet room",
        "audio_text_similarity": 0.96,
        "source": null
      },
      {
        "text": "keyboard keys are being pressed rapidly with occasional mouse clicks",
        "audio_text_similarity": 0.94,
        "source": null
      },
      {
        "text": "someone types quickly on a mechanical keyboard at an office desk",
        "audio_text_similarity": 0.93,
        "source": null
      },
      {
        "text": "fingers tap against plastic keys in a steady rhythm",
        "audio_text_similarity": 0.90,
        "source": null
      },
      {
        "text": "a woodpecker is tapping on a tree in a forest",
        "audio_text_similarity": 0.72,
        "source": null
      },
      {
        "text": "rain taps lightly on a window pane",
        "audio_text_similarity": 0.68,
        "source": null
      },
      {
        "text": "someone is playing a piano softly in the next room",
        "audio_text_similarity": 0.60,
        "source": null
      },
      {
        "text": "a drummer taps a snare drum with brushes",
        "audio_text_similarity": 0.55,
        "source": null
      }
    ]
  },
  {
    "idx": 3,
    "audio_file": "synthetic_003.wav",
    "true_captions": [
      "birds chirp loudly in the early morning outdoors",
      "multiple birds are singing in a forest at dawn",
      "a chorus of birdsong fills the morning air",
      "several birds tweet and call to one another outside",
      "morning birdsong echoes through trees"
    ],
    "generated_captions": [
      {
        "text": "birds are chirping loudly in the morning",
        "audio_text_similarity": 0.97,
        "source": null
      },
      {
        "text": "a variety of birds sing in a forest at sunrise",
        "audio_text_similarity": 0.95,
        "source": null
      },
      {
        "text": "many birds tweet and call out in an outdoor setting",
        "audio_text_similarity": 0.94,
        "source": null
      },
      {
        "text": "a chorus of songbirds fills the air near trees",
        "audio_text_similarity": 0.92,
        "source": null
      },
      {
        "text": "a single bird whistles a short tune and then stops",
        "audio_text_similarity": 0.80,
        "source": null
      },
      {
        "text": "a flute plays a high-pitched melody indoors",
        "audio_text_similarity": 0.65,
        "source": null
      },
      {
        "text": "an alarm clock beeps repeatedly in a bedroom",
        "audio_text_similarity": 0.50,
        "source": null
      },
      {
        "text": "a microwave oven signals that cooking is complete",
        "audio_text_similarity": 0.45,
        "source": null
      }
    ]
  },
  {
    "idx": 4,
    "audio_file": "synthetic_004.wav",
    "true_captions": [
      "a crowd cheers and claps at a sports event",
      "loud applause and cheering from a large audience",
      "people are clapping and shouting in a stadium",
      "an excited crowd applauds and yells",
      "spectators cheer loudly during a game"
    ],
    "generated_captions": [
      {
        "text": "a large crowd is cheering and clapping loudly",
        "audio_text_similarity": 0.97,
        "source": null
      },
      {
        "text": "people in a stadium applaud and shout with excitement",
        "audio_text_similarity": 0.96,
        "source": null
      },
      {
        "text": "an audience erupts in applause and cheering at a game",
        "audio_text_similarity": 0.94,
        "source": null
      },
      {
        "text": "loud cheers and whistles come from a sports crowd",
        "audio_text_similarity": 0.92,
        "source": null
      },
      {
        "text": "heavy rain falls on a metal surface in a storm",
        "audio_text_similarity": 0.68,
        "source": null
      },
      {
        "text": "a fire crackles loudly in an outdoor pit",
        "audio_text_similarity": 0.55,
        "source": null
      },
      {
        "text": "ocean waves wash onto a pebble beach",
        "audio_text_similarity": 0.50,
        "source": null
      },
      {
        "text": "a single person claps slowly in an empty auditorium",
        "audio_text_similarity": 0.72,
        "source": null
      }
    ]
  },
  {
    "idx": 5,
    "audio_file": "synthetic_005.wav",
    "true_captions": [
      "a baby is crying while adults talk in the background",
      "an infant cries loudly as people speak nearby",
      "a young child wails while conversation continues",
      "a baby's cries are heard over muffled voices",
      "crying from a baby accompanies quiet talking"
    ],
    "generated_captions": [
      {
        "text": "a baby cries loudly while people talk in the background",
        "audio_text_similarity": 0.97,
        "source": null
      },
      {
        "text": "an infant is wailing while adults converse quietly nearby",
        "audio_text_similarity": 0.95,
        "source": null
      },
      {
        "text": "a child is crying and voices can be heard speaking softly",
        "audio_text_similarity": 0.93,
        "source": null
      },
      {
        "text": "a young baby wails as muffled conversation continues",
        "audio_text_similarity": 0.91,
        "source": null
      },
      {
        "text": "a kitten meows repeatedly in a small room",
        "audio_text_similarity": 0.62,
        "source": null
      },
      {
        "text": "seagulls cry overhead at a coastal pier",
        "audio_text_similarity": 0.55,
        "source": null
      },
      {
        "text": "a person laughs heartily at a joke during a dinner",
        "audio_text_similarity": 0.48,
        "source": null
      },
      {
        "text": "a violin plays a mournful tune in a concert hall",
        "audio_text_similarity": 0.40,
        "source": null
      }
    ]
  },
  {
    "idx": 6,
    "audio_file": "synthetic_006.wav",
    "true_captions": [
      "water is running from a faucet into a sink",
      "a tap is pouring water steadily into a basin",
      "running water flows from a kitchen faucet",
      "water streams from a tap and splashes in a sink",
      "a faucet is left on with water flowing continuously"
    ],
    "generated_captions": [
      {
        "text": "water is running from a faucet into a metal sink",
        "audio_text_similarity": 0.96,
        "source": null
      },
      {
        "text": "a tap pours water continuously into a basin",
        "audio_text_similarity": 0.94,
        "source": null
      },
      {
        "text": "a kitchen faucet is left running with water splashing",
        "audio_text_similarity": 0.93,
        "source": null
      },
      {
        "text": "steady water flow from a tap can be heard in a sink",
        "audio_text_similarity": 0.91,
        "source": null
      },
      {
        "text": "heavy rain falls on pavement during a thunderstorm",
        "audio_text_similarity": 0.75,
        "source": null
      },
      {
        "text": "a waterfall cascades down rocks in a remote forest",
        "audio_text_similarity": 0.68,
        "source": null
      },
      {
        "text": "someone is taking a shower behind a closed door",
        "audio_text_similarity": 0.70,
        "source": null
      },
      {
        "text": "a tea kettle whistles loudly on a gas stove",
        "audio_text_similarity": 0.42,
        "source": null
      }
    ]
  },
  {
    "idx": 7,
    "audio_file": "synthetic_007.wav",
    "true_captions": [
      "a motorcycle engine revs and accelerates down a road",
      "a motorbike speeds by with a loud engine",
      "an engine roars as a motorcycle drives past",
      "a motorcycle accelerates rapidly on a street",
      "the sound of a motorcycle engine is heard passing by"
    ],
    "generated_captions": [
      {
        "text": "a motorcycle engine revs loudly and accelerates down a street",
        "audio_text_similarity": 0.97,
        "source": null
      },
      {
        "text": "a motorbike speeds past with its engine roaring",
        "audio_text_similarity": 0.96,
        "source": null
      },
      {
        "text": "the loud sound of a motorcycle passing by on a road",
        "audio_text_similarity": 0.94,
        "source": null
      },
      {
        "text": "a two wheeled vehicle accelerates with a deep engine note",
        "audio_text_similarity": 0.90,
        "source": null
      },
      {
        "text": "a chainsaw is cutting through a log in a backyard",
        "audio_text_similarity": 0.70,
        "source": null
      },
      {
        "text": "a lawn mower is being pushed across a grass lawn",
        "audio_text_similarity": 0.65,
        "source": null
      },
      {
        "text": "a car engine idles in a driveway before shutting off",
        "audio_text_similarity": 0.72,
        "source": null
      },
      {
        "text": "a blender is crushing ice in a kitchen",
        "audio_text_similarity": 0.40,
        "source": null
      }
    ]
  }
]
JSON

"${PY}" -u "${REPO_MODEL_DIR}/reranking/llm_summarize_sap.py" \
  "${TMP_DIR}" \
  "${LLM_MODEL}" \
  3 \
  2 \
  "none"

echo "[ok] smoke output:"
ls -l "${TMP_DIR}/llm_sap_summary_output.csv" "${TMP_DIR}/llm_sap_summary_details.json"
echo "[ok] preview csv:"
cat "${TMP_DIR}/llm_sap_summary_output.csv"

echo
echo "[ok] scoring synthesized caption with FENSE..."
REPO_MODEL_DIR="${REPO_MODEL_DIR}" TMP_DIR="${TMP_DIR}" \
LLM_MODEL="${LLM_MODEL}" FENSE_SBERT="${FENSE_SBERT}" FENSE_ECHECKER="${FENSE_ECHECKER}" \
"${PY}" -u - <<'PY'
import csv, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.environ["REPO_MODEL_DIR"], "caption_evaluation_tools", "coco_caption", "pycocoevalcap"))
from fense.evaluator import Evaluator

tmp = os.environ["TMP_DIR"]
with open(os.path.join(tmp, "gen_captions_encoder_reranked.json")) as f:
    refs_by_file = {s["audio_file"]: s["true_captions"] for s in json.load(f)}
with open(os.path.join(tmp, "llm_sap_summary_output.csv")) as f:
    rows = list(csv.DictReader(f))

print(f"  LLM (SAP synthesis):  {os.environ['LLM_MODEL']}")
print(f"  FENSE SBERT:          {os.environ['FENSE_SBERT']}")
print(f"  FENSE error checker:  {os.environ['FENSE_ECHECKER']}")

ev = Evaluator(device="cpu", sbert_model=os.environ["FENSE_SBERT"],
               echecker_model=os.environ["FENSE_ECHECKER"])
for row in rows:
    fn, cap = row["file_name"], row["caption_predicted"]
    refs = refs_by_file.get(fn, [])
    if not refs:
        print(f"  [warn] {fn}: no refs, skipping")
        continue
    raw, err_prob, fense = ev.sentence_score(cap, refs, return_error_prob=True)
    print(f"  file:      {fn}")
    print(f"  caption:   {cap}")
    print(f"  SBERT sim: {raw:.4f}   error_prob: {err_prob:.4f}   FENSE: {fense:.4f}")
PY