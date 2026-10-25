# =============================================================================
# clip_labels.py — Single source of truth for all category labels
#
# 13 canonical categories (finalized):
#   top, sports_bra, pants, leggings, shorts, skirt, dress,
#   sweater, jacket, shoes, hat, bag, jumpsuit
#
# MATCHING APPROACH: exact token matching (split on spaces).
# Substring matching was considered but rejected — "hat" inside "that",
# "cap" inside "capsule", "bag" inside "baggy" all produce silent wrong
# classifications. When a word and its compound form both need to match
# (sleeve / sleeveless), add both explicitly.
# =============================================================================

CANONICAL_LABELS = [
    "top",
    "sports_bra",
    "pants",
    "leggings",
    "shorts",
    "skirt",
    "dress",
    "sweater",
    "jacket",
    "shoes",
    "hat",
    "bag",
    "jumpsuit",
    "not_fashion",
]

# =============================================================================
# CLIP_PROMPTS
# Richer text prompts fed to CLIPProcessor at startup — same order as
# CANONICAL_LABELS.  The output category names (keys in category_scores,
# stored in Qdrant) still come from CANONICAL_LABELS; only the text
# embedding used for image→category matching changes.
#
# Guidelines per entry:
#   • Lead with the canonical token so the embedding stays anchored.
#   • Add the 3-5 most visually distinctive synonyms / sub-types.
#   • Keep each string short enough for CLIP (<77 tokens).
# =============================================================================

CLIP_PROMPTS = [
    # top — "shirt halterneck" targets case 1 (halterneck→dress) while keeping shirt
    "top shirt halterneck",
    # sports_bra
    "sports bra",
    # pants — keep short; avoid leaking into leggings/shirt territory
    "pants trousers",
    # leggings — "tights" widens the gap over "pants trousers" (case 3)
    "leggings tights",
    # shorts
    "shorts",
    # skirt — keep single token; long prompt caused sandals regression
    "skirt",
    # dress — keep single token; long prompt caused mini-dress→top regression
    "dress",
    # sweater — keep single token; long prompt caused blazer/puffer→sweater regression
    "sweater",
    # jacket — "coat" alone fixes case 2 (longline wool coat→top)
    "jacket coat",
    # shoes — add "sandals heels" to fix strappy-sandal→skirt regression
    "shoes sandals heels",
    # hat — "cap" targets cases 5 & 6; also activates gateway accessory fallback
    "hat cap",
    # bag — "handbag" targets case 4; also activates gateway accessory fallback
    "bag handbag",
    # jumpsuit
    "jumpsuit",
    # not_fashion — conservative at index time; only fires on clearly non-garment items
    "socks underwear swimsuit belt scarf non-clothing",
]

assert len(CLIP_PROMPTS) == len(CANONICAL_LABELS), (
    "CLIP_PROMPTS and CANONICAL_LABELS must have the same length and order"
)

# =============================================================================
# QUERY_CLIP_PROMPTS
# Used at query time when the user draws a crop from a real-world photo.
# Standard CLIP prompt engineering: "a photo of a person wearing X" outperforms
# bare label names on in-the-wild images (vs product-on-white-background).
# Same order as CANONICAL_LABELS. Does NOT affect index-time classification.
# =============================================================================

QUERY_CLIP_PROMPTS = [
    "a shirt or top covering the chest and upper torso",              # top
    "a sports bra or athletic crop top on the chest",                 # sports_bra
    "pants or jeans covering the legs and lower body from waist down",# pants
    "leggings or tights, stretchy fabric covering the legs",          # leggings
    "shorts on the upper legs above the knee",                        # shorts
    "a skirt around the hips and upper legs",                         # skirt
    "a dress covering the body from shoulders down to the legs",      # dress
    "a sweater or hoodie covering the chest and shoulders",           # sweater
    "a jacket or coat over the upper body",                           # jacket
    "shoes or boots on the feet",                                     # shoes
    "a hat or cap on the head",                                       # hat
    "a handbag or purse",                                             # bag
    "a jumpsuit or one-piece garment covering the full body",         # jumpsuit
    "socks, underwear, swimwear, belt, or non-clothing accessory",    # not_fashion
]

assert len(QUERY_CLIP_PROMPTS) == len(CANONICAL_LABELS), (
    "QUERY_CLIP_PROMPTS and CANONICAL_LABELS must have the same length and order"
)

# =============================================================================
# UNAMBIGUOUS_TOKEN_MAP
# =============================================================================

UNAMBIGUOUS_TOKEN_MAP = {

    # ── top ────────────────────────────────────────────────────────────────
    "shirt":        "top",
    "blouse":       "top",
    "tee":          "top",
    "tees":         "top",
    "tshirt":       "top",
    "camisole":     "top",
    "cami":         "top",
    "jersey":       "top",
    "t-shirt":      "top",
    "Teeshirt":     "top",
    "top":          "top",
    "tops":         "top",
    "camisole":     "top",
    "bustier":      "top",
    "corset":       "top",
    "halterneck":   "top",
    "bodysuit":     "top",
    "tankini":      "top",
    "polo":         "top",
    "tank":         "top",
    "sleeve":       "top",      # catches "long sleeve", "short sleeve"
    "sleeved":      "top",      # catches "short sleeved", "long sleeved"
    "sleeveless":   "top",
    "baselayer":    "top",
    "base":         "top",      # catches "base layer" after split
    "chemise":      "top",
    "débardeur":    "top",

    # ── sports_bra ─────────────────────────────────────────────────────────
    "bra":          "sports_bra",
    "bralette":     "sports_bra",
    "sports bra":   "sports_bra",
    "sport bra":    "sports_bra",
    "brassière":    "sports_bra",

    # ── sweater ────────────────────────────────────────────────────────────
    "sweater":      "sweater",
    "cardigan":     "sweater",
    "hoodie":       "sweater",
    "hoody":        "sweater",
    "sweatshirt":   "sweater",
    "hyperfleece":    "sweater",
    "pullover":     "sweater",
    "knitwear":     "sweater",
    "jumper":       "sweater",
    "crewneck":     "sweater",
    "hooded":   "sweater",
    "fleece":       "sweater",
    "Thermal":       "sweater",
    "turtleneck":   "sweater",
    "sherpa":       "sweater",
    "pull":         "sweater",
    "tricot":       "sweater",

    # ── jacket ─────────────────────────────────────────────────────────────
    "jacket":       "jacket",
    "blazer":       "jacket",
    "gilet":        "jacket",
    "waistcoat":    "jacket",
    "waiscoat":     "jacket",   # common product title typo
    "vest":         "jacket",   # title token only — YOLO "vest" label is handled by CLIP relabel
    "bomber":       "jacket",
    "windbreaker":  "jacket",
    "anorak":       "jacket",
    "cagoule":      "jacket",
    "puffer":       "jacket",
    "overshirt":    "jacket",
    "shacket":      "jacket",
    "varsity":      "jacket",
    "letterman":    "jacket",
    "coat":         "jacket",
    "overcoat":     "jacket",
    "trench":       "jacket",
    "peacoat":      "jacket",
    "parka":        "jacket",
    "raincoat":     "jacket",
    "mackintosh":   "jacket",
    "cape":         "jacket",
    "veste":        "jacket",
    "blouson":      "jacket",
    "doudoune":     "jacket",
    "manteau":      "jacket",
    "imperméable":  "jacket",

    # ── dress ──────────────────────────────────────────────────────────────
    "dress":        "dress",
    "gown":         "dress",
    "sundress":     "dress",
    "bodycon":      "dress",
    "kaftan":       "dress",
    "abaya":        "dress",
    "jalabiya":     "dress",
    "robe":         "dress",

    # ── jumpsuit ───────────────────────────────────────────────────────────
    "jumpsuit":     "jumpsuit",
    "romper":       "jumpsuit",
    "playsuit":     "jumpsuit",
    "overalls":     "jumpsuit",
    "dungarees":    "jumpsuit",
    "catsuit":      "jumpsuit",
    "boilersuit":   "jumpsuit",
    "tracksuit":    "jumpsuit",
    "combinaison":  "jumpsuit",
    "salopette":    "jumpsuit",

    # ── skirt ──────────────────────────────────────────────────────────────
    "skirt":        "skirt",
    "skorts":       "skirt",
    "skort":        "skirt",
    "jupe":         "skirt",

    # ── pants ──────────────────────────────────────────────────────────────
    "pants":        "pants",
    "trousers":     "pants",
    "trouser":      "pants",
    "jeans":        "pants",
    "jean":        "pants",
    "joggers":      "pants",
    "jogger":       "pants",
    "sweatpant":    "pants",
    "sweatpants":   "pants",
    "trackpants":   "pants",
    "chinos":       "pants",
    "chino":        "pants",
    "khakis":       "pants",
    "culottes":     "pants",
    "palazzos":     "pants",
    "pantalon":     "pants",
    "pant":         "pants",

    # ── leggings ───────────────────────────────────────────────────────────
    "leggings":     "leggings",
    "legging":      "leggings",
    "tight":        "leggings",
    "tights":       "leggings",
    "collant":      "leggings",

    # ── shorts ─────────────────────────────────────────────────────────────
    "shorts":       "shorts",
    "short":        "shorts",
    "trunks":       "shorts",
    "sweatshorts":   "shorts",
    "bermuda":      "shorts",

    # ── shoes ──────────────────────────────────────────────────────────────
    "shoes":        "shoes",
    "shoe":         "shoes",
    "sneakers":     "shoes",
    "sneaker":      "shoes",
    "trainers":     "shoes",
    "boots":        "shoes",
    "boot":         "shoes",
    "sandals":      "shoes",
    "sandal":       "shoes",
    "slippers":     "shoes",
    "slipper":      "shoes",
    "loafers":      "shoes",
    "loafer":       "shoes",
    "heels":        "shoes",
    "pumps":        "shoes",
    "mules":        "shoes",
    "espadrilles":  "shoes",
    "wedges":       "shoes",
    "ballerinas":   "shoes",
    "flats":        "shoes",
    "flat":         "shoes",
    "clogs":        "shoes",
    "clog":         "shoes",
    "chaussures":   "shoes",
    "bottes":       "shoes",
    "footwear":     "shoes",

    # ── bag ────────────────────────────────────────────────────────────────
    "bag":          "bag",
    "bags":         "bag",
    "backpack":     "bag",
    "backpacks":    "bag",
    "handbag":      "bag",
    "handbags":     "bag",
    "tote":         "bag",
    "clutch":       "bag",
    "pouch":        "bag",
    "purse":        "bag",
    "satchel":      "bag",
    "duffel":       "bag",
    "duffle":       "bag",
    "holdall":      "bag",
    "sacoche":      "bag",
    "pochette":     "bag",

    # ── hat ────────────────────────────────────────────────────────────────
    "hat":          "hat",
    "cap":          "hat",
    "beanie":       "hat",
    "beret":        "hat",
    "snapback":     "hat",
    "visor":        "hat",
    "fedora":       "hat",
    "panama":       "hat",
    "casquette":    "hat",
    "chapeau":      "hat",

    # ── not_fashion ────────────────────────────────────────────────────────
    "belt":         "not_fashion",
    "scarf":        "not_fashion",
    "sarong":       "not_fashion",
    "kimono":       "not_fashion",
    "tie":          "not_fashion",
    "jibbitz":      "not_fashion",
    "shinguard":    "not_fashion",
    "shinguards":   "not_fashion",
    "goalkeeper":   "not_fashion",
    "shin":         "not_fashion",
    "ball":         "not_fashion",
    "balls":        "not_fashion",
    "racket":       "not_fashion",
    "racquet":      "not_fashion",
    "helmet":       "not_fashion",
    "towel":        "not_fashion",
    "mat":          "not_fashion",
    "dumbbells":    "not_fashion",
    "weights":      "not_fashion",
    "flask":        "not_fashion",
    "swimsuit":     "not_fashion",
    "bikini":       "not_fashion",
    "swimwear":     "not_fashion",
    "swimshirt":    "not_fashion",
    "wallet":        "not_fashion",
    "billfold":       "not_fashion",
    "hair":         "not_fashion",
    "socks":        "not_fashion",
    "sock":         "not_fashion",
    "underwear":    "not_fashion",
    "boxer":        "not_fashion",
    "briefs":       "not_fashion",
    "hair":         "not_fashion",
    "bottom":       "not_fashion",
    "bottoms":      "not_fashion",
    "necklace":      "not_fashion",
    "jewelry":       "not_fashion",
    "ring":          "not_fashion",
    "glove":         "not_fashion",
    "gloves":        "not_fashion",
    "boxing glove":   "not_fashion",
    "goalkeeper glove":   "not_fashion",
    "sunglasses":    "not_fashion",
    "set":           "not_fashion",
    "bracelet":       "not_fashion",
    "charm":          "not_fashion",
    "lipstick":       "not_fashion",
    
}


# =============================================================================
# LABEL_DESCRIPTIONS
# =============================================================================

LABEL_DESCRIPTIONS = {

    "top": (
        "top t-shirt tee tank camisole tube crop top "
        "halter halterneck polo button down button up henley "
        "long sleeve short sleeve sleeveless strapless "
        "base layer baselayer thermal top performance top gym top workout top "
        "fitted top casual top everyday top "
        "débardeur haut "
    ),

    "sports_bra": (
        "sports bra bralette athletic bra workout bra gym bra "
        "high impact sports bra medium support bra low impact bra "
        "crop sports bra padded sports bra zip front sports bra "
        "activewear bra fitness bra running bra yoga bra "
        "brassière de sport "
    ),

    "sweater": (
        "sweater cardigan hoodie sweatshirt pullover knitwear knit jumper "
        "crewneck turtleneck mock neck zip up quarter zip full zip "
        "fleece sherpa long cardigan cable knit chunky knit "
        "cashmere wool merino round neck "
        "pull tricot sweat chandail "
    ),

    "jacket": (
        "jacket blazer gilet waistcoat vest puffer quilted padded "
        "windbreaker anorak rain jacket softshell hardshell "
        "bomber varsity letterman trucker denim jacket leather jacket "
        "biker jacket moto jacket track jacket zip jacket "
        "suit jacket sport coat tuxedo jacket "
        "overshirt shacket field jacket utility jacket cargo jacket "
        "military jacket ski jacket snowboard jacket "
        "coat overcoat trench peacoat duster longline coat "
        "wool coat cashmere coat wrap coat belted coat "
        "parka hooded coat duffle coat raincoat mac "
        "faux fur coat shearling coat teddy coat cape poncho "
        "veste blouson doudoune manteau imperméable "
    ),

    "dress": (
        "dress mini dress midi dress maxi dress gown evening gown "
        "sundress shirt dress wrap dress bodycon slip dress "
        "skater dress a-line dress smock dress tiered dress "
        "strapless dress halter dress one shoulder dress "
        "party dress occasion dress kaftan abaya "
        "robe robe courte robe longue "
    ),

    "jumpsuit": (
        "jumpsuit romper playsuit overalls dungarees one-piece outfit "
        "boilersuit catsuit tracksuit utility jumpsuit wide leg jumpsuit "
        "combinaison salopette "
    ),

    "skirt": (
        "skirt mini skirt midi skirt maxi skirt "
        "pleated skirt wrap skirt pencil skirt a-line skirt "
        "tiered skirt ruffle skirt flared skirt circle skirt "
        "denim skirt tennis skirt skort "
        "jupe jupe courte jupe longue "
    ),

    "pants": (
        "pants trousers trouser jeans pant wide leg straight leg slim "
        "barrel leg flare bootcut cropped pants "
        "joggers sweatpants trackpants cargo pants "
        "chinos chino khakis dress pants suit pants tailored trousers "
        "palazzo culottes parachute pants "
        "pantalon jean "
    ),

    "leggings": (
        "leggings tights tight running tights compression tights "
        "thermal tights yoga pants gym leggings workout leggings "
        "high waist leggings seamless leggings printed leggings "
        "athletic leggings performance leggings activewear leggings "
        "collant legging "
    ),

    "shorts": (
        "shorts denim shorts cargo shorts chino shorts "
        "running shorts sport shorts board shorts "
        "hiking shorts casual shorts bermuda trunks "
    ),

    "shoes": (
        "shoes sneakers boots sandals slippers loafers heels pumps "
        "mules flip flops espadrilles wedges platforms ballerinas flats "
        "ankle boots chelsea boots combat boots knee high boots "
        "running shoes basketball shoes tennis shoes "
        "hiking boots trail shoes clogs slip-on "
        "ski boots snowboard boots "
        "chaussures bottes sandales footwear "
    ),

    "bag": (
        "bag backpack handbag tote shoulder bag crossbody "
        "clutch pouch purse satchel messenger bum bag "
        "fanny pack waist bag gym bag duffel duffle "
        "weekender travel bag holdall laptop bag "
        "mini bag belt bag "
        "sac sac à dos sacoche pochette "
    ),

    "hat": (
        "hat cap beanie bucket hat beret snapback "
        "baseball cap trucker cap fitted cap visor "
        "sun hat straw hat fedora panama "
        "bobble hat ski hat winter hat "
        "chapeau bonnet casquette "
    ),

    "not_fashion": (
        "swimsuit bikini one piece swimwear bathing suit "
        "belt leather belt scarf tie sarong kimono "
        "jibbitz decoration pin novelty "
        "ball football basketball tennis ball volleyball rugby "
        "shinguard shin guard shin pad knee pad elbow pad "
        "racket racquet bat hockey stick "
        "helmet protection guard "
        "towel mat yoga mat water bottle flask "
        "resistance band jump rope weights dumbbells "
        "socks underwear boxer brief "
        "glove goalkeeper glove boxing glove "
    ),
}


# =============================================================================
# YOLO_TO_CANONICAL
#
# "vest" removed — DeepFashion2 uses "vest" for ALL sleeveless garments,
# including tank tops, halternecks, and crop tops. Mapping it to "jacket"
# caused every sleeveless top to be labelled jacket at search time.
# The CLIP relabel block in detect_objects() handles these correctly
# by looking at the actual crop instead of relying on the YOLO label.
# =============================================================================

YOLO_TO_CANONICAL = {
    "short sleeved shirt":   "top",
    "long sleeved shirt":    "top",
    "short sleeved outwear": "jacket",
    "long sleeved outwear":  "jacket",
    "vest":                  "top",    # default "top"; CLIP relabel overrides → "jacket" for actual waistcoats (conf ≥ 0.52)
    "sling":                 "top",
    "shorts":                "shorts",
    "trousers":              "pants",
    "skirt":                 "skirt",
    "short sleeved dress":   "dress",
    "long sleeved dress":    "dress",
    "vest dress":            "dress",
    "sling dress":           "dress",
}

