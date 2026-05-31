"""
MVCNN-style multi-view image renderer for Minecraft schematics.
Reads schematics_0.tfrecords, renders N views per schematic (ortho projection),
saves to data/multiview/<index>_<title>/view_00.png ... view_NN.png
"""

import struct, gzip, io, json, os, re, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import nbtlib

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
TFRECORD    = os.path.join(DATA_DIR, 'schematics_0.tfrecords')
META_FILE   = os.path.join(DATA_DIR, 'schematicsWithFinalUrl.json')
OUT_DIR     = os.path.join(DATA_DIR, 'multiview_out')
N_VIEWS     = 12        # views evenly spaced around Y axis
ELEVATION   = 30        # degrees above the equator
IMG_SIZE    = 2048      # pixels per side
BLOCK_R     = 2         # each block covers (2*BLOCK_R+1)^2 pixels → 5×5
MAX_SCHEMAS = 1000      # how many schematics to process (set lower to limit)
MAX_VOL     = 500_000   # skip schematics larger than this (too slow)
# ─────────────────────────────────────────────────────────────────────────────


# ── Minimal TFRecords reader (no TensorFlow) ─────────────────────────────────
def _read_tfrecords_raw(path):
    """Yields raw serialized Example bytes."""
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            length = struct.unpack('<Q', hdr[:8])[0]
            data = f.read(length)
            f.read(4)          # skip data crc
            if len(data) < length:
                break
            yield data


def _parse_varint(buf, pos):
    result, shift = 0, 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _parse_example(buf):
    """Parse a tf.train.Example protobuf into {field_name: bytes}."""
    features = {}
    pos = 0
    # Example { Features features = 1 }
    while pos < len(buf):
        tag_varint, pos = _parse_varint(buf, pos)
        wire = tag_varint & 0x7
        field = tag_varint >> 3
        if wire == 2:              # length-delimited
            length, pos = _parse_varint(buf, pos)
            value = buf[pos:pos + length]; pos += length
            if field == 1:         # features
                _parse_features(value, features)
        else:
            break
    return features


def _parse_features(buf, out):
    """Parse Features { map<string,Feature> feature = 1 }"""
    pos = 0
    while pos < len(buf):
        tag_varint, pos = _parse_varint(buf, pos)
        wire = tag_varint & 0x7
        field = tag_varint >> 3
        if wire == 2:
            length, pos = _parse_varint(buf, pos)
            value = buf[pos:pos + length]; pos += length
            if field == 1:         # a single map entry
                _parse_feature_entry(value, out)
        else:
            break


def _parse_feature_entry(buf, out):
    """Parse a map entry: key(string)=1, value(Feature)=2"""
    pos, key, val = 0, None, None
    while pos < len(buf):
        tag_varint, pos = _parse_varint(buf, pos)
        wire = tag_varint & 0x7
        field = tag_varint >> 3
        if wire == 2:
            length, pos = _parse_varint(buf, pos)
            value = buf[pos:pos + length]; pos += length
            if field == 1:
                key = value.decode()
            elif field == 2:
                val = value
        else:
            break
    if key and val is not None:
        # val is a Feature: bytes_list { value: bytes } is field 1 wrapping field 1
        out[key] = _extract_bytes_list(val)


def _extract_bytes_list(buf):
    """Extract first bytes value from BytesList { repeated bytes value = 1 }"""
    pos = 0
    while pos < len(buf):
        tag_varint, pos = _parse_varint(buf, pos)
        wire = tag_varint & 0x7
        field = tag_varint >> 3
        if wire == 2:
            length, pos = _parse_varint(buf, pos)
            value = buf[pos:pos + length]; pos += length
            if field == 1:         # BytesList
                return _extract_first_bytes(value)
        else:
            break
    return None


def _extract_first_bytes(buf):
    """BytesList.value[0]"""
    pos = 0
    while pos < len(buf):
        tag_varint, pos = _parse_varint(buf, pos)
        wire = tag_varint & 0x7
        field = tag_varint >> 3
        if wire == 2:
            length, pos = _parse_varint(buf, pos)
            value = buf[pos:pos + length]; pos += length
            if field == 1:
                return value
        else:
            break
    return None


def read_schematics(tfrecord_path, meta_path):
    with open(meta_path, encoding='utf-8') as f:
        meta_list = json.load(f)
    meta_index = {e['url']: e for e in meta_list}

    for raw in _read_tfrecords_raw(tfrecord_path):
        fields = _parse_example(raw)
        if not fields:
            continue
        url_bytes = fields.get('url')
        data_bytes = fields.get('schematicData')
        if not url_bytes or not data_bytes:
            continue
        url = url_bytes.decode()
        meta = meta_index.get(url, {})
        yield data_bytes, url, meta


# ── NBT / Schematic parsing ──────────────────────────────────────────────────
def parse_schematic(raw_bytes):
    """Return (W, H, L, blocks_3d) where blocks_3d[y, z, x] = block_id."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.nbt', delete=False) as tmp:
            tmp.write(raw_bytes)
            tmpname = tmp.name
        nbt = nbtlib.load(tmpname, gzipped=True, byteorder='big')
        os.unlink(tmpname)
    except Exception:
        return None

    # Root is the schematic directly (no 'Schematic' wrapper key)
    root = nbt.get('Schematic', nbt)
    if 'Width' not in root:
        return None

    W = int(root['Width'])
    H = int(root['Height'])
    L = int(root['Length'])

    if W * H * L > MAX_VOL:
        return None

    raw = root.get('Blocks')
    if raw is None:
        return None

    blocks = np.frombuffer(bytes(raw), dtype=np.int8).astype(np.int16)
    blocks[blocks < 0] += 256
    if len(blocks) < W * H * L:
        return None

    return W, H, L, blocks[:W * H * L].reshape(H, L, W)


# ── Block color palette (legacy numeric IDs → RGB) ───────────────────────────
BLOCK_COLORS = {
    0:   (0,   0,   0  ),   # air          — not rendered
    1:   (125, 125, 125),   # stone
    2:   (89,  154, 48 ),   # grass
    3:   (134, 96,  67 ),   # dirt
    4:   (115, 115, 115),   # cobblestone
    5:   (188, 152, 98 ),   # oak planks
    7:   (55,  55,  55 ),   # bedrock
    8:   (38,  92,  179),   # water
    9:   (38,  92,  179),   # still water
    10:  (207, 95,  0  ),   # lava
    11:  (207, 95,  0  ),   # still lava
    12:  (219, 211, 160),   # sand
    13:  (136, 125, 116),   # gravel
    14:  (143, 140, 60 ),   # gold ore
    15:  (136, 130, 127),   # iron ore
    16:  (100, 100, 100),   # coal ore
    17:  (102, 79,  50 ),   # log (oak)
    18:  (60,  110, 42 ),   # leaves
    20:  (175, 215, 230),   # glass
    22:  (29,  73,  163),   # lapis block
    24:  (220, 200, 140),   # sandstone
    35:  (220, 220, 220),   # wool (white)
    41:  (250, 213, 50 ),   # gold block
    42:  (200, 200, 200),   # iron block
    43:  (120, 120, 120),   # double stone slab
    44:  (120, 120, 120),   # stone slab
    45:  (160, 83,  65 ),   # brick
    46:  (180, 40,  40 ),   # TNT
    49:  (26,  12,  46 ),   # obsidian
    50:  (255, 214, 0  ),   # torch
    54:  (165, 110, 50 ),   # chest
    57:  (60,  195, 195),   # diamond block
    58:  (140, 90,  50 ),   # crafting table
    78:  (255, 255, 255),   # snow layer
    79:  (180, 210, 230),   # ice
    80:  (240, 240, 250),   # snow block
    81:  (60,  140, 40 ),   # cactus
    82:  (158, 164, 175),   # clay
    86:  (210, 115, 20 ),   # pumpkin
    87:  (100, 40,  40 ),   # netherrack
    89:  (255, 230, 160),   # glowstone
    95:  (175, 215, 230),   # stained glass
    98:  (120, 118, 125),   # stone brick
    112: (75,  40,  50 ),   # nether brick
    121: (220, 225, 185),   # end stone
    133: (50,  180, 80 ),   # emerald block
    146: (165, 110, 50 ),   # trapped chest
    155: (240, 237, 230),   # quartz block
    156: (240, 237, 230),   # quartz stairs
    159: (175, 130, 100),   # stained hardened clay
    162: (102, 79,  50 ),   # acacia/dark oak log
    165: (120, 190, 70 ),   # slime block
    168: (55,  145, 145),   # prismarine
    169: (175, 225, 220),   # sea lantern
    170: (210, 185, 40 ),   # hay bale
    172: (165, 90,  70 ),   # terracotta
    173: (20,  20,  20 ),   # coal block
    174: (180, 210, 230),   # packed ice
    179: (195, 135, 90 ),   # red sandstone
}
_DEFAULT_COLOR = (160, 120, 100)   # fallback for unknown IDs

# ── Legacy numeric ID → minecraft: name (pre-1.13, all 256 entries) ──────────
BLOCK_NAMES = {
    0:   'minecraft:air',
    1:   'minecraft:stone',
    2:   'minecraft:grass',
    3:   'minecraft:dirt',
    4:   'minecraft:cobblestone',
    5:   'minecraft:planks',
    6:   'minecraft:sapling',
    7:   'minecraft:bedrock',
    8:   'minecraft:flowing_water',
    9:   'minecraft:water',
    10:  'minecraft:flowing_lava',
    11:  'minecraft:lava',
    12:  'minecraft:sand',
    13:  'minecraft:gravel',
    14:  'minecraft:gold_ore',
    15:  'minecraft:iron_ore',
    16:  'minecraft:coal_ore',
    17:  'minecraft:log',
    18:  'minecraft:leaves',
    19:  'minecraft:sponge',
    20:  'minecraft:glass',
    21:  'minecraft:lapis_ore',
    22:  'minecraft:lapis_block',
    23:  'minecraft:dispenser',
    24:  'minecraft:sandstone',
    25:  'minecraft:noteblock',
    26:  'minecraft:bed',
    27:  'minecraft:golden_rail',
    28:  'minecraft:detector_rail',
    29:  'minecraft:sticky_piston',
    30:  'minecraft:web',
    31:  'minecraft:tallgrass',
    32:  'minecraft:deadbush',
    33:  'minecraft:piston',
    34:  'minecraft:piston_head',
    35:  'minecraft:wool',
    37:  'minecraft:yellow_flower',
    38:  'minecraft:red_flower',
    39:  'minecraft:brown_mushroom',
    40:  'minecraft:red_mushroom',
    41:  'minecraft:gold_block',
    42:  'minecraft:iron_block',
    43:  'minecraft:double_stone_slab',
    44:  'minecraft:stone_slab',
    45:  'minecraft:brick_block',
    46:  'minecraft:tnt',
    47:  'minecraft:bookshelf',
    48:  'minecraft:mossy_cobblestone',
    49:  'minecraft:obsidian',
    50:  'minecraft:torch',
    51:  'minecraft:fire',
    52:  'minecraft:mob_spawner',
    53:  'minecraft:oak_stairs',
    54:  'minecraft:chest',
    55:  'minecraft:redstone_wire',
    56:  'minecraft:diamond_ore',
    57:  'minecraft:diamond_block',
    58:  'minecraft:crafting_table',
    59:  'minecraft:wheat',
    60:  'minecraft:farmland',
    61:  'minecraft:furnace',
    62:  'minecraft:lit_furnace',
    63:  'minecraft:standing_sign',
    64:  'minecraft:wooden_door',
    65:  'minecraft:ladder',
    66:  'minecraft:rail',
    67:  'minecraft:stone_stairs',
    68:  'minecraft:wall_sign',
    69:  'minecraft:lever',
    70:  'minecraft:stone_pressure_plate',
    71:  'minecraft:iron_door',
    72:  'minecraft:wooden_pressure_plate',
    73:  'minecraft:redstone_ore',
    74:  'minecraft:lit_redstone_ore',
    75:  'minecraft:unlit_redstone_torch',
    76:  'minecraft:redstone_torch',
    77:  'minecraft:stone_button',
    78:  'minecraft:snow_layer',
    79:  'minecraft:ice',
    80:  'minecraft:snow',
    81:  'minecraft:cactus',
    82:  'minecraft:clay',
    83:  'minecraft:reeds',
    84:  'minecraft:jukebox',
    85:  'minecraft:fence',
    86:  'minecraft:pumpkin',
    87:  'minecraft:netherrack',
    88:  'minecraft:soul_sand',
    89:  'minecraft:glowstone',
    90:  'minecraft:portal',
    91:  'minecraft:lit_pumpkin',
    92:  'minecraft:cake',
    93:  'minecraft:unpowered_repeater',
    94:  'minecraft:powered_repeater',
    95:  'minecraft:stained_glass',
    96:  'minecraft:trapdoor',
    97:  'minecraft:monster_egg',
    98:  'minecraft:stonebrick',
    99:  'minecraft:brown_mushroom_block',
    100: 'minecraft:red_mushroom_block',
    101: 'minecraft:iron_bars',
    102: 'minecraft:glass_pane',
    103: 'minecraft:melon_block',
    104: 'minecraft:pumpkin_stem',
    105: 'minecraft:melon_stem',
    106: 'minecraft:vine',
    107: 'minecraft:fence_gate',
    108: 'minecraft:brick_stairs',
    109: 'minecraft:stone_brick_stairs',
    110: 'minecraft:mycelium',
    111: 'minecraft:waterlily',
    112: 'minecraft:nether_brick',
    113: 'minecraft:nether_brick_fence',
    114: 'minecraft:nether_brick_stairs',
    115: 'minecraft:nether_wart',
    116: 'minecraft:enchanting_table',
    117: 'minecraft:brewing_stand',
    118: 'minecraft:cauldron',
    119: 'minecraft:end_portal',
    120: 'minecraft:end_portal_frame',
    121: 'minecraft:end_stone',
    122: 'minecraft:dragon_egg',
    123: 'minecraft:redstone_lamp',
    124: 'minecraft:lit_redstone_lamp',
    125: 'minecraft:double_wooden_slab',
    126: 'minecraft:wooden_slab',
    127: 'minecraft:cocoa',
    128: 'minecraft:sandstone_stairs',
    129: 'minecraft:emerald_ore',
    130: 'minecraft:ender_chest',
    131: 'minecraft:tripwire_hook',
    132: 'minecraft:tripwire',
    133: 'minecraft:emerald_block',
    134: 'minecraft:spruce_stairs',
    135: 'minecraft:birch_stairs',
    136: 'minecraft:jungle_stairs',
    137: 'minecraft:command_block',
    138: 'minecraft:beacon',
    139: 'minecraft:cobblestone_wall',
    140: 'minecraft:flower_pot',
    141: 'minecraft:carrots',
    142: 'minecraft:potatoes',
    143: 'minecraft:wooden_button',
    144: 'minecraft:skull',
    145: 'minecraft:anvil',
    146: 'minecraft:trapped_chest',
    147: 'minecraft:light_weighted_pressure_plate',
    148: 'minecraft:heavy_weighted_pressure_plate',
    149: 'minecraft:unpowered_comparator',
    150: 'minecraft:powered_comparator',
    151: 'minecraft:daylight_detector',
    152: 'minecraft:redstone_block',
    153: 'minecraft:quartz_ore',
    154: 'minecraft:hopper',
    155: 'minecraft:quartz_block',
    156: 'minecraft:quartz_stairs',
    157: 'minecraft:activator_rail',
    158: 'minecraft:dropper',
    159: 'minecraft:stained_hardened_clay',
    160: 'minecraft:stained_glass_pane',
    161: 'minecraft:leaves2',
    162: 'minecraft:log2',
    163: 'minecraft:acacia_stairs',
    164: 'minecraft:dark_oak_stairs',
    165: 'minecraft:slime',
    166: 'minecraft:barrier',
    167: 'minecraft:iron_trapdoor',
    168: 'minecraft:prismarine',
    169: 'minecraft:sea_lantern',
    170: 'minecraft:hay_block',
    171: 'minecraft:carpet',
    172: 'minecraft:hardened_clay',
    173: 'minecraft:coal_block',
    174: 'minecraft:packed_ice',
    175: 'minecraft:double_plant',
    176: 'minecraft:standing_banner',
    177: 'minecraft:wall_banner',
    178: 'minecraft:daylight_detector_inverted',
    179: 'minecraft:red_sandstone',
    180: 'minecraft:red_sandstone_stairs',
    181: 'minecraft:double_stone_slab2',
    182: 'minecraft:stone_slab2',
    183: 'minecraft:spruce_fence_gate',
    184: 'minecraft:birch_fence_gate',
    185: 'minecraft:jungle_fence_gate',
    186: 'minecraft:dark_oak_fence_gate',
    187: 'minecraft:acacia_fence_gate',
    188: 'minecraft:spruce_fence',
    189: 'minecraft:birch_fence',
    190: 'minecraft:jungle_fence',
    191: 'minecraft:dark_oak_fence',
    192: 'minecraft:acacia_fence',
    193: 'minecraft:spruce_door',
    194: 'minecraft:birch_door',
    195: 'minecraft:jungle_door',
    196: 'minecraft:acacia_door',
    197: 'minecraft:dark_oak_door',
    198: 'minecraft:end_rod',
    199: 'minecraft:chorus_plant',
    200: 'minecraft:chorus_flower',
    201: 'minecraft:purpur_block',
    202: 'minecraft:purpur_pillar',
    203: 'minecraft:purpur_stairs',
    204: 'minecraft:purpur_double_slab',
    205: 'minecraft:purpur_slab',
    206: 'minecraft:end_bricks',
    207: 'minecraft:beetroots',
    208: 'minecraft:grass_path',
    209: 'minecraft:end_gateway',
    210: 'minecraft:repeating_command_block',
    211: 'minecraft:chain_command_block',
    212: 'minecraft:frosted_ice',
    213: 'minecraft:magma',
    214: 'minecraft:nether_wart_block',
    215: 'minecraft:red_nether_brick',
    216: 'minecraft:bone_block',
    217: 'minecraft:structure_void',
    218: 'minecraft:observer',
    219: 'minecraft:white_shulker_box',
    220: 'minecraft:orange_shulker_box',
    221: 'minecraft:magenta_shulker_box',
    222: 'minecraft:light_blue_shulker_box',
    223: 'minecraft:yellow_shulker_box',
    224: 'minecraft:lime_shulker_box',
    225: 'minecraft:pink_shulker_box',
    226: 'minecraft:gray_shulker_box',
    227: 'minecraft:silver_shulker_box',
    228: 'minecraft:cyan_shulker_box',
    229: 'minecraft:purple_shulker_box',
    230: 'minecraft:blue_shulker_box',
    231: 'minecraft:brown_shulker_box',
    232: 'minecraft:green_shulker_box',
    233: 'minecraft:red_shulker_box',
    234: 'minecraft:black_shulker_box',
    235: 'minecraft:white_glazed_terracotta',
    236: 'minecraft:orange_glazed_terracotta',
    237: 'minecraft:magenta_glazed_terracotta',
    238: 'minecraft:light_blue_glazed_terracotta',
    239: 'minecraft:yellow_glazed_terracotta',
    240: 'minecraft:lime_glazed_terracotta',
    241: 'minecraft:pink_glazed_terracotta',
    242: 'minecraft:gray_glazed_terracotta',
    243: 'minecraft:silver_glazed_terracotta',
    244: 'minecraft:cyan_glazed_terracotta',
    245: 'minecraft:purple_glazed_terracotta',
    246: 'minecraft:blue_glazed_terracotta',
    247: 'minecraft:brown_glazed_terracotta',
    248: 'minecraft:green_glazed_terracotta',
    249: 'minecraft:red_glazed_terracotta',
    250: 'minecraft:black_glazed_terracotta',
    251: 'minecraft:concrete',
    252: 'minecraft:concrete_powder',
    255: 'minecraft:structure_block',
}

def block_name(block_id):
    """Return minecraft: name for a legacy numeric ID."""
    return BLOCK_NAMES.get(int(block_id), f'minecraft:unknown_{block_id}')

def _block_color(block_id):
    return np.array(BLOCK_COLORS.get(int(block_id), _DEFAULT_COLOR), dtype=np.float32)


# ── Multi-view renderer ──────────────────────────────────────────────────────
def _surface_mask(occ):
    """Blocks with at least one air neighbor (visible surface only)."""
    s = np.zeros_like(occ)
    s[1:,  :,  :] |= ~occ[:-1, :,  :]
    s[:-1, :,  :] |= ~occ[1:,  :,  :]
    s[:,  1:,  :] |= ~occ[:,  :-1, :]
    s[:, :-1,  :] |= ~occ[:,  1:,  :]
    s[:,  :,  1:] |= ~occ[:,  :, :-1]
    s[:,  :, :-1] |= ~occ[:,  :,  1:]
    return occ & s


TOP_SHADE   = 1.00
RIGHT_SHADE = 0.75
LEFT_SHADE  = 0.55

def _shade_color(color, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


# ── Per-face colors: (top_rgb, side_rgb)  None = use BLOCK_COLORS for both ────
# Lets grass show green top + brown sides, log show ring top + bark sides, etc.
BLOCK_FACE_COLORS = {
    2:   ((100, 170,  50), (134,  96,  67)),   # grass: green top, dirt side
    3:   ((134,  96,  67), (134,  96,  67)),   # dirt
    17:  ((110,  90,  55), (102,  79,  50)),   # oak log: ring top, bark side
    162: ((110,  90,  55), ( 80,  55,  35)),   # dark oak log
    18:  (( 55, 100,  40), ( 55, 100,  40)),   # leaves
    161: (( 70, 110,  35), ( 70, 110,  35)),   # leaves2 (acacia/dark oak)
    87:  ((120,  55,  55), (100,  40,  40)),   # netherrack
    110: (( 90,  40,  90), ( 80,  80,  80)),   # mycelium: purple top, grey side
    86:  ((210, 140,  20), (140,  90,  20)),   # pumpkin: face side, orange top
    91:  ((210, 140,  20), (140,  90,  20)),   # lit pumpkin
    2:   ((100, 170,  50), (134,  96,  67)),   # grass (duplicate key ignored, Python keeps last)
}

def _block_face_colors(block_id):
    """Return (top_rgb, side_rgb) for a block, respecting per-face differences."""
    bid = int(block_id)
    if bid in BLOCK_FACE_COLORS:
        return BLOCK_FACE_COLORS[bid]
    base = BLOCK_COLORS.get(bid, _DEFAULT_COLOR)
    return base, base


def _make_sky_gradient(size):
    """Vertical gradient: light sky blue at top → deeper blue at bottom."""
    top    = np.array([160, 210, 250], dtype=np.float32)
    bottom = np.array([ 80, 140, 200], dtype=np.float32)
    t = np.linspace(0, 1, size, dtype=np.float32)[:, np.newaxis]   # (H,1)
    row = (top + (bottom - top) * t).astype(np.uint8)              # (H,3)
    arr = np.broadcast_to(row[:, np.newaxis, :], (size, size, 3)).copy()
    return Image.fromarray(arr, 'RGB')


def _apply_vignette(img):
    """Darken corners to draw focus to center."""
    size = img.width
    cx = cy = size / 2
    arr = np.array(img, dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
    vignette = np.clip(1.0 - dist * 0.35, 0.55, 1.0)[..., np.newaxis]
    arr = np.clip(arr * vignette, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _compute_ao(occ):
    """Per-block AO: count solid neighbors → return float array [0..1], 1=no occlusion."""
    H, L, W = occ.shape
    neighbor_count = np.zeros((H, L, W), dtype=np.float32)
    for dy, dz, dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        slc_src  = [slice(max(0,-dy), H+min(0,-dy)),
                    slice(max(0,-dz), L+min(0,-dz)),
                    slice(max(0,-dx), W+min(0,-dx))]
        slc_dst  = [slice(max(0, dy), H+min(0, dy)),
                    slice(max(0, dz), L+min(0, dz)),
                    slice(max(0, dx), W+min(0, dx))]
        neighbor_count[tuple(slc_dst)] += occ[tuple(slc_src)].astype(np.float32)
    # AO factor: 0 neighbors solid → 1.0 (full brightness), 6 solid → 0.5 (darkest)
    return 1.0 - neighbor_count / 6.0 * 0.5


def render_view(blocks_3d, angle_deg, elevation_deg=ELEVATION, size=IMG_SIZE, **_):
    """
    Isometric cube renderer — draws 3 faces per block (top/left/right)
    with Minecraft-style directional lighting. Rotate azimuth for multi-view.
    """
    H, L, W = blocks_3d.shape
    occ = blocks_3d != 0

    surf = _surface_mask(occ)
    ao   = _compute_ao(occ)          # (H, L, W) float, 0.5..1.0
    ys, zs, xs = np.where(surf)
    if len(xs) == 0:
        return Image.new('RGB', (size, size))

    block_ids = blocks_3d[ys, zs, xs]
    ao_vals   = ao[ys, zs, xs]      # per-block AO factor

    # Rotate block positions around Y axis
    cx, cy, cz = W / 2.0, H / 2.0, L / 2.0
    bx = (xs - cx).astype(np.float32)
    by = (ys - cy).astype(np.float32)
    bz = (zs - cz).astype(np.float32)

    az  = np.radians(angle_deg)
    rx  =  np.cos(az) * bx + np.sin(az) * bz
    ry  =  by
    rz  = -np.sin(az) * bx + np.cos(az) * bz

    # Render at 2× internal resolution then downscale → crisp anti-aliasing
    render_size = size * 2
    margin = 40

    iso_x_raw = (rx - rz)
    iso_y_raw = (rx + rz) * 0.5 - ry

    x_min, x_max = iso_x_raw.min(), iso_x_raw.max()
    y_min, y_max = iso_y_raw.min(), iso_y_raw.max()

    x_span = max(x_max - x_min + 2, 1.0)
    y_span = max(y_max - y_min + 2, 1.0)

    usable = render_size - margin * 2
    s = max(1, int(min(usable / x_span, usable / y_span)))

    cx_off = render_size / 2 - ((x_min + x_max) / 2) * s
    cy_off = render_size / 2 - ((y_min + y_max) / 2) * s - s // 2

    iso_x = iso_x_raw * s + cx_off
    iso_y = iso_y_raw * s + cy_off

    # Painter's order: back → front (depth = rx + rz)
    depth = rx + rz + ry * 0.001
    order = np.argsort(depth)

    # Sky gradient background
    img  = _make_sky_gradient(render_size)
    draw = ImageDraw.Draw(img)

    outline_w = max(1, s // 8)

    for i in order:
        ox, oy   = float(iso_x[i]), float(iso_y[i])
        ao_f     = float(ao_vals[i])
        top_rgb, side_rgb = _block_face_colors(block_ids[i])

        c_top   = _shade_color(top_rgb,  TOP_SHADE   * ao_f)
        c_right = _shade_color(side_rgb, RIGHT_SHADE * ao_f)
        c_left  = _shade_color(side_rgb, LEFT_SHADE  * ao_f)
        e_top   = _shade_color(top_rgb,  0.45 * ao_f)
        e_side  = _shade_color(side_rgb, 0.35 * ao_f)

        left_pts = [
            (ox - s, oy),
            (ox,     oy + s // 2),
            (ox,     oy + s // 2 + s),
            (ox - s, oy + s),
        ]
        right_pts = [
            (ox,     oy + s // 2),
            (ox + s, oy),
            (ox + s, oy + s),
            (ox,     oy + s // 2 + s),
        ]
        top_pts = [
            (ox,     oy - s // 2),
            (ox + s, oy),
            (ox,     oy + s // 2),
            (ox - s, oy),
        ]

        draw.polygon(left_pts,  fill=c_left)
        draw.polygon(right_pts, fill=c_right)
        draw.polygon(top_pts,   fill=c_top)

        if s >= 4:
            draw.line(left_pts  + [left_pts[0]],  fill=e_side, width=outline_w)
            draw.line(right_pts + [right_pts[0]], fill=e_side, width=outline_w)
            draw.line(top_pts   + [top_pts[0]],   fill=e_top,  width=outline_w)

    img = img.resize((size, size), Image.LANCZOS)
    return img


def render_multiview(blocks_3d, n_views=N_VIEWS):
    angles = np.linspace(0, 360, n_views, endpoint=False)
    return [render_view(blocks_3d, a) for a in angles]


# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_name(s):
    return re.sub(r'[^\w\-]', '_', s)[:40]


def make_grid(images, cols=4):
    """Combine views into a single grid image (RGB)."""
    n = len(images)
    rows = (n + cols - 1) // cols
    W, H = images[0].size
    grid = Image.new('RGB', (cols * W, rows * H), color=(0, 0, 0))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img.convert('RGB'), (c * W, r * H))
    return grid


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Reading {TFRECORD}')
    print(f'Output  → {OUT_DIR}')
    print(f'Views per schematic: {N_VIEWS}  |  Image size: {IMG_SIZE}px\n')

    done = 0
    for raw, url, meta in read_schematics(TFRECORD, META_FILE):
        result = parse_schematic(raw)
        if result is None:
            continue

        W, H, L, blocks_3d = result
        title = meta.get('title', f'schematic_{done}')
        tags  = ', '.join(meta.get('tags') or []) or '—'
        print(f'[{done+1:02d}] "{title}"  {W}×{H}×{L}  tags: {tags}')

        # Block inventory: count each non-air ID → show as minecraft: names
        ids, counts = np.unique(blocks_3d[blocks_3d != 0], return_counts=True)
        inventory = sorted(zip(counts, ids), reverse=True)
        inv_str = '  '.join(f'{block_name(bid)}×{cnt}' for cnt, bid in inventory[:8])
        if len(inventory) > 8:
            inv_str += f'  (+{len(inventory)-8} more)'
        print(f'       blocks: {inv_str}')

        folder = os.path.join(OUT_DIR, f'{done+1:02d}_{safe_name(title)}')
        os.makedirs(folder, exist_ok=True)

        # Save block inventory as blocks.json  { "minecraft:name": count }
        block_inv = {block_name(int(bid)): int(cnt) for cnt, bid in inventory}
        with open(os.path.join(folder, 'blocks.json'), 'w') as f:
            json.dump(block_inv, f, indent=2)

        views = render_multiview(blocks_3d)

        # Save individual views
        for i, img in enumerate(views):
            img.save(os.path.join(folder, f'view_{i:02d}.png'))

        # Save grid overview
        grid = make_grid(views)
        grid.save(os.path.join(folder, 'grid.png'))

        done += 1
        if done >= MAX_SCHEMAS:
            break

    print(f'\nDone. {done} schematics → {OUT_DIR}')


if __name__ == '__main__':
    main()
