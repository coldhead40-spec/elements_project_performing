# TouchDesigner: paste into a Text DAT and Run Script ONCE (re-running is safe: it updates what it made).
#
# Builds, next to your existing nodes:
#   pose_logic   Execute DAT  - the whole experience (idle -> pose 1..4 -> finale), reads oscin1
#   reveal_glsl  Text DAT     + reveal GLSL TOP - grey painting, each element's zone awakens as its pose is held
#   pose_text    Text TOP     - the on-screen messages
#   pose_comp    Composite TOP (Add) + pose_out Null TOP - reveal + your shadow + the avatar + text  -> out1
#   snd_*        Audio File In CHOPs -> snd_mix -> snd_out - one loop per element, volume = how awake it is
#
# Needs your existing: oscin1 (OSC In CHOP, port 9000), switch1 (avatar switch), level_shadow1 (avatar level),
#   level_shadow (YOUR shadow from camera 1), mono1 + water_vid/earth_vid/air_vid/fire_vid, out1.
# Camera roles:  videodevin1 = camera 1 (shadow, stays in TouchDesigner)
#                camera 2    = python main_poses.py --camera N  (pose analysis)  -> OSC on port 9000
net = parent()

# ---------------- edit to match your network ----------------
BASE          = 'mono1'                    # the grey painting
VIDEOS        = {'water': 'water_vid', 'earth': 'earth_vid', 'air': 'air_vid', 'fire': 'fire_vid'}
AVATAR_LEVEL  = 'level_shadow1'            # Level TOP after the avatar switch (opacity is driven by the logic)
USER_LEVEL    = 'level_shadow'             # Level TOP at the end of the camera-1 shadow chain
AVATAR_SWITCH = 'switch1'
SOUND_DIR     = 'sounds'                   # folder next to the .toe with water.wav earth.wav air.wav fire.wav
CONNECT_OUT   = True                       # wire pose_out into out1 (set False to do it by hand)
# ------------------------------------------------------------

GLSL = r'''
// inputs: 0 grey painting, 1 water_vid, 2 earth_vid, 3 air_vid, 4 fire_vid
uniform vec4  uProg;     // water earth air fire   0..1  (how far each pose has been held)
uniform float uFull;     // 0..1 finale: whole painting in colour
out vec4 fragColor;

// zones: xy = centre in UV (origin bottom-left), z = radius in picture heights.  TUNE THESE to your painting.
const vec3 Z_WATER = vec3(0.12, 0.45, 0.45);
const vec3 Z_EARTH = vec3(0.50, 0.12, 0.45);
const vec3 Z_AIR   = vec3(0.55, 0.88, 0.42);
const vec3 Z_FIRE  = vec3(0.82, 0.38, 0.40);

float zone(vec2 uv, vec3 z, float p, float asp) {
    float r = z.z * (0.10 + 1.00 * p);                       // the zone grows while the pose is held
    float d = length((uv - z.xy) * vec2(asp, 1.0));
    float m = 1.0 - smoothstep(r * 0.45, r, d);              // soft edge
    return m * smoothstep(0.0, 0.08, p) * pow(p, 0.6);       // and fades in
}

void main() {
    vec2 uv = vUV.st;
    float asp = uTDOutputInfo.res.z / uTDOutputInfo.res.w;
    vec4 base = texture(sTD2DInputs[0], uv);
    vec4 w = texture(sTD2DInputs[1], uv);
    vec4 e = texture(sTD2DInputs[2], uv);
    vec4 a = texture(sTD2DInputs[3], uv);
    vec4 f = texture(sTD2DInputs[4], uv);
    vec4 col = base;
    col = mix(col, w, zone(uv, Z_WATER, uProg.x, asp));
    col = mix(col, e, zone(uv, Z_EARTH, uProg.y, asp));
    col = mix(col, a, zone(uv, Z_AIR,   uProg.z, asp));
    col = mix(col, f, zone(uv, Z_FIRE,  uProg.w, asp));
    col = mix(col, 0.25 * (w + e + a + f), smoothstep(0.0, 1.0, uFull));
    fragColor = TDOutputSwizzle(vec4(col.rgb, 1.0));
}
'''

LOGIC = r'''
# ---------------- edit these ----------------
ORDER        = ['water', 'earth', 'air', 'fire']                    # storyboard order
AVATAR_INDEX = {'water': 0, 'earth': 1, 'air': 2, 'fire': 3}        # input ORDER of the avatar switch
SWITCH       = 'switch1'
MATCH_ON, MATCH_OFF = 0.55, 0.40   # match score to start / stop counting the hold (hysteresis)
NEED_STEADY  = {'earth': 0.4}      # earth = standing still like a mountain: only counts while the body is steady (0..1)
HOLD         = 6.0     # s the pose must be held (cumulatively) to fully awaken the zone
DECAY        = 0.5     # progress lost per second of not holding, relative to the build-up speed
START_AFTER  = 1.5     # s a person must be in view before the first avatar appears
LEAVE_AFTER  = 6.0     # s without a person -> back to the start
SETTLE       = 3.0     # s after a pose is completed (zone keeps blooming) before the next avatar
POSE_TIMEOUT = 45.0    # s without success -> move on to the next pose (progress is kept)
FINALE_HOLD  = 25.0    # s the finished painting stays before restarting
FADE, RESET_FADE, FINALE_RAMP = 1.0, 3.0, 4.0
T_IDLE   = "Step in front of the painting"
T_POSE   = "Follow the shadow"
T_HOLD   = "Follow the shadow\nHold the pose..."
T_BACK   = "Step back until your whole body is visible"
T_DONE   = "{} awakens"
T_FINALE = "You brought the painting to life"
T_PARTIAL = "Come back to awaken the rest"
# --------------------------------------------

S = {'state': None, 'k': 0, 'prev': None, 't0': 0.0, 'seen': 0.0, 'absent': 0.0, 'holding': False,
     'prog': {e: 0.0 for e in ORDER}, 'done': set(), 'avatar': 0.0, 'user': 0.0, 'full': 0.0, 'txt': None}

def val(name):
    c = op('oscin1')[name]
    return float(c.eval()) if c is not None else 0.0

def say(text):
    if text != S['txt']:
        S['txt'] = text
        op('pose_text').par.text = text

def goto(state):
    S['state'], S['t0'], S['holding'] = state, absTime.seconds, False
    if state == 'idle':
        S['k'], S['seen'] = 0, 0.0
        S['done'] = set()
    print('pose_logic ->', state, ORDER[S['k']] if S['k'] < len(ORDER) else '')

def next_pose():
    S['k'] += 1
    goto('finale' if S['k'] >= len(ORDER) else 'pose')

def complete(element):
    """Test without the camera:  op('pose_logic').module.complete('water')"""
    S['k'] = ORDER.index(element)
    S['prog'][element] = 1.0
    S['done'].add(element)
    goto('settle')

def reset():
    goto('idle')

def approach(cur, target, step):
    return cur + max(-step, min(step, target - cur))

def onFrameEnd(frame):
    now = absTime.seconds
    if S['state'] is None:
        S['prev'] = now
        goto('idle')
    dt = min(now - S['prev'], 0.2)
    S['prev'] = now

    present = val('present') > 0.5
    S['absent'] = 0.0 if present else S['absent'] + dt
    st = S['state']
    el = ORDER[S['k']] if S['k'] < len(ORDER) else None

    if st == 'idle':
        say(T_IDLE)
        S['seen'] = S['seen'] + dt if present else 0.0
        for e in ORDER:                                   # previous visitor's colours fade away
            S['prog'][e] = max(0.0, S['prog'][e] - dt / RESET_FADE)
        S['full'] = max(0.0, S['full'] - dt / RESET_FADE)
        if S['seen'] >= START_AFTER and sum(S['prog'].values()) + S['full'] < 0.05:
            goto('pose')

    elif st == 'pose':
        if S['absent'] >= LEAVE_AFTER:
            goto('idle')
        else:
            m = val('match_' + el)
            if m >= MATCH_ON:
                S['holding'] = True
            elif m < MATCH_OFF:
                S['holding'] = False
            if S['holding'] and val('steady') < NEED_STEADY.get(el, 0.0):
                S['holding'] = False
            if S['holding']:
                S['prog'][el] = min(1.0, S['prog'][el] + dt / HOLD)
            else:
                S['prog'][el] = max(0.0, S['prog'][el] - dt * DECAY / HOLD)
            if present and val('full_body') < 0.5:
                say(T_BACK)
            else:
                say(T_HOLD if S['holding'] else T_POSE)
            if S['prog'][el] >= 1.0:
                S['done'].add(el)
                goto('settle')
            elif now - S['t0'] >= POSE_TIMEOUT:
                next_pose()

    elif st == 'settle':
        say(T_DONE.format(ORDER[S['k']].capitalize()))
        if S['absent'] >= LEAVE_AFTER:
            goto('idle')
        elif now - S['t0'] >= SETTLE:
            next_pose()

    elif st == 'finale':
        everything = len(S['done']) == len(ORDER)
        if everything:
            S['full'] = min(1.0, S['full'] + dt / FINALE_RAMP)
        say(T_FINALE if everything else T_PARTIAL)
        if now - S['t0'] >= FINALE_HOLD or S['absent'] >= LEAVE_AFTER:
            goto('idle')

    # fades
    S['avatar'] = approach(S['avatar'], 1.0 if S['state'] == 'pose' else 0.0, dt / FADE)
    S['user'] = approach(S['user'], 1.0 if present else 0.0, dt / FADE)
    for e in ORDER:
        me.store('p_' + e, S['prog'][e])
    me.store('full', S['full'])
    me.store('avatar_alpha', S['avatar'])
    me.store('user_alpha', S['user'])

    # which avatar is shown
    if S['k'] < len(ORDER):
        idx = AVATAR_INDEX[ORDER[S['k']]]
        sw = op(SWITCH)
        if sw is not None and sw.par.index.eval() != idx:
            sw.par.index = idx
    return
'''


# ---------------------------------------------------------------------------------------
def make(kind, name, x, y):
    o = net.op(name) or net.create(kind, name)
    o.nodeX, o.nodeY = x, y
    return o


def tryset(o, names, value, expr=False):
    for n in names:
        p = getattr(o.par, n, None)
        if p is not None:
            if expr:
                p.expr = value
                p.mode = ParMode.EXPRESSION
            else:
                p.mode = ParMode.CONSTANT
                p.val = value
            return True
    print('!! could not set', o.name, names, '- set it by hand')
    return False


def fetch(key):
    return f"op('pose_logic').fetch('{key}', 0)"


X0, Y0 = 1600, -400   # where the new nodes are placed (scroll right of your network)

# 1. logic (and retire the movement-quality experience_logic from experience_setup.py if present)
old = net.op('experience_logic')
if old is not None:
    old.par.active = False
logic = make(executeDAT, 'pose_logic', X0, Y0 + 300)
logic.text = LOGIC
logic.par.active = True
tryset(logic, ['frameend'], True)       # if this prints '!!': tick "Frame End" on the DAT by hand

# 2. reveal shader
glsl_dat = make(textDAT, 'reveal_glsl', X0, Y0 + 150)
glsl_dat.text = GLSL
reveal = make(glslTOP, 'reveal', X0 + 200, Y0 + 150)
tryset(reveal, ['pixeldat'], glsl_dat.path)
tryset(reveal, ['vec0name'], 'uProg')
for comp, key in zip('xyzw', ['p_water', 'p_earth', 'p_air', 'p_fire']):
    tryset(reveal, ['vec0value' + comp], fetch(key), expr=True)
tryset(reveal, ['vec1name'], 'uFull')
tryset(reveal, ['vec1valuex'], fetch('full'), expr=True)
for i, src in enumerate([BASE] + [VIDEOS[e] for e in ('water', 'earth', 'air', 'fire')]):
    if net.op(src) is None:
        print('!! missing', src, '- connect input', i, 'of reveal by hand')
    else:
        reveal.inputConnectors[i].connect(net.op(src))

# 3. text + final composite
text = make(textTOP, 'pose_text', X0 + 200, Y0)
tryset(text, ['fontsizex'], 48)
tryset(text, ['alignx'], 'center')
tryset(text, ['aligny'], 'top')
tryset(text, ['fontcolorr'], 1)
tryset(text, ['fontcolorg'], 1)
tryset(text, ['fontcolorb'], 1)
comp = make(compositeTOP, 'pose_comp', X0 + 400, Y0 + 150)
tryset(comp, ['operand'], 'add')        # add = order independent, works for silhouettes on black; try 'screen' if too bright
for i, src in enumerate([reveal, net.op(USER_LEVEL), net.op(AVATAR_LEVEL), text]):
    if src is None:
        print('!! missing level node, connect pose_comp input', i, 'by hand')
    else:
        comp.inputConnectors[0].connect(src)
if len(comp.inputs) != 4:
    print('!! pose_comp has', len(comp.inputs), 'inputs - wire reveal, level_shadow, level_shadow1 and pose_text into it by hand')
out = make(nullTOP, 'pose_out', X0 + 600, Y0 + 150)
out.inputConnectors[0].connect(comp)
if CONNECT_OUT and net.op('out1') is not None:
    net.op('out1').inputConnectors[0].connect(out)     # to revert: reconnect over_final -> out1

# 4. Level opacities now come from the logic
for name, key in ((AVATAR_LEVEL, 'avatar_alpha'), (USER_LEVEL, 'user_alpha')):
    lv = net.op(name)
    if lv is not None:
        tryset(lv, ['opacity'], fetch(key), expr=True)

# 5. sound: one loop per element, volume follows how awake the zone is
mix = make(mathCHOP, 'snd_mix', X0 + 200, Y0 - 400)
tryset(mix, ['chanop'], 'add')
for n, el in enumerate(('water', 'earth', 'air', 'fire')):
    s = make(audiofileinCHOP, 'snd_' + el, X0, Y0 - 250 - n * 100)
    tryset(s, ['file'], project.folder + '/' + SOUND_DIR + '/' + el + '.wav')
    tryset(s, ['play'], True)
    tryset(s, ['repeat'], 'on')
    tryset(s, ['volume'], fetch('p_' + el), expr=True)
    mix.inputConnectors[0].connect(s)       # Math CHOP accepts several inputs; each connect() adds one
if len(mix.inputs) != 4:
    print('!! snd_mix has', len(mix.inputs), 'inputs - wire the four snd_* CHOPs into it by hand')
snd_out = make(audiodeviceoutCHOP, 'snd_out', X0 + 400, Y0 - 400)
snd_out.inputConnectors[0].connect(mix)

print('pose_logic ready. oscin1 channels:', [c.name for c in net.op('oscin1').chans()][:12])
print('test without the camera: op("pose_logic").module.complete("water")   /   .reset()')
