# Checkpoint samples

These are raw 200-character continuations. Both prompts use seed 2026 and temperature 0.8 at every checkpoint.
The replacement character `�` represents the reserved unknown token.

| Optimizer updates | Training loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.2767 | 4.2801 |
| 100 | 2.3503 | 2.3715 |
| 1,200 | 0.5358 | 0.8381 |
| 2,400 | 0.2042 | 0.7365 |

The final test loss was 0.7059 nats per character. The training and evaluation run took 97.1 seconds on CPU.

## Training venue prompt

Prompt

```text
Venue: Alder House
Shift: Early
Topic: Booking request
Detail: 
```

### Step 0

```text
4GAcKBBew8�gG4j-iEx40c:Nc2Ohhs 3cL0yHF3r1rfR
x0uplBSpnRvwPdKP2Svkjcxp802rO4OC56
:jClAxx:ps1
.4Hzu4pTsP1 S6J3yFgt
ist6EFCzaIH6u
Nfsp4jsPEOMEktK4I0o92fK3
I2H:j
tK3p9C0R-RjgbF�zadW
�6J 692om5SW48bSdxO38w
```

### Step 100

```text
oircheseck guegeiextac: cilafifoome Hooror steduplal: the thet: Cacapestopthes: Chilorend 1ingenuepind thesty g 1istange athetepist tafthilite c:sk2eetand rictather the ge thedin fthere mendanbtathili
```

### Step 1,200

```text
A guest asked gried for ceire at 12 in the sugest ar gro 6.
Vext Din Dan To ean.
Dextail: Din Din chals to a grey ana grelymbe in the folis the pannft 10: Theck number aft leading in the shif brendier
```

### Step 2,400

```text
A guest asked for a tablcount ion the drawing room just with fleft desk rister.
Next step: Check the route.


Venue: Biry Inn
Shift: Lunch
Topic: Suppplie are
Detail: Rooms
Shift: Early
Topic: Weather
```

## Validation venue prompt

Prompt

```text
Venue: Spruce House
Shift: Late
Topic: Lost property
Detail: 
```

### Step 0

```text
4GAcdBBeA8IgGSg-iEME0M:Nc28dDL kcs0yHDob:rcsfAduAlMSvgRvwPtKP3SvBjcxp802rO4OC56
:jClAxx:ps1
.4Hzu4pTsP1 S6J3yFgt
ist6EFCzaIH6u
Nfsp4jsPEOMEktK4I0o92fK3
Ib �C.tK3p9C0R-RjgbF�zadW
�6J 692om5SW48bSdxO38w
```

### Step 100

```text
ifrche e nnguegiiextat: catafickcen Hoorinchere Alal: teng Kethe acapestopthes: Chilorend 1.
She tis Pfthesty gt istange athetepist tafthilite c:sk2eetand rictather thange thedin fthere mendanbtathile
```

### Step 1,200

```text
A rcaller acasked for a caller to 12 in the fle aler the guest vior pth parize t 16.

Statatus: us: The drity the shift the nuests nug boxes nup arter 10:30.
Status: Ther oreand has ofooprt and ats ro
```

### Step 2,400

```text
A party of guest lawn.
Status: The kitchen has the question arer and call the guest with dut manager before calling back.
Next step: Check the cloakroom at 12: the ribe lack.


Venue: Junipde Lodge
Sh
```
