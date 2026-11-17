from vidruma.normalise import CompoundLexicon, canon_nasal, join, participle_stem


def test_participle_stems():
    cases = {("gatAH", "Case=Nom|Number=Plur|VerbForm=Part"): "gata",
             ("gatA", "Case=Nom|Gender=Fem|VerbForm=Part"): "gata",
             ("pibantaH", "Case=Nom|Number=Plur|Tense=Pres|VerbForm=Part"): "pibat",
             ("pibantI", "Case=Nom|Gender=Fem|Tense=Pres|VerbForm=Part"): "pibat",
             ("kurvARaH", "Case=Nom|Tense=Pres|VerbForm=Part"): "kurvARa",
             ("gatavAn", "Case=Nom|VerbForm=Part"): "gatavat",
             ("kartavyam", "Case=Nom|VerbForm=Gdv"): "kartavya",
             ("duzwa", "Case=Cpd|VerbForm=Part"): "duzwa",
             ("sati", "Case=Loc|Tense=Pres|VerbForm=Part"): "sat",
             ("yAntaH", "Case=Nom|Tense=Pres|VerbForm=Part"): "yAt",
             ("dIdyatam", "Case=Acc|Tense=Pres|VerbForm=Part"): "dIdyat",
             ("izayantIH", "Case=Acc|Tense=Pres|VerbForm=Part"): "izayat",
             ("dabDena", "Case=Ins|Tense=Past|VerbForm=Part"): "dabDa",
             ("vartamAnasya", "Case=Gen|Tense=Pres|VerbForm=Part"): "vartamAna"}
    for (f, ft), st in cases.items():
        assert participle_stem(f, ft) == st, (f, participle_stem(f, ft))


def test_nasal_and_join():
    assert canon_nasal("saMDi") == "sanDi" and canon_nasal("SaMkara") == "SaNkara"
    assert canon_nasal("saMsAra") == "saMsAra"          # before a sibilant: unchanged
    assert join("padma", "Akara") == "padmAkara" and join("jagat", "guru") == "jagadguru"
    assert join("manas", "raTa") == "manoraTa"


def test_compound_lexicon_fuses_only_associated_pairs():
    lx = CompoundLexicon(min_count=3, theta=0.5)
    for _ in range(20):
        lx.add("mahat", "rAja")
    for x in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        lx.add("sarva", x); lx.add(x, "jana")
    assert lx.fuse("mahat", "rAja") and not lx.fuse("sarva", "a")
