from pravaala.normalise import CompoundLexicon, canon_nasal, join, participle_stem


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


def test_respeller_bridges_conventions():
    from pravaala.normalise import Respeller
    r = Respeller({"kadAcid": 10, "Bogin": 5, "pattra": 7, "Cattra": 3, "oM": 9, "jala": 50})
    assert r("kadAcit") == "kadAcid" and r("BogI") == "Bogin" and r("om") == "oM"
    assert r("patra") == "pattra" and r("Catra") == "Cattra"
    assert r("jala") == "jala" and r("nadI") is None


def test_space_respelling_is_off_by_default():
    import numpy as np
    from pravaala.embed import Space
    sp = Space(["kadAcid", "jala"], np.eye(2), np.array([3.0, 4.0]))
    assert "kadAcit" not in sp
    sp.use_respelling()
    assert "kadAcit" in sp and sp.sim("kadAcit", "kadAcid") == 1.0


def test_gemination_only_in_optional_contexts():
    from pravaala.normalise import gemination_key as g
    assert g("pattra") == g("patra") and g("vArtta") == g("vArta") and g("Cattra") == g("Catra")
    assert g("mUrcCita") == g("mUrCita")
    assert g("kuwwaka") != g("kuwaka") and g("appati") != g("apati")
