from pravaala.translit import (deva_to_slp1, iast_to_slp1, slp1_to_iast,
                              slp1_to_wx, wx_to_slp1, detect_scheme)


def test_wx_slp1_is_bijection():
    # SLP1 alphabet minus 'X' (long vocalic l), which has no 1-letter WX code
    alphabet = "aAiIuUfFxeEoOMH~kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh"
    assert wx_to_slp1(slp1_to_wx(alphabet)) == alphabet
    wx_alphabet = slp1_to_wx(alphabet)
    assert len(set(wx_alphabet)) == len(alphabet)          # injective
    assert slp1_to_wx(wx_to_slp1(wx_alphabet)) == wx_alphabet


def test_wx_examples():
    assert wx_to_slp1("amqwa") == "amfta"        # amṛta
    assert wx_to_slp1("paxmA") == "padmA"        # padmā
    assert wx_to_slp1("wAnwrika") == "tAntrika"  # tāntrika


def test_iast():
    assert iast_to_slp1("maharṣayaḥ") == "maharzayaH"
    assert iast_to_slp1("śirorogavijñānīyaṃ") == "SirorogavijYAnIyaM"
    assert iast_to_slp1("aiśvarya") == "ESvarya"
    assert iast_to_slp1("kṛṣṇa") == "kfzRa"


def test_iast_roundtrip():
    for w in ["kfzRa", "ESvarya", "SirorogavijYAnIyaM", "praUga"]:
        assert iast_to_slp1(slp1_to_iast(w)) == w


def test_devanagari():
    assert deva_to_slp1("स्वर्ग") == "svarga"
    assert deva_to_slp1("त्रिदशालय") == "tridaSAlaya"
    assert deva_to_slp1("पञ्चशर") == "paYcaSara"
    assert deva_to_slp1("अनङ्ग") == "anaNga"
    assert deva_to_slp1("कृष्णः") == "kfzRaH"


def test_detect():
    assert detect_scheme(["paxmA", "amqwa", "xeva"])["guess"] == "WX"
    assert detect_scheme(["kfzRa", "puruza", "vizRu"])["guess"] == "SLP1"
