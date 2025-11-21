# स्थापना गाइड

यह गाइड सभी स्थापना विधियों और वैकल्पिक निर्भरताओं को कवर करती है।

## आवश्यकताएँ

- **Python:** 3.11 या उच्चतर
- **ऑपरेटिंग सिस्टम:** Linux, macOS, Windows
- **API कुंजी:** OpenAI, Anthropic, GigaChat, या Yandex

## बुनियादी स्थापना

### pip का उपयोग करना (अनुशंसित)

```bash
pip install kttc
```

यह कोर निर्भरताओं (~50MB) को स्थापित करता है:
- CLI इंटरफ़ेस
- बहु-एजेंट QA सिस्टम
- रूसी NLP (MAWO लाइब्रेरीज़)
- बुनियादी बहु-भाषा समर्थन (spaCy, jieba)

### pipx का उपयोग करना (अलग वातावरण)

```bash
pipx install kttc
```

लाभ:
- सिस्टम Python से अलग
- स्वचालित PATH कॉन्फ़िगरेशन
- आसान अपग्रेड

### स्रोत से (विकास)

```bash
git clone https://github.com/kttc-ai/kttc.git
cd kttc
pip install -e ".[dev]"
```

## वैकल्पिक निर्भरताएँ

### अंग्रेजी भाषा समर्थन

LanguageTool के साथ उन्नत अंग्रेजी व्याकरण जांच जोड़ता है:

```bash
pip install kttc[english]
```

**आवश्यकताएँ:**
- Java 17.0 या उच्चतर
- ~200MB डिस्क स्पेस

विशेषताएं:
- 5,000+ व्याकरण नियम
- विषय-क्रिया समझौता
- आर्टिकल जांच (a/an/the)
- प्रीपोज़िशन सत्यापन

### चीनी भाषा समर्थन

HanLP के साथ उन्नत चीनी NLP जोड़ता है:

```bash
pip install kttc[chinese]
```

विशेषताएं:
- माप शब्द सत्यापन (量词)
- आस्पेक्ट पार्टिकल जांच (了/过)
- उच्च-सटीकता POS टैगिंग (~95%)
- ~300MB मॉडल डाउनलोड पहले उपयोग पर

### हिंदी भाषा समर्थन

Indic NLP + Stanza + Spello के साथ उन्नत हिंदी NLP जोड़ता है:

```bash
pip install kttc[hindi]
```

विशेषताएं:
- देवनागरी लिपि प्रोसेसिंग
- हिंदी POS टैगिंग और NER
- वर्तनी जांच
- व्याकरण सत्यापन

### फ़ारसी भाषा समर्थन

DadmaTools के साथ उन्नत फ़ारसी NLP जोड़ता है:

```bash
pip install kttc[persian]
```

विशेषताएं:
- फ़ारसी लिपि प्रोसेसिंग
- POS टैगिंग और NER
- वर्तनी जांच
- व्याकरण सत्यापन

### सभी भाषा सहायक

सभी भाषा-विशिष्ट संवर्द्धन स्थापित करें:

```bash
pip install kttc[all-languages]
```

इसके बराबर है:

```bash
pip install kttc[english,chinese,hindi,persian]
```

## स्थापना सत्यापित करें

KTTC संस्करण जांचें:

```bash
kttc --version
```

डेमो मोड के साथ परीक्षण करें (API कुंजी की आवश्यकता नहीं):

```bash
echo "Hello" > source.txt
echo "नमस्ते" > trans.txt
kttc check source.txt trans.txt --source-lang en --target-lang hi --demo
```

## API कुंजी सेटअप

### OpenAI

```bash
export KTTC_OPENAI_API_KEY="sk-..."
```

### Anthropic

```bash
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

### स्थायी कॉन्फ़िगरेशन

अपनी शेल प्रोफ़ाइल (`~/.bashrc`, `~/.zshrc`) में जोड़ें:

```bash
# KTTC API कुंजी
export KTTC_OPENAI_API_KEY="sk-..."
export KTTC_ANTHROPIC_API_KEY="sk-ant-..."
```

फिर पुनः लोड करें:

```bash
source ~/.bashrc  # या ~/.zshrc
```

## अपग्रेड करना

### नवीनतम संस्करण में अपग्रेड करें

```bash
pip install --upgrade kttc
```

### वैकल्पिक निर्भरताओं के साथ अपग्रेड करें

```bash
pip install --upgrade kttc[all-languages,metrics]
```

## अनइंस्टॉल करना

```bash
pip uninstall kttc
```

## समस्या निवारण

### Python संस्करण मुद्दे

**त्रुटि:** `TypeError: unsupported operand type(s) for |`

**समाधान:** Python 3.11+ का उपयोग करें:

```bash
python3.11 -m pip install kttc
```

### अनुमति अस्वीकृत

**त्रुटि:** `ERROR: Could not install packages`

**समाधान:** उपयोगकर्ता स्थापना का उपयोग करें:

```bash
pip install --user kttc
```

या वर्चुअल वातावरण का उपयोग करें:

```bash
python3 -m venv venv
source venv/bin/activate
pip install kttc
```

## अगले चरण

- [कॉन्फ़िगरेशन](configuration.md) - सेटिंग्स कॉन्फ़िगर करें
- [CLI उपयोग](cli-usage.md) - कमांड सीखें
- [त्वरित शुरुआत](../tutorials/quickstart.md) - अपनी पहली जांच चलाएं
