# Migration Guide: Hybrid CLI Format

Welcome to KTTC's new **Hybrid Format**! This guide helps you migrate from the old CLI to the new smart `check` command.

## 📊 What Changed?

### Before (Multiple Commands)
```bash
kttc check source.txt translation.txt --source-lang en --target-lang ru
kttc compare --source source.txt --translation trans1.txt --translation trans2.txt ...
kttc batch --file translations.csv
```

### After (One Smart Command) ✨
```bash
kttc check source.txt translation.txt --source-lang en --target-lang ru        # → single
kttc check source.txt trans1.txt trans2.txt --source-lang en --target-lang ru  # → compare
kttc check translations.csv                                                    # → batch
```

---

## 🚀 New Smart Defaults

### 1. Smart Routing (Auto-Enabled)

**Before:**
```bash
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --smart-routing  # Had to enable manually
```

**After:**
```bash
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru
# ✅ Smart routing enabled by default!
# Saves money by using cheaper models for simple texts

# To disable:
kttc check ... --no-smart-routing
```

### 2. Auto-Glossary Detection

**Before:**
```bash
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --glossary base  # Had to specify manually
```

**After:**
```bash
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru
# ✅ Auto-detects glossaries/base.json if exists!

# To disable:
kttc check ... --glossary none
```

### 3. Auto-Format Detection

**Before:**
```bash
# Step 1: Generate JSON
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --output results.json --format json

# Step 2: Convert to HTML
kttc report results.json --format html --output report.html
```

**After:**
```bash
# One step! Format auto-detected from extension
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --output report.html
# ✅ Auto-detects HTML format!

# Also works for .md, .json
kttc check ... --output report.md   # → Markdown
kttc check ... --output report.json # → JSON
```

---

## 📋 Migration Examples

### Example 1: Compare Translations

**Old Way:**
```bash
kttc compare \
  --source source.txt \
  --translation trans1.txt \
  --translation trans2.txt \
  --translation trans3.txt \
  --source-lang en \
  --target-lang ru
```

**New Way:**
```bash
kttc check source.txt trans1.txt trans2.txt trans3.txt \
  --source-lang en --target-lang ru
# ✅ Auto-detects compare mode!
```

### Example 2: Batch Processing (CSV)

**Old Way:**
```bash
kttc batch --file translations.csv \
  --show-progress \
  --output report.json
```

**New Way:**
```bash
kttc check translations.csv
# ✅ Auto-detects batch mode!
# ✅ Progress shown by default
# ✅ Smart routing enabled by default
# ✅ Output to report.json by default
```

### Example 3: Batch Processing (Directories)

**Old Way:**
```bash
kttc batch \
  --source-dir ./source \
  --translation-dir ./translations \
  --source-lang en \
  --target-lang es \
  --parallel 4
```

**New Way:**
```bash
kttc check ./source ./translations \
  --source-lang en --target-lang es
# ✅ Auto-detects batch mode!
# ✅ Parallel=4 by default
```

### Example 4: HTML Report Generation

**Old Way (2 steps):**
```bash
# Step 1
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --output results.json --format json

# Step 2
kttc report results.json --format html --output report.html
```

**New Way (1 step):**
```bash
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --output report.html
# ✅ Auto-detects HTML from extension!
```

---

## 🔧 Backwards Compatibility

**Good news:** All old commands still work!

```bash
# ✅ Still works
kttc compare --source src.txt --translation trans1.txt --translation trans2.txt ...

# ✅ Still works
kttc batch --file translations.csv

# ✅ Still works
kttc batch --source-dir ./source --translation-dir ./translations ...
```

**But we recommend** using the new `check` command for simplicity.

---

## 💡 Quick Reference

| Old Command | New Command (Recommended) |
|-------------|---------------------------|
| `kttc check src.txt trans.txt ...` | `kttc check src.txt trans.txt ...` ✅ Same! |
| `kttc compare --source src.txt --translation t1.txt --translation t2.txt` | `kttc check src.txt t1.txt t2.txt` |
| `kttc batch --file data.csv` | `kttc check data.csv` |
| `kttc batch --source-dir s/ --translation-dir t/` | `kttc check s/ t/` |
| `kttc check ... --output r.json` + `kttc report r.json --format html` | `kttc check ... --output r.html` |
| `kttc check ... --smart-routing` | `kttc check ...` (enabled by default) |
| `kttc check ... --glossary base` | `kttc check ...` (auto-detected) |

---

## ❓ FAQ

### Q: Do I need to update my scripts?

**A:** No! Old commands still work. But new syntax is simpler.

### Q: Can I disable smart defaults?

**A:** Yes!
```bash
kttc check source.txt translation.txt \
  --source-lang en --target-lang ru \
  --no-smart-routing \  # Disable smart routing
  --glossary none       # Disable auto-glossary
```

### Q: How do I know which mode was detected?

**A:** Use `--verbose`:
```bash
kttc check ... --verbose

# Output:
# 🎯 Mode: compare
# 📚 Glossary: base
# 🧠 Smart routing: enabled
# 📄 Output format: html
```

### Q: What if I want the old behavior?

**A:** Just use the dedicated commands (`compare`, `batch`) - they still work!

---

## 🎯 Summary

### What to Do:
1. ✅ **Try the new syntax** - it's simpler!
2. ✅ **Remove `--smart-routing`** - it's default now
3. ✅ **Remove `--glossary base`** - auto-detected
4. ✅ **Use file extensions** for output format (.html, .md, .json)
5. ✅ **Simplify compare/batch** to just `check`

### What NOT to Do:
- ❌ Don't worry about breaking changes - old commands still work
- ❌ Don't rush - migrate at your own pace
- ❌ Don't manually specify what's auto-detected

---

## 🚀 Benefits

| Feature | Before | After |
|---------|--------|-------|
| Commands to remember | 3 (check, compare, batch) | 1 (check) |
| Smart routing | Manual (`--smart-routing`) | Auto-enabled |
| Glossary | Manual (`--glossary base`) | Auto-detected |
| Output format | Manual (`--format html`) + `report` command | Auto-detected from extension |
| Cost savings | Only if you remember `--smart-routing` | Always! |

---

## 📚 Next Steps

1. Try the new syntax with `--demo` mode (no API costs!)
2. Read updated [CLI README](cli/CLI_README.md)
3. Check [examples](../examples/cli/README.md)
4. Join the discussion on GitHub

---

**Happy migrating!** 🎉

The KTTC Team
